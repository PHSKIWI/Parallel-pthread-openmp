#include <vector>
#include <cstring>
#include <string>
#include <iostream>
#include <fstream>
#include <set>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <sys/time.h>
#include <omp.h>
#include "hnswlib/hnswlib/hnswlib.h"
#include "flat_scan.h"
// 可以自行添加需要的头文件
#include <arm_neon.h>
#include <queue>
#include <cstdint>
#include <cmath>
#include <functional>
#include <algorithm>
#include <limits>
//多线程实验
#include <pthread.h>
#include <atomic>

using namespace hnswlib;

//===========全局实验参数==============
//=================================================================
const size_t PQ_M = 8;
const size_t PQ_KS = 256;
const size_t PQ_TRAIN_SAMPLE = 2048;  // 训练 codebook 的采样数，太大构建会慢
const size_t PQ_ITERS = 4;            // KMeans 迭代次数，第一版先少一点
//多线程实验 参数设置
const int PTHREAD_NUM_THREADS = 4;
const int OMP_NUM_THREADS = 8;

// ===================== IVF 实验参数 =====================
const size_t IVF_NLIST = 64;          // 聚类簇数量，第一版先用 64，构建较快
const size_t IVF_TRAIN_SAMPLE = 4096; // 用于训练聚类中心的采样数
const size_t IVF_ITERS = 5;           // KMeans 迭代次数
const size_t IVF_NPROBE = 8;          // 查询时访问最近的多少个簇

// ===================== IVF-PQ 实验参数 =====================
const size_t IVFPQ_TOP_P = 1100;   // IVF 候选中 PQ 粗排后保留的候选数

// ===================== HNSW 实验参数 =====================
const int HNSW_M = 16;
const int HNSW_EF_CONSTRUCTION = 150;
const int HNSW_EF_SEARCH = 50;

// ===================== IVF-HNSW 实验参数 =====================
const int IVF_HNSW_M = 16;
const int IVF_HNSW_EF_CONSTRUCTION = 100;
const int IVF_HNSW_EF_SEARCH = 50;


template<typename T>
T *LoadData(std::string data_path, size_t& n, size_t& d)
{
    std::ifstream fin;
    fin.open(data_path, std::ios::in | std::ios::binary);
    fin.read((char*)&n,4);
    fin.read((char*)&d,4);
    T* data = new T[n*d];
    int sz = sizeof(T);
    for(int i = 0; i < n; ++i){
        fin.read(((char*)data + i*d*sz), d*sz);
    }
    fin.close();

    std::cerr<<"load data "<<data_path<<"\n";
    std::cerr<<"dimension: "<<d<<"  number:"<<n<<"  size_per_element:"<<sizeof(T)<<"\n";

    return data;
}

struct SearchResult
{
    float recall;
    int64_t latency; // 单位us
};

void build_index(float* base, size_t base_number, size_t vecdim)
{
    const int efConstruction = 150; // 为防止索引构建时间过长，efc建议设置200以下
    const int M = 16; // M建议设置为16以下

    HierarchicalNSW<float> *appr_alg;
    InnerProductSpace ipspace(vecdim);
    appr_alg = new HierarchicalNSW<float>(&ipspace, base_number, M, efConstruction);

    appr_alg->addPoint(base, 0);
    #pragma omp parallel for
    for(int i = 1; i < base_number; ++i) {
        appr_alg->addPoint(base + 1ll*vecdim*i, i);
    }

    char path_index[1024] = "files/hnsw.index";
    appr_alg->saveIndex(path_index);
}

// 使用 ARM NEON 加速内积计算
// DEEP100K 使用 IP 距离：dis = 1 - sum(base[d] * query[d])
static inline float inner_product_neon(const float* base_vec, const float* query, size_t vecdim)
{
    float32x4_t sum0 = vdupq_n_f32(0.0f);
    float32x4_t sum1 = vdupq_n_f32(0.0f);

    size_t d = 0;

    // 每轮处理 8 个 float，96 维向量正好循环 12 次
    for (; d + 7 < vecdim; d += 8) {
        float32x4_t b0 = vld1q_f32(base_vec + d);
        float32x4_t q0 = vld1q_f32(query + d);

        float32x4_t b1 = vld1q_f32(base_vec + d + 4);
        float32x4_t q1 = vld1q_f32(query + d + 4);

        sum0 = vmlaq_f32(sum0, b0, q0);
        sum1 = vmlaq_f32(sum1, b1, q1);
    }

    float32x4_t sum = vaddq_f32(sum0, sum1);

    float tmp[4];
    vst1q_f32(tmp, sum);

    float result = tmp[0] + tmp[1] + tmp[2] + tmp[3];

    // 处理剩余维度。DEEP100K 的 vecdim = 96，一般不会进入这里
    for (; d < vecdim; ++d) {
        result += base_vec[d] * query[d];
    }

    return result;
}


// Flat-NEON 搜索版本：返回值类型保持和 flat_search 完全一致
std::priority_queue<std::pair<float, uint32_t> > flat_search_neon(
    float* base,
    float* query,
    size_t base_number,
    size_t vecdim,
    size_t k
) {
    std::priority_queue<std::pair<float, uint32_t> > q;

    for (uint32_t i = 0; i < base_number; ++i) {
        const float* base_vec = base + (size_t)i * vecdim;

        float dis = 1.0f - inner_product_neon(base_vec, query, vecdim);

        if (q.size() < k) {
            q.push({dis, i});
        } else {
            if (dis < q.top().first) {
                q.pop();
                q.push({dis, i});
            }
        }
    }

    return q;
}

// ===================== SQ-SIMD 第一版：int8 粗排 + float 精排 =====================

// 计算全局对称量化 scale
static inline float compute_symmetric_scale(const float* data, size_t total)
{
    float max_abs = 0.0f;
    for (size_t i = 0; i < total; ++i) {
        float v = std::fabs(data[i]);
        if (v > max_abs) {
            max_abs = v;
        }
    }

    if (max_abs < 1e-12f) {
        return 1.0f;
    }

    return max_abs / 127.0f;
}


// 将单个 float 值量化为 int8
static inline int8_t quantize_one_float(float x, float scale)
{
    int v = (int)std::round(x / scale);

    if (v > 127) {
        v = 127;
    }
    if (v < -127) {
        v = -127;
    }

    return (int8_t)v;
}


// 量化整个 base 数据集，作为离线预处理
int8_t* quantize_base_int8(const float* base, size_t base_number, size_t vecdim, float base_scale)
{
    size_t total = base_number * vecdim;
    int8_t* base_i8 = new int8_t[total];

    for (size_t i = 0; i < total; ++i) {
        base_i8[i] = quantize_one_float(base[i], base_scale);
    }

    return base_i8;
}


// 每个 query 单独量化，使用 query 自身 scale
void quantize_query_int8(const float* query, size_t vecdim, int8_t* query_i8)
{
    float query_scale = compute_symmetric_scale(query, vecdim);

    for (size_t i = 0; i < vecdim; ++i) {
        query_i8[i] = quantize_one_float(query[i], query_scale);
    }
}


// int8 NEON 内积：每轮处理16个int8
static inline int32_t inner_product_i8_neon(const int8_t* base_vec, const int8_t* query, size_t vecdim)
{
    int32x4_t acc0 = vdupq_n_s32(0);
    int32x4_t acc1 = vdupq_n_s32(0);

    size_t d = 0;

    for (; d + 15 < vecdim; d += 16) {
        int8x16_t b = vld1q_s8(base_vec + d);
        int8x16_t q = vld1q_s8(query + d);

        int16x8_t mul_low = vmull_s8(vget_low_s8(b), vget_low_s8(q));
        int16x8_t mul_high = vmull_s8(vget_high_s8(b), vget_high_s8(q));

        acc0 = vpadalq_s16(acc0, mul_low);
        acc1 = vpadalq_s16(acc1, mul_high);
    }

    int32x4_t acc = vaddq_s32(acc0, acc1);

    int32_t tmp[4];
    vst1q_s32(tmp, acc);

    int32_t result = tmp[0] + tmp[1] + tmp[2] + tmp[3];

    for (; d < vecdim; ++d) {
        result += (int32_t)base_vec[d] * (int32_t)query[d];
    }

    return result;
}


// SQ 搜索：int8粗排Top-p，再用原始float NEON精排Top-k
std::priority_queue<std::pair<float, uint32_t> > flat_search_sq(
    float* base,
    int8_t* base_i8,
    float* query,
    size_t base_number,
    size_t vecdim,
    size_t k,
    size_t top_p
) {
    if (top_p < k) {
        top_p = k;
    }
    if (top_p > base_number) {
        top_p = base_number;
    }

    // 1. 量化 query
    std::vector<int8_t> query_i8(vecdim);
    quantize_query_int8(query, vecdim, query_i8.data());

    // 2. int8 粗排，分数越大表示内积越大、距离越近
    typedef std::pair<int32_t, uint32_t> ScoreId;
    std::priority_queue<ScoreId, std::vector<ScoreId>, std::greater<ScoreId> > coarse_heap;

    for (uint32_t i = 0; i < base_number; ++i) {
        const int8_t* base_vec_i8 = base_i8 + (size_t)i * vecdim;
        int32_t score = inner_product_i8_neon(base_vec_i8, query_i8.data(), vecdim);

        if (coarse_heap.size() < top_p) {
            coarse_heap.push({score, i});
        } else {
            if (score > coarse_heap.top().first) {
                coarse_heap.pop();
                coarse_heap.push({score, i});
            }
        }
    }

    // 3. 收集粗排候选
    std::vector<uint32_t> candidates;
    candidates.reserve(top_p);

    while (!coarse_heap.empty()) {
        candidates.push_back(coarse_heap.top().second);
        coarse_heap.pop();
    }

    // 4. 对 Top-p 候选使用原始 float NEON 精排
    std::priority_queue<std::pair<float, uint32_t> > final_q;

    for (size_t t = 0; t < candidates.size(); ++t) {
        uint32_t idx = candidates[t];
        const float* base_vec = base + (size_t)idx * vecdim;

        float dis = 1.0f - inner_product_neon(base_vec, query, vecdim);

        if (final_q.size() < k) {
            final_q.push({dis, idx});
        } else {
            if (dis < final_q.top().first) {
                final_q.pop();
                final_q.push({dis, idx});
            }
        }
    }

    return final_q;
}

// ===================== PQ-ADC 第一版 =====================
// 维度 vecdim = 96，这里切成 4 个子空间，每个子空间 24 维
// Ks = 256，表示每个子空间有 256 个聚类中心，可以用 uint8_t 编码


struct PQIndex {
    size_t M;
    size_t Ks;
    size_t subdim;

    // codebook[(m * Ks + c) * subdim + d]
    std::vector<float> codebook;

    // AoS布局：codes[i * M + m]
    // 第 i 个向量的 M 个 code 连续存储
    std::vector<uint8_t> codes;

    // SoA布局：codes_soa[m * base_number + i]
    // 第 m 个子空间下，所有 base 向量的 code 连续存储
    // 用于优化 ADC 查表累加阶段
    std::vector<uint8_t> codes_soa;
};
static inline float l2_distance_neon(const float* a, const float* b, size_t dim)
{
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);

    size_t d = 0;

    for (; d + 7 < dim; d += 8) {
        float32x4_t a0 = vld1q_f32(a + d);
        float32x4_t b0 = vld1q_f32(b + d);
        float32x4_t diff0 = vsubq_f32(a0, b0);
        acc0 = vmlaq_f32(acc0, diff0, diff0);

        float32x4_t a1 = vld1q_f32(a + d + 4);
        float32x4_t b1 = vld1q_f32(b + d + 4);
        float32x4_t diff1 = vsubq_f32(a1, b1);
        acc1 = vmlaq_f32(acc1, diff1, diff1);
    }

    float32x4_t acc = vaddq_f32(acc0, acc1);

    float tmp[4];
    vst1q_f32(tmp, acc);

    float result = tmp[0] + tmp[1] + tmp[2] + tmp[3];

    for (; d < dim; ++d) {
        float diff = a[d] - b[d];
        result += diff * diff;
    }

    return result;
}


static inline uint8_t nearest_centroid_l2(
    const float* vec,
    const float* centroids,
    size_t Ks,
    size_t subdim
) {
    float best_dist = std::numeric_limits<float>::max();
    uint32_t best_id = 0;

    for (uint32_t c = 0; c < Ks; ++c) {
        const float* center = centroids + (size_t)c * subdim;
        float dist = l2_distance_neon(vec, center, subdim);

        if (dist < best_dist) {
            best_dist = dist;
            best_id = c;
        }
    }

    return (uint8_t)best_id;
}

void train_pq_codebooks(
    float* base,
    size_t base_number,
    size_t vecdim,
    PQIndex& pq
) {
    pq.M = PQ_M;
    pq.Ks = PQ_KS;
    pq.subdim = vecdim / PQ_M;

    pq.codebook.resize(pq.M * pq.Ks * pq.subdim);

    size_t sample_n = std::min(PQ_TRAIN_SAMPLE, base_number);

    std::vector<uint32_t> sample_ids(sample_n);
    for (size_t i = 0; i < sample_n; ++i) {
        sample_ids[i] = (uint32_t)((i * base_number) / sample_n);
    }

    for (size_t m = 0; m < pq.M; ++m) {
        float* centroids = pq.codebook.data() + (m * pq.Ks * pq.subdim);

        // 初始化中心：从样本中均匀选取 Ks 个子向量
        for (size_t c = 0; c < pq.Ks; ++c) {
            size_t sample_pos = (c * sample_n) / pq.Ks;
            uint32_t base_id = sample_ids[sample_pos];

            const float* src = base + (size_t)base_id * vecdim + m * pq.subdim;
            float* dst = centroids + c * pq.subdim;

            std::memcpy(dst, src, pq.subdim * sizeof(float));
        }

        // KMeans 迭代
        for (size_t iter = 0; iter < PQ_ITERS; ++iter) {
            std::vector<float> sums(pq.Ks * pq.subdim, 0.0f);
            std::vector<int> counts(pq.Ks, 0);

            for (size_t s = 0; s < sample_n; ++s) {
                uint32_t base_id = sample_ids[s];
                const float* vec = base + (size_t)base_id * vecdim + m * pq.subdim;

                uint8_t cid = nearest_centroid_l2(vec, centroids, pq.Ks, pq.subdim);

                float* sum_ptr = sums.data() + (size_t)cid * pq.subdim;
                for (size_t d = 0; d < pq.subdim; ++d) {
                    sum_ptr[d] += vec[d];
                }

                counts[cid]++;
            }

            for (size_t c = 0; c < pq.Ks; ++c) {
                if (counts[c] == 0) {
                    continue;
                }

                float* center = centroids + c * pq.subdim;
                float* sum_ptr = sums.data() + c * pq.subdim;

                float inv_count = 1.0f / counts[c];
                for (size_t d = 0; d < pq.subdim; ++d) {
                    center[d] = sum_ptr[d] * inv_count;
                }
            }
        }

        std::cerr << "PQ train subspace " << m << " finished\n";
    }
}

void encode_base_pq(
    float* base,
    size_t base_number,
    size_t vecdim,
    PQIndex& pq
) {
    pq.codes.resize(base_number * pq.M);

    #pragma omp parallel for
    for (int i = 0; i < (int)base_number; ++i) {
        for (size_t m = 0; m < pq.M; ++m) {
            const float* vec = base + (size_t)i * vecdim + m * pq.subdim;
            const float* centroids = pq.codebook.data() + m * pq.Ks * pq.subdim;

            uint8_t cid = nearest_centroid_l2(vec, centroids, pq.Ks, pq.subdim);
            pq.codes[(size_t)i * pq.M + m] = cid;
        }
    }

    std::cerr << "PQ base encoding finished\n";
}

void build_codes_soa(
    PQIndex& pq,
    size_t base_number
) {
    pq.codes_soa.resize(pq.M * base_number);

    #pragma omp parallel for
    for (int i = 0; i < (int)base_number; ++i) {
        for (size_t m = 0; m < pq.M; ++m) {
            pq.codes_soa[m * base_number + (size_t)i] =
                pq.codes[(size_t)i * pq.M + m];
        }
    }

    std::cerr << "PQ codes SoA layout finished\n";
}

void build_pq_index(
    float* base,
    size_t base_number,
    size_t vecdim,
    PQIndex& pq
) {
    if (vecdim % PQ_M != 0) {
        std::cerr << "vecdim must be divisible by PQ_M\n";
        return;
    }

    train_pq_codebooks(base, base_number, vecdim, pq);
    encode_base_pq(base, base_number, vecdim, pq);
    build_codes_soa(pq, base_number);

    std::cerr << "PQ index finished: M=" << pq.M
              << " Ks=" << pq.Ks
              << " subdim=" << pq.subdim << "\n";
}

void build_adc_lut(
    float* query,
    const PQIndex& pq,
    std::vector<float>& lut
) {
    lut.resize(pq.M * pq.Ks);

    for (size_t m = 0; m < pq.M; ++m) {
        const float* query_sub = query + m * pq.subdim;
        const float* centroids = pq.codebook.data() + m * pq.Ks * pq.subdim;

        for (size_t c = 0; c < pq.Ks; ++c) {
            const float* center = centroids + c * pq.subdim;

            // 使用内积作为近似相似度，分数越大越好
            float score = inner_product_neon(center, query_sub, pq.subdim);
            lut[m * pq.Ks + c] = score;
        }
    }
}


std::priority_queue<std::pair<float, uint32_t> > flat_search_pq_adc(
    float* base,
    float* query,
    size_t base_number,
    size_t vecdim,
    size_t k,
    size_t top_p,
    const PQIndex& pq
) {
    if (top_p < k) {
        top_p = k;
    }

    if (top_p > base_number) {
        top_p = base_number;
    }

    // 1. 构建 ADC LUT：M × Ks
    std::vector<float> lut;
    build_adc_lut(query, pq, lut);

    // 2. 查表粗排：分数越大越好
    typedef std::pair<float, uint32_t> ScoreId;
    std::priority_queue<ScoreId, std::vector<ScoreId>, std::greater<ScoreId> > coarse_heap;

    for (uint32_t i = 0; i < base_number; ++i) {
        float score = 0.0f;

        const uint8_t* code = pq.codes.data() + (size_t)i * pq.M;

        for (size_t m = 0; m < pq.M; ++m) {
            uint8_t cid = code[m];
            score += lut[m * pq.Ks + cid];
        }

        if (coarse_heap.size() < top_p) {
            coarse_heap.push({score, i});
        } else {
            if (score > coarse_heap.top().first) {
                coarse_heap.pop();
                coarse_heap.push({score, i});
            }
        }
    }

    // 3. 收集 Top-p 候选
    std::vector<uint32_t> candidates;
    candidates.reserve(top_p);

    while (!coarse_heap.empty()) {
        candidates.push_back(coarse_heap.top().second);
        coarse_heap.pop();
    }

    // 4. 用原始 float NEON 精排，返回最终 Top-k
    std::priority_queue<std::pair<float, uint32_t> > final_q;

    for (size_t t = 0; t < candidates.size(); ++t) {
        uint32_t idx = candidates[t];
        const float* base_vec = base + (size_t)idx * vecdim;

        float dis = 1.0f - inner_product_neon(base_vec, query, vecdim);

        if (final_q.size() < k) {
            final_q.push({dis, idx});
        } else {
            if (dis < final_q.top().first) {
                final_q.pop();
                final_q.push({dis, idx});
            }
        }
    }

    return final_q;
}

// ===================== PQ ADC 查表累加优化版：SoA + block =====================
// 优化目标：
// 1. 使用 codes_soa[m * base_number + i]，让同一子空间下的 code 连续访问
// 2. 每次处理一个 block 的多个 base 向量，提高缓存友好性
// 3. 仍然保留 Top-p 粗排 + float NEON 精排流程
std::priority_queue<std::pair<float, uint32_t> > flat_search_pq_adc_soa_block(
    float* base,
    float* query,
    size_t base_number,
    size_t vecdim,
    size_t k,
    size_t top_p,
    const PQIndex& pq
) {
    if (top_p < k) {
        top_p = k;
    }

    if (top_p > base_number) {
        top_p = base_number;
    }

    // 1. 构建 ADC LUT：M × Ks
    std::vector<float> lut;
    build_adc_lut(query, pq, lut);

    typedef std::pair<float, uint32_t> ScoreId;
    std::priority_queue<ScoreId, std::vector<ScoreId>, std::greater<ScoreId> > coarse_heap;

    const size_t BLOCK = 64;
    float block_score[BLOCK];

    // 2. block 化查表累加
    for (size_t start = 0; start < base_number; start += BLOCK) {
        size_t block_size = std::min(BLOCK, base_number - start);

        for (size_t t = 0; t < block_size; ++t) {
            block_score[t] = 0.0f;
        }

        // 外层遍历子空间 m
        // 对每个 m，codes_soa[m * base_number + start ... start+block_size) 连续访问
        for (size_t m = 0; m < pq.M; ++m) {
            const uint8_t* code_m = pq.codes_soa.data() + m * base_number + start;
            const float* lut_m = lut.data() + m * pq.Ks;

            for (size_t t = 0; t < block_size; ++t) {
                uint8_t cid = code_m[t];
                block_score[t] += lut_m[cid];
            }
        }

        // 3. 用 block 内分数更新 Top-p 小根堆
        for (size_t t = 0; t < block_size; ++t) {
            uint32_t idx = (uint32_t)(start + t);
            float score = block_score[t];

            if (coarse_heap.size() < top_p) {
                coarse_heap.push({score, idx});
            } else {
                if (score > coarse_heap.top().first) {
                    coarse_heap.pop();
                    coarse_heap.push({score, idx});
                }
            }
        }
    }

    // 4. 收集 Top-p 候选
    std::vector<uint32_t> candidates;
    candidates.reserve(top_p);

    while (!coarse_heap.empty()) {
        candidates.push_back(coarse_heap.top().second);
        coarse_heap.pop();
    }

    // 5. 用原始 float NEON 精排，返回最终 Top-k
    std::priority_queue<std::pair<float, uint32_t> > final_q;

    for (size_t t = 0; t < candidates.size(); ++t) {
        uint32_t idx = candidates[t];
        const float* base_vec = base + (size_t)idx * vecdim;

        float dis = 1.0f - inner_product_neon(base_vec, query, vecdim);

        if (final_q.size() < k) {
            final_q.push({dis, idx});
        } else {
            if (dis < final_q.top().first) {
                final_q.pop();
                final_q.push({dis, idx});
            }
        }
    }

    return final_q;
}

//=============多线程实验优化==================

// ===================== PQ ADC 优化版：SoA + block + Pthread 跨向量并行 =====================
// 优化思路：
// 1. 每个线程负责 base 向量的一段连续区间；
// 2. 每个线程独立进行 PQ-ADC 查表累加；
// 3. 每个线程维护自己的局部 Top-p 小根堆，避免多线程竞争同一个全局堆；
// 4. 主线程 pthread_join 后合并所有局部 Top-p，得到全局 Top-p；
// 5. 最后仍然使用原始 float NEON 对 Top-p 候选进行精排，保证最终 Top-k 质量。

typedef std::pair<float, uint32_t> PQScoreId;
typedef std::priority_queue<
    PQScoreId,
    std::vector<PQScoreId>,
    std::greater<PQScoreId>
> PQMinHeap;

struct PQPthreadParam {
    int tid;
    int num_threads;

    size_t base_number;
    size_t top_p;

    const PQIndex* pq;
    const std::vector<float>* lut;

    std::vector<PQScoreId>* local_candidates;
};

void* pq_adc_worker(void* arg)
{
    PQPthreadParam* param = (PQPthreadParam*)arg;

    int tid = param->tid;
    int num_threads = param->num_threads;
    size_t base_number = param->base_number;
    size_t top_p = param->top_p;

    const PQIndex& pq = *(param->pq);
    const std::vector<float>& lut = *(param->lut);

    // 静态连续块划分：每个线程负责 [begin, end)
    size_t begin = base_number * (size_t)tid / (size_t)num_threads;
    size_t end = base_number * (size_t)(tid + 1) / (size_t)num_threads;

    const size_t BLOCK = 64;
    float block_score[BLOCK];

    PQMinHeap local_heap;

    for (size_t start = begin; start < end; start += BLOCK) {
        size_t block_size = std::min(BLOCK, end - start);

        for (size_t t = 0; t < block_size; ++t) {
            block_score[t] = 0.0f;
        }

        // SoA + block 查表累加
        for (size_t m = 0; m < pq.M; ++m) {
            const uint8_t* code_m = pq.codes_soa.data() + m * base_number + start;
            const float* lut_m = lut.data() + m * pq.Ks;

            for (size_t t = 0; t < block_size; ++t) {
                uint8_t cid = code_m[t];
                block_score[t] += lut_m[cid];
            }
        }

        // 更新线程局部 Top-p
        for (size_t t = 0; t < block_size; ++t) {
            uint32_t idx = (uint32_t)(start + t);
            float score = block_score[t];

            if (local_heap.size() < top_p) {
                local_heap.push({score, idx});
            } else {
                if (score > local_heap.top().first) {
                    local_heap.pop();
                    local_heap.push({score, idx});
                }
            }
        }
    }

    // 把局部堆中的候选复制到该线程对应的 local_candidates[tid]
    std::vector<PQScoreId>& out = param->local_candidates[tid];
    out.reserve(local_heap.size());

    while (!local_heap.empty()) {
        out.push_back(local_heap.top());
        local_heap.pop();
    }

    pthread_exit(NULL);
}


std::priority_queue<std::pair<float, uint32_t> > flat_search_pq_adc_soa_block_pthread(
    float* base,
    float* query,
    size_t base_number,
    size_t vecdim,
    size_t k,
    size_t top_p,
    const PQIndex& pq
) {
    if (top_p < k) {
        top_p = k;
    }

    if (top_p > base_number) {
        top_p = base_number;
    }

    // 1. 构建 ADC LUT：M × Ks
    std::vector<float> lut;
    build_adc_lut(query, pq, lut);

    int num_threads = PTHREAD_NUM_THREADS;
    if (num_threads < 1) {
        num_threads = 1;
    }
    if ((size_t)num_threads > base_number) {
        num_threads = (int)base_number;
    }

    std::vector<pthread_t> threads(num_threads);
    std::vector<PQPthreadParam> params(num_threads);
    std::vector<std::vector<PQScoreId> > local_candidates(num_threads);

    // 2. 创建线程
    for (int t = 0; t < num_threads; ++t) {
        params[t].tid = t;
        params[t].num_threads = num_threads;
        params[t].base_number = base_number;
        params[t].top_p = top_p;
        params[t].pq = &pq;
        params[t].lut = &lut;
        params[t].local_candidates = local_candidates.data();

        pthread_create(&threads[t], NULL, pq_adc_worker, (void*)&params[t]);
    }

    // 3. 等待所有线程完成
    for (int t = 0; t < num_threads; ++t) {
        pthread_join(threads[t], NULL);
    }

    // 4. 合并各线程局部 Top-p，得到全局 Top-p
    PQMinHeap global_heap;

    for (int t = 0; t < num_threads; ++t) {
        for (size_t i = 0; i < local_candidates[t].size(); ++i) {
            const PQScoreId& item = local_candidates[t][i];

            if (global_heap.size() < top_p) {
                global_heap.push(item);
            } else {
                if (item.first > global_heap.top().first) {
                    global_heap.pop();
                    global_heap.push(item);
                }
            }
        }
    }

    std::vector<uint32_t> candidates;
    candidates.reserve(top_p);

    while (!global_heap.empty()) {
        candidates.push_back(global_heap.top().second);
        global_heap.pop();
    }

    // 5. 使用原始 float NEON 对 Top-p 候选精排
    std::priority_queue<std::pair<float, uint32_t> > final_q;

    for (size_t t = 0; t < candidates.size(); ++t) {
        uint32_t idx = candidates[t];
        const float* base_vec = base + (size_t)idx * vecdim;

        float dis = 1.0f - inner_product_neon(base_vec, query, vecdim);

        if (final_q.size() < k) {
            final_q.push({dis, idx});
        } else {
            if (dis < final_q.top().first) {
                final_q.pop();
                final_q.push({dis, idx});
            }
        }
    }

    return final_q;
}

// ===================== PQ ADC 优化版：SoA + block + 静态 Pthread 线程池 =====================
// 与动态 Pthread 版本相比：
// 1. 线程只在查询循环前创建一次；
// 2. 每个 query 通过 barrier 唤醒工作线程；
// 3. 工作线程完成本轮扫描后再次 barrier 同步；
// 4. 全部 query 结束后统一销毁线程；
// 5. 避免每个 query 反复 pthread_create / pthread_join 的开销。

struct PQPthreadStaticPool;

struct PQPthreadStaticParam {
    int tid;
    PQPthreadStaticPool* pool;
};

struct PQPthreadStaticPool {
    int num_threads;

    float* base;
    size_t base_number;
    size_t vecdim;
    size_t k;
    size_t top_p;

    const PQIndex* pq;

    const float* current_query;
    std::vector<float> lut;

    std::atomic<bool> stop;

    pthread_barrier_t start_barrier;
    pthread_barrier_t end_barrier;

    std::vector<pthread_t> threads;
    std::vector<PQPthreadStaticParam> params;
    std::vector<std::vector<PQScoreId> > local_candidates;
};


void* pq_adc_static_worker(void* arg)
{
    PQPthreadStaticParam* param = (PQPthreadStaticParam*)arg;
    int tid = param->tid;
    PQPthreadStaticPool* pool = param->pool;

    while (true) {
        // 等待主线程发布本轮 query 任务
        pthread_barrier_wait(&pool->start_barrier);

        // 收到结束信号，退出线程
        if (pool->stop.load()) {
            break;
        }

        const PQIndex& pq = *(pool->pq);
        const std::vector<float>& lut = pool->lut;

        size_t base_number = pool->base_number;
        size_t top_p = pool->top_p;
        int num_threads = pool->num_threads;

        // 静态连续块划分：[begin, end)
        size_t begin = base_number * (size_t)tid / (size_t)num_threads;
        size_t end = base_number * (size_t)(tid + 1) / (size_t)num_threads;

        const size_t BLOCK = 128;
        float block_score[BLOCK];

        PQMinHeap local_heap;

        for (size_t start = begin; start < end; start += BLOCK) {
            size_t block_size = std::min(BLOCK, end - start);

            for (size_t t = 0; t < block_size; ++t) {
                block_score[t] = 0.0f;
            }

            // SoA + block 查表累加
            for (size_t m = 0; m < pq.M; ++m) {
                const uint8_t* code_m = pq.codes_soa.data() + m * base_number + start;
                const float* lut_m = lut.data() + m * pq.Ks;

                for (size_t t = 0; t < block_size; ++t) {
                    uint8_t cid = code_m[t];
                    block_score[t] += lut_m[cid];
                }
            }

            // 更新线程局部 Top-p
            for (size_t t = 0; t < block_size; ++t) {
                uint32_t idx = (uint32_t)(start + t);
                float score = block_score[t];

                if (local_heap.size() < top_p) {
                    local_heap.push({score, idx});
                } else {
                    if (score > local_heap.top().first) {
                        local_heap.pop();
                        local_heap.push({score, idx});
                    }
                }
            }
        }

        // 保存该线程的局部 Top-p
        std::vector<PQScoreId>& out = pool->local_candidates[tid];
        out.clear();
        out.reserve(local_heap.size());

        while (!local_heap.empty()) {
            out.push_back(local_heap.top());
            local_heap.pop();
        }

        // 通知主线程：本轮 query 的粗排扫描已经完成
        pthread_barrier_wait(&pool->end_barrier);
    }

    return NULL;
}


void init_pq_pthread_static_pool(
    PQPthreadStaticPool& pool,
    float* base,
    size_t base_number,
    size_t vecdim,
    size_t k,
    size_t top_p,
    const PQIndex& pq,
    int num_threads
) {
    if (top_p < k) {
        top_p = k;
    }
    if (top_p > base_number) {
        top_p = base_number;
    }

    if (num_threads < 1) {
        num_threads = 1;
    }
    if ((size_t)num_threads > base_number) {
        num_threads = (int)base_number;
    }

    pool.num_threads = num_threads;
    pool.base = base;
    pool.base_number = base_number;
    pool.vecdim = vecdim;
    pool.k = k;
    pool.top_p = top_p;
    pool.pq = &pq;
    pool.current_query = NULL;
    pool.stop.store(false);

    pool.threads.resize(num_threads);
    pool.params.resize(num_threads);
    pool.local_candidates.resize(num_threads);

    // barrier 数量为 num_threads + 1，因为主线程也参与同步
    pthread_barrier_init(&pool.start_barrier, NULL, num_threads + 1);
    pthread_barrier_init(&pool.end_barrier, NULL, num_threads + 1);

    for (int t = 0; t < num_threads; ++t) {
        pool.params[t].tid = t;
        pool.params[t].pool = &pool;
        pthread_create(&pool.threads[t], NULL, pq_adc_static_worker, (void*)&pool.params[t]);
    }
}


std::priority_queue<std::pair<float, uint32_t> > flat_search_pq_adc_soa_block_pthread_static(
    PQPthreadStaticPool& pool,
    const float* query
) {
    pool.current_query = query;

    // 1. 主线程构建本轮 query 的 ADC LUT
    build_adc_lut((float*)query, *(pool.pq), pool.lut);

    for (int t = 0; t < pool.num_threads; ++t) {
        pool.local_candidates[t].clear();
    }

    // 2. 唤醒所有工作线程开始扫描
    pthread_barrier_wait(&pool.start_barrier);

    // 3. 等待所有工作线程完成扫描
    pthread_barrier_wait(&pool.end_barrier);

    // 4. 合并各线程局部 Top-p，得到全局 Top-p
    PQMinHeap global_heap;

    for (int t = 0; t < pool.num_threads; ++t) {
        for (size_t i = 0; i < pool.local_candidates[t].size(); ++i) {
            const PQScoreId& item = pool.local_candidates[t][i];

            if (global_heap.size() < pool.top_p) {
                global_heap.push(item);
            } else {
                if (item.first > global_heap.top().first) {
                    global_heap.pop();
                    global_heap.push(item);
                }
            }
        }
    }

    std::vector<uint32_t> candidates;
    candidates.reserve(pool.top_p);

    while (!global_heap.empty()) {
        candidates.push_back(global_heap.top().second);
        global_heap.pop();
    }

    // 5. 原始 float NEON 精排
    std::priority_queue<std::pair<float, uint32_t> > final_q;

    for (size_t t = 0; t < candidates.size(); ++t) {
        uint32_t idx = candidates[t];
        const float* base_vec = pool.base + (size_t)idx * pool.vecdim;

        float dis = 1.0f - inner_product_neon(base_vec, query, pool.vecdim);

        if (final_q.size() < pool.k) {
            final_q.push({dis, idx});
        } else {
            if (dis < final_q.top().first) {
                final_q.pop();
                final_q.push({dis, idx});
            }
        }
    }

    return final_q;
}


void destroy_pq_pthread_static_pool(PQPthreadStaticPool& pool)
{
    // 发出终止信号
    pool.stop.store(true);

    // 唤醒所有阻塞在 start_barrier 的工作线程
    pthread_barrier_wait(&pool.start_barrier);

    for (int t = 0; t < pool.num_threads; ++t) {
        pthread_join(pool.threads[t], NULL);
    }

    pthread_barrier_destroy(&pool.start_barrier);
    pthread_barrier_destroy(&pool.end_barrier);
}

// ===================== PQ ADC 优化版：SoA + block + OpenMP 跨向量并行 =====================
// 优化思路：
// 1. 构建每个 query 的 ADC LUT；
// 2. 使用 OpenMP 将 base 向量 block 划分给多个线程；
// 3. 每个线程维护自己的局部 Top-p 小根堆，避免多个线程竞争全局堆；
// 4. 并行区域结束后，主线程合并所有局部 Top-p；
// 5. 最后使用原始 float NEON 对 Top-p 候选精排，保证最终 Top-k 质量。

std::priority_queue<std::pair<float, uint32_t> > flat_search_pq_adc_soa_block_openmp(
    float* base,
    float* query,
    size_t base_number,
    size_t vecdim,
    size_t k,
    size_t top_p,
    const PQIndex& pq
) {
    if (top_p < k) {
        top_p = k;
    }

    if (top_p > base_number) {
        top_p = base_number;
    }

    // 1. 构建 ADC LUT：M × Ks
    std::vector<float> lut;
    build_adc_lut(query, pq, lut);

    int num_threads = OMP_NUM_THREADS;
    if (num_threads < 1) {
        num_threads = 1;
    }
    if ((size_t)num_threads > base_number) {
        num_threads = (int)base_number;
    }

    std::vector<std::vector<PQScoreId> > local_candidates(num_threads);

    const size_t BLOCK = 128;
    size_t num_blocks = (base_number + BLOCK - 1) / BLOCK;

    #pragma omp parallel num_threads(num_threads)
    {
        int tid = omp_get_thread_num();

        PQMinHeap local_heap;
        float block_score[BLOCK];

        #pragma omp for schedule(static)
        for (long long block_id = 0; block_id < (long long)num_blocks; ++block_id) {
            size_t start = (size_t)block_id * BLOCK;
            size_t block_size = std::min(BLOCK, base_number - start);

            for (size_t t = 0; t < block_size; ++t) {
                block_score[t] = 0.0f;
            }

            // SoA + block 查表累加
            for (size_t m = 0; m < pq.M; ++m) {
                const uint8_t* code_m = pq.codes_soa.data() + m * base_number + start;
                const float* lut_m = lut.data() + m * pq.Ks;

                for (size_t t = 0; t < block_size; ++t) {
                    uint8_t cid = code_m[t];
                    block_score[t] += lut_m[cid];
                }
            }

            // 更新线程局部 Top-p
            for (size_t t = 0; t < block_size; ++t) {
                uint32_t idx = (uint32_t)(start + t);
                float score = block_score[t];

                if (local_heap.size() < top_p) {
                    local_heap.push({score, idx});
                } else {
                    if (score > local_heap.top().first) {
                        local_heap.pop();
                        local_heap.push({score, idx});
                    }
                }
            }
        }

        local_candidates[tid].reserve(local_heap.size());
        while (!local_heap.empty()) {
            local_candidates[tid].push_back(local_heap.top());
            local_heap.pop();
        }
    }

    // 2. 合并各线程局部 Top-p，得到全局 Top-p
    PQMinHeap global_heap;

    for (int tid = 0; tid < num_threads; ++tid) {
        for (size_t i = 0; i < local_candidates[tid].size(); ++i) {
            const PQScoreId& item = local_candidates[tid][i];

            if (global_heap.size() < top_p) {
                global_heap.push(item);
            } else {
                if (item.first > global_heap.top().first) {
                    global_heap.pop();
                    global_heap.push(item);
                }
            }
        }
    }

    std::vector<uint32_t> candidates;
    candidates.reserve(top_p);

    while (!global_heap.empty()) {
        candidates.push_back(global_heap.top().second);
        global_heap.pop();
    }

    // 3. 原始 float NEON 精排
    std::priority_queue<std::pair<float, uint32_t> > final_q;

    for (size_t t = 0; t < candidates.size(); ++t) {
        uint32_t idx = candidates[t];
        const float* base_vec = base + (size_t)idx * vecdim;

        float dis = 1.0f - inner_product_neon(base_vec, query, vecdim);

        if (final_q.size() < k) {
            final_q.push({dis, idx});
        } else {
            if (dis < final_q.top().first) {
                final_q.pop();
                final_q.push({dis, idx});
            }
        }
    }

    return final_q;
}


// ===================== Flat-SIMD + 静态 Pthread 线程池 =====================
// 优化思路：
// 1. 每个线程负责 base 向量的一段连续区间；
// 2. 每个线程内部仍然调用 inner_product_neon 进行 SIMD 内积计算；
// 3. 每个线程维护局部 Top-k；
// 4. 主线程合并所有局部 Top-k，得到最终 Top-k；
// 5. 该版本用于验证 Flat-SIMD 与线程级并行结合的效果。

struct FlatPthreadStaticPool;

struct FlatPthreadStaticParam {
    int tid;
    FlatPthreadStaticPool* pool;
};

struct FlatPthreadStaticPool {
    int num_threads;

    float* base;
    size_t base_number;
    size_t vecdim;
    size_t k;

    const float* current_query;

    std::atomic<bool> stop;

    pthread_barrier_t start_barrier;
    pthread_barrier_t end_barrier;

    std::vector<pthread_t> threads;
    std::vector<FlatPthreadStaticParam> params;

    // 每个线程保存自己的局部 Top-k 候选，pair<distance, id>
    std::vector<std::vector<std::pair<float, uint32_t> > > local_candidates;
};


void* flat_static_worker(void* arg)
{
    FlatPthreadStaticParam* param = (FlatPthreadStaticParam*)arg;
    int tid = param->tid;
    FlatPthreadStaticPool* pool = param->pool;

    while (true) {
        // 等待主线程发布本轮 query
        pthread_barrier_wait(&pool->start_barrier);

        if (pool->stop.load()) {
            break;
        }

        const float* query = pool->current_query;
        float* base = pool->base;
        size_t base_number = pool->base_number;
        size_t vecdim = pool->vecdim;
        size_t k = pool->k;
        int num_threads = pool->num_threads;

        size_t begin = base_number * (size_t)tid / (size_t)num_threads;
        size_t end = base_number * (size_t)(tid + 1) / (size_t)num_threads;

        // 大根堆：堆顶是当前 Top-k 中距离最大的元素，便于淘汰
        std::priority_queue<std::pair<float, uint32_t> > local_heap;

        for (uint32_t i = (uint32_t)begin; i < (uint32_t)end; ++i) {
            const float* base_vec = base + (size_t)i * vecdim;

            float dis = 1.0f - inner_product_neon(base_vec, query, vecdim);

            if (local_heap.size() < k) {
                local_heap.push({dis, i});
            } else {
                if (dis < local_heap.top().first) {
                    local_heap.pop();
                    local_heap.push({dis, i});
                }
            }
        }

        std::vector<std::pair<float, uint32_t> >& out = pool->local_candidates[tid];
        out.clear();
        out.reserve(local_heap.size());

        while (!local_heap.empty()) {
            out.push_back(local_heap.top());
            local_heap.pop();
        }

        pthread_barrier_wait(&pool->end_barrier);
    }

    return NULL;
}


void init_flat_pthread_static_pool(
    FlatPthreadStaticPool& pool,
    float* base,
    size_t base_number,
    size_t vecdim,
    size_t k,
    int num_threads
) {
    if (num_threads < 1) {
        num_threads = 1;
    }
    if ((size_t)num_threads > base_number) {
        num_threads = (int)base_number;
    }

    pool.num_threads = num_threads;
    pool.base = base;
    pool.base_number = base_number;
    pool.vecdim = vecdim;
    pool.k = k;
    pool.current_query = NULL;
    pool.stop.store(false);

    pool.threads.resize(num_threads);
    pool.params.resize(num_threads);
    pool.local_candidates.resize(num_threads);

    pthread_barrier_init(&pool.start_barrier, NULL, num_threads + 1);
    pthread_barrier_init(&pool.end_barrier, NULL, num_threads + 1);

    for (int t = 0; t < num_threads; ++t) {
        pool.params[t].tid = t;
        pool.params[t].pool = &pool;
        pthread_create(&pool.threads[t], NULL, flat_static_worker, (void*)&pool.params[t]);
    }
}


std::priority_queue<std::pair<float, uint32_t> > flat_search_neon_pthread_static(
    FlatPthreadStaticPool& pool,
    const float* query
) {
    pool.current_query = query;

    for (int t = 0; t < pool.num_threads; ++t) {
        pool.local_candidates[t].clear();
    }

    pthread_barrier_wait(&pool.start_barrier);
    pthread_barrier_wait(&pool.end_barrier);

    // 合并各线程局部 Top-k
    std::priority_queue<std::pair<float, uint32_t> > global_heap;

    for (int t = 0; t < pool.num_threads; ++t) {
        for (size_t i = 0; i < pool.local_candidates[t].size(); ++i) {
            const std::pair<float, uint32_t>& item = pool.local_candidates[t][i];

            if (global_heap.size() < pool.k) {
                global_heap.push(item);
            } else {
                if (item.first < global_heap.top().first) {
                    global_heap.pop();
                    global_heap.push(item);
                }
            }
        }
    }

    return global_heap;
}


void destroy_flat_pthread_static_pool(FlatPthreadStaticPool& pool)
{
    pool.stop.store(true);

    pthread_barrier_wait(&pool.start_barrier);

    for (int t = 0; t < pool.num_threads; ++t) {
        pthread_join(pool.threads[t], NULL);
    }

    pthread_barrier_destroy(&pool.start_barrier);
    pthread_barrier_destroy(&pool.end_barrier);
}

// ===================== Flat-SIMD + OpenMP 多线程扫描 =====================
// 优化思路：
// 1. 将 base 向量集合按 OpenMP 线程划分；
// 2. 每个线程内部仍然调用 inner_product_neon 进行 SIMD 内积计算；
// 3. 每个线程维护自己的局部 Top-k，避免多线程竞争同一个全局堆；
// 4. 并行区域结束后，主线程合并各线程局部 Top-k，得到最终 Top-k。

std::priority_queue<std::pair<float, uint32_t> > flat_search_neon_openmp(
    float* base,
    float* query,
    size_t base_number,
    size_t vecdim,
    size_t k
) {
    int num_threads = OMP_NUM_THREADS;

    if (num_threads < 1) {
        num_threads = 1;
    }

    if ((size_t)num_threads > base_number) {
        num_threads = (int)base_number;
    }

    std::vector<std::vector<std::pair<float, uint32_t> > > local_candidates(num_threads);

    #pragma omp parallel num_threads(num_threads)
    {
        int tid = omp_get_thread_num();

        // 大根堆：堆顶是当前 Top-k 中距离最大的元素
        std::priority_queue<std::pair<float, uint32_t> > local_heap;

        #pragma omp for schedule(static)
        for (long long i = 0; i < (long long)base_number; ++i) {
            const float* base_vec = base + (size_t)i * vecdim;

            float dis = 1.0f - inner_product_neon(base_vec, query, vecdim);

            if (local_heap.size() < k) {
                local_heap.push({dis, (uint32_t)i});
            } else {
                if (dis < local_heap.top().first) {
                    local_heap.pop();
                    local_heap.push({dis, (uint32_t)i});
                }
            }
        }

        local_candidates[tid].reserve(local_heap.size());

        while (!local_heap.empty()) {
            local_candidates[tid].push_back(local_heap.top());
            local_heap.pop();
        }
    }

    // 合并所有线程的局部 Top-k
    std::priority_queue<std::pair<float, uint32_t> > global_heap;

    for (int tid = 0; tid < num_threads; ++tid) {
        for (size_t i = 0; i < local_candidates[tid].size(); ++i) {
            const std::pair<float, uint32_t>& item = local_candidates[tid][i];

            if (global_heap.size() < k) {
                global_heap.push(item);
            } else {
                if (item.first < global_heap.top().first) {
                    global_heap.pop();
                    global_heap.push(item);
                }
            }
        }
    }

    return global_heap;
}


// ===================== IVF-SIMD baseline =====================
// IVF 思路：
// 1. 使用 KMeans 思路训练 nlist 个中心；
// 2. 将每个 base 向量分配到最相似的中心；
// 3. 查询时只访问距离 query 最近的 nprobe 个簇；
// 4. 簇内候选仍使用 inner_product_neon 进行精确 SIMD 内积计算。

struct IVFIndex {
    size_t nlist;
    size_t vecdim;

    // centroids[c * vecdim + d]
    std::vector<float> centroids;

    // 每个簇保存 base 向量编号
    std::vector<std::vector<uint32_t> > lists;
};


// 使用内积选择最近中心。score 越大，表示越相似。
static inline uint32_t nearest_centroid_ip(
    const float* vec,
    const std::vector<float>& centroids,
    size_t nlist,
    size_t vecdim
) {
    float best_score = -std::numeric_limits<float>::max();
    uint32_t best_id = 0;

    for (uint32_t c = 0; c < (uint32_t)nlist; ++c) {
        const float* center = centroids.data() + (size_t)c * vecdim;
        float score = inner_product_neon(center, vec, vecdim);

        if (score > best_score) {
            best_score = score;
            best_id = c;
        }
    }

    return best_id;
}


void train_ivf_centroids(
    float* base,
    size_t base_number,
    size_t vecdim,
    IVFIndex& ivf
) {
    ivf.nlist = IVF_NLIST;
    ivf.vecdim = vecdim;
    ivf.centroids.resize(ivf.nlist * vecdim);

    size_t sample_n = std::min(IVF_TRAIN_SAMPLE, base_number);

    std::vector<uint32_t> sample_ids(sample_n);
    for (size_t i = 0; i < sample_n; ++i) {
        sample_ids[i] = (uint32_t)((i * base_number) / sample_n);
    }

    // 初始化中心：从样本中均匀选取 nlist 个向量
    for (size_t c = 0; c < ivf.nlist; ++c) {
        size_t sample_pos = (c * sample_n) / ivf.nlist;
        uint32_t base_id = sample_ids[sample_pos];

        const float* src = base + (size_t)base_id * vecdim;
        float* dst = ivf.centroids.data() + c * vecdim;

        std::memcpy(dst, src, vecdim * sizeof(float));
    }

    // 简化版 KMeans 迭代
    for (size_t iter = 0; iter < IVF_ITERS; ++iter) {
        std::vector<float> sums(ivf.nlist * vecdim, 0.0f);
        std::vector<int> counts(ivf.nlist, 0);

        for (size_t s = 0; s < sample_n; ++s) {
            uint32_t base_id = sample_ids[s];
            const float* vec = base + (size_t)base_id * vecdim;

            uint32_t cid = nearest_centroid_ip(
                vec,
                ivf.centroids,
                ivf.nlist,
                vecdim
            );

            float* sum_ptr = sums.data() + (size_t)cid * vecdim;

            for (size_t d = 0; d < vecdim; ++d) {
                sum_ptr[d] += vec[d];
            }

            counts[cid]++;
        }

        for (size_t c = 0; c < ivf.nlist; ++c) {
            if (counts[c] == 0) {
                continue;
            }

            float* center = ivf.centroids.data() + c * vecdim;
            float* sum_ptr = sums.data() + c * vecdim;
            float inv_count = 1.0f / counts[c];

            for (size_t d = 0; d < vecdim; ++d) {
                center[d] = sum_ptr[d] * inv_count;
            }
        }

        std::cerr << "IVF train iter " << iter << " finished\n";
    }
}


void build_ivf_index(
    float* base,
    size_t base_number,
    size_t vecdim,
    IVFIndex& ivf
) {
    train_ivf_centroids(base, base_number, vecdim, ivf);

    ivf.lists.clear();
    ivf.lists.resize(ivf.nlist);

    for (uint32_t i = 0; i < (uint32_t)base_number; ++i) {
        const float* vec = base + (size_t)i * vecdim;

        uint32_t cid = nearest_centroid_ip(
            vec,
            ivf.centroids,
            ivf.nlist,
            vecdim
        );

        ivf.lists[cid].push_back(i);
    }

    size_t min_size = base_number;
    size_t max_size = 0;
    size_t total_size = 0;

    for (size_t c = 0; c < ivf.nlist; ++c) {
        size_t sz = ivf.lists[c].size();
        min_size = std::min(min_size, sz);
        max_size = std::max(max_size, sz);
        total_size += sz;
    }

    std::cerr << "IVF index finished: nlist=" << ivf.nlist
              << " avg_list_size=" << (double)total_size / ivf.nlist
              << " min_list_size=" << min_size
              << " max_list_size=" << max_size << "\n";
}


std::priority_queue<std::pair<float, uint32_t> > flat_search_ivf_simd(
    float* base,
    float* query,
    size_t base_number,
    size_t vecdim,
    size_t k,
    size_t nprobe,
    const IVFIndex& ivf
) {
    if (nprobe < 1) {
        nprobe = 1;
    }

    if (nprobe > ivf.nlist) {
        nprobe = ivf.nlist;
    }

    // 1. 选择 query 最近的 nprobe 个簇
    typedef std::pair<float, uint32_t> ScoreId;

    // 小根堆：保留内积分数最高的 nprobe 个中心
    std::priority_queue<
        ScoreId,
        std::vector<ScoreId>,
        std::greater<ScoreId>
    > centroid_heap;

    for (uint32_t c = 0; c < (uint32_t)ivf.nlist; ++c) {
        const float* center = ivf.centroids.data() + (size_t)c * vecdim;
        float score = inner_product_neon(center, query, vecdim);

        if (centroid_heap.size() < nprobe) {
            centroid_heap.push({score, c});
        } else {
            if (score > centroid_heap.top().first) {
                centroid_heap.pop();
                centroid_heap.push({score, c});
            }
        }
    }

    std::vector<uint32_t> probe_lists;
    probe_lists.reserve(nprobe);

    while (!centroid_heap.empty()) {
        probe_lists.push_back(centroid_heap.top().second);
        centroid_heap.pop();
    }

    // 2. 只扫描 nprobe 个簇中的 base 向量
    std::priority_queue<std::pair<float, uint32_t> > final_q;

    for (size_t p = 0; p < probe_lists.size(); ++p) {
        uint32_t cid = probe_lists[p];
        const std::vector<uint32_t>& list = ivf.lists[cid];

        for (size_t t = 0; t < list.size(); ++t) {
            uint32_t idx = list[t];
            const float* base_vec = base + (size_t)idx * vecdim;

            float dis = 1.0f - inner_product_neon(base_vec, query, vecdim);

            if (final_q.size() < k) {
                final_q.push({dis, idx});
            } else {
                if (dis < final_q.top().first) {
                    final_q.pop();
                    final_q.push({dis, idx});
                }
            }
        }
    }

    return final_q;
}

// ===================== IVF-SIMD + OpenMP 多线程扫描 =====================
// 优化思路：
// 1. 仍然先选择 query 最近的 nprobe 个 IVF 簇；
// 2. 将这些簇中的 base 向量 id 收集为 candidates；
// 3. 使用 OpenMP 并行扫描 candidates；
// 4. 每个线程内部调用 inner_product_neon 做 SIMD 内积；
// 5. 每个线程维护局部 Top-k，最后主线程合并，避免多个线程竞争全局堆。

std::priority_queue<std::pair<float, uint32_t> > flat_search_ivf_simd_openmp(
    float* base,
    float* query,
    size_t base_number,
    size_t vecdim,
    size_t k,
    size_t nprobe,
    const IVFIndex& ivf
) {
    if (nprobe < 1) {
        nprobe = 1;
    }

    if (nprobe > ivf.nlist) {
        nprobe = ivf.nlist;
    }

    typedef std::pair<float, uint32_t> ScoreId;

    // 1. 选择 query 最近的 nprobe 个簇
    std::priority_queue<
        ScoreId,
        std::vector<ScoreId>,
        std::greater<ScoreId>
    > centroid_heap;

    for (uint32_t c = 0; c < (uint32_t)ivf.nlist; ++c) {
        const float* center = ivf.centroids.data() + (size_t)c * vecdim;
        float score = inner_product_neon(center, query, vecdim);

        if (centroid_heap.size() < nprobe) {
            centroid_heap.push({score, c});
        } else {
            if (score > centroid_heap.top().first) {
                centroid_heap.pop();
                centroid_heap.push({score, c});
            }
        }
    }

    std::vector<uint32_t> probe_lists;
    probe_lists.reserve(nprobe);

    while (!centroid_heap.empty()) {
        probe_lists.push_back(centroid_heap.top().second);
        centroid_heap.pop();
    }

    // 2. 收集需要扫描的候选向量 id
    std::vector<uint32_t> candidates;
    for (size_t p = 0; p < probe_lists.size(); ++p) {
        uint32_t cid = probe_lists[p];
        const std::vector<uint32_t>& list = ivf.lists[cid];

        candidates.insert(candidates.end(), list.begin(), list.end());
    }

    int num_threads = OMP_NUM_THREADS;
    if (num_threads < 1) {
        num_threads = 1;
    }

    if ((size_t)num_threads > candidates.size() && candidates.size() > 0) {
        num_threads = (int)candidates.size();
    }

    if (candidates.empty()) {
        return std::priority_queue<std::pair<float, uint32_t> >();
    }

    std::vector<std::vector<std::pair<float, uint32_t> > > local_candidates(num_threads);

    // 3. OpenMP 并行扫描候选向量
    #pragma omp parallel num_threads(num_threads)
    {
        int tid = omp_get_thread_num();

        // 大根堆：堆顶是当前局部 Top-k 中距离最大的元素
        std::priority_queue<std::pair<float, uint32_t> > local_heap;

        #pragma omp for schedule(static)
        for (long long t = 0; t < (long long)candidates.size(); ++t) {
            uint32_t idx = candidates[(size_t)t];
            const float* base_vec = base + (size_t)idx * vecdim;

            float dis = 1.0f - inner_product_neon(base_vec, query, vecdim);

            if (local_heap.size() < k) {
                local_heap.push({dis, idx});
            } else {
                if (dis < local_heap.top().first) {
                    local_heap.pop();
                    local_heap.push({dis, idx});
                }
            }
        }

        local_candidates[tid].reserve(local_heap.size());

        while (!local_heap.empty()) {
            local_candidates[tid].push_back(local_heap.top());
            local_heap.pop();
        }
    }

    // 4. 合并各线程局部 Top-k
    std::priority_queue<std::pair<float, uint32_t> > final_q;

    for (int tid = 0; tid < num_threads; ++tid) {
        for (size_t i = 0; i < local_candidates[tid].size(); ++i) {
            const std::pair<float, uint32_t>& item = local_candidates[tid][i];

            if (final_q.size() < k) {
                final_q.push(item);
            } else {
                if (item.first < final_q.top().first) {
                    final_q.pop();
                    final_q.push(item);
                }
            }
        }
    }

    return final_q;
}

// ===================== IVF-SIMD + 静态 Pthread 线程池 =====================
// 优化思路：
// 1. 主线程为每个 query 选择最近的 nprobe 个 IVF 簇；
// 2. 将这些簇中的 base id 收集成 candidates；
// 3. 通过 barrier 唤醒静态线程池；
// 4. 每个线程扫描 candidates 的一段，内部仍调用 inner_product_neon；
// 5. 每个线程维护局部 Top-k，主线程最后合并局部结果。

struct IVFPthreadStaticPool;

struct IVFPthreadStaticParam {
    int tid;
    IVFPthreadStaticPool* pool;
};

struct IVFPthreadStaticPool {
    int num_threads;

    float* base;
    size_t base_number;
    size_t vecdim;
    size_t k;

    const float* current_query;
    const std::vector<uint32_t>* current_candidates;

    std::atomic<bool> stop;

    pthread_barrier_t start_barrier;
    pthread_barrier_t end_barrier;

    std::vector<pthread_t> threads;
    std::vector<IVFPthreadStaticParam> params;

    std::vector<std::vector<std::pair<float, uint32_t> > > local_candidates;
};


void* ivf_static_worker(void* arg)
{
    IVFPthreadStaticParam* param = (IVFPthreadStaticParam*)arg;
    int tid = param->tid;
    IVFPthreadStaticPool* pool = param->pool;

    while (true) {
        pthread_barrier_wait(&pool->start_barrier);

        if (pool->stop.load()) {
            break;
        }

        const float* query = pool->current_query;
        const std::vector<uint32_t>& candidates = *(pool->current_candidates);

        float* base = pool->base;
        size_t vecdim = pool->vecdim;
        size_t k = pool->k;
        int num_threads = pool->num_threads;

        size_t cand_num = candidates.size();

        size_t begin = cand_num * (size_t)tid / (size_t)num_threads;
        size_t end = cand_num * (size_t)(tid + 1) / (size_t)num_threads;

        std::priority_queue<std::pair<float, uint32_t> > local_heap;

        for (size_t t = begin; t < end; ++t) {
            uint32_t idx = candidates[t];
            const float* base_vec = base + (size_t)idx * vecdim;

            float dis = 1.0f - inner_product_neon(base_vec, query, vecdim);

            if (local_heap.size() < k) {
                local_heap.push({dis, idx});
            } else {
                if (dis < local_heap.top().first) {
                    local_heap.pop();
                    local_heap.push({dis, idx});
                }
            }
        }

        std::vector<std::pair<float, uint32_t> >& out = pool->local_candidates[tid];
        out.clear();
        out.reserve(local_heap.size());

        while (!local_heap.empty()) {
            out.push_back(local_heap.top());
            local_heap.pop();
        }

        pthread_barrier_wait(&pool->end_barrier);
    }

    return NULL;
}


void init_ivf_pthread_static_pool(
    IVFPthreadStaticPool& pool,
    float* base,
    size_t base_number,
    size_t vecdim,
    size_t k,
    int num_threads
) {
    if (num_threads < 1) {
        num_threads = 1;
    }

    if ((size_t)num_threads > base_number) {
        num_threads = (int)base_number;
    }

    pool.num_threads = num_threads;
    pool.base = base;
    pool.base_number = base_number;
    pool.vecdim = vecdim;
    pool.k = k;
    pool.current_query = NULL;
    pool.current_candidates = NULL;
    pool.stop.store(false);

    pool.threads.resize(num_threads);
    pool.params.resize(num_threads);
    pool.local_candidates.resize(num_threads);

    pthread_barrier_init(&pool.start_barrier, NULL, num_threads + 1);
    pthread_barrier_init(&pool.end_barrier, NULL, num_threads + 1);

    for (int t = 0; t < num_threads; ++t) {
        pool.params[t].tid = t;
        pool.params[t].pool = &pool;
        pthread_create(&pool.threads[t], NULL, ivf_static_worker, (void*)&pool.params[t]);
    }
}


std::priority_queue<std::pair<float, uint32_t> > flat_search_ivf_simd_pthread_static(
    IVFPthreadStaticPool& pool,
    float* query,
    size_t nprobe,
    const IVFIndex& ivf
) {
    if (nprobe < 1) {
        nprobe = 1;
    }

    if (nprobe > ivf.nlist) {
        nprobe = ivf.nlist;
    }

    typedef std::pair<float, uint32_t> ScoreId;

    // 1. 选择 query 最近的 nprobe 个 IVF 簇
    std::priority_queue<
        ScoreId,
        std::vector<ScoreId>,
        std::greater<ScoreId>
    > centroid_heap;

    for (uint32_t c = 0; c < (uint32_t)ivf.nlist; ++c) {
        const float* center = ivf.centroids.data() + (size_t)c * pool.vecdim;
        float score = inner_product_neon(center, query, pool.vecdim);

        if (centroid_heap.size() < nprobe) {
            centroid_heap.push({score, c});
        } else {
            if (score > centroid_heap.top().first) {
                centroid_heap.pop();
                centroid_heap.push({score, c});
            }
        }
    }

    std::vector<uint32_t> probe_lists;
    probe_lists.reserve(nprobe);

    while (!centroid_heap.empty()) {
        probe_lists.push_back(centroid_heap.top().second);
        centroid_heap.pop();
    }

    // 2. 收集待扫描候选向量 id
    std::vector<uint32_t> candidates;

    for (size_t p = 0; p < probe_lists.size(); ++p) {
        uint32_t cid = probe_lists[p];
        const std::vector<uint32_t>& list = ivf.lists[cid];

        candidates.insert(candidates.end(), list.begin(), list.end());
    }

    if (candidates.empty()) {
        return std::priority_queue<std::pair<float, uint32_t> >();
    }

    for (int t = 0; t < pool.num_threads; ++t) {
        pool.local_candidates[t].clear();
    }

    pool.current_query = query;
    pool.current_candidates = &candidates;

    // 3. 唤醒工作线程扫描 candidates
    pthread_barrier_wait(&pool.start_barrier);

    // 4. 等待所有工作线程完成
    pthread_barrier_wait(&pool.end_barrier);

    // 5. 合并各线程局部 Top-k
    std::priority_queue<std::pair<float, uint32_t> > final_q;

    for (int tid = 0; tid < pool.num_threads; ++tid) {
        for (size_t i = 0; i < pool.local_candidates[tid].size(); ++i) {
            const std::pair<float, uint32_t>& item = pool.local_candidates[tid][i];

            if (final_q.size() < pool.k) {
                final_q.push(item);
            } else {
                if (item.first < final_q.top().first) {
                    final_q.pop();
                    final_q.push(item);
                }
            }
        }
    }

    return final_q;
}


void destroy_ivf_pthread_static_pool(IVFPthreadStaticPool& pool)
{
    pool.stop.store(true);

    pthread_barrier_wait(&pool.start_barrier);

    for (int t = 0; t < pool.num_threads; ++t) {
        pthread_join(pool.threads[t], NULL);
    }

    pthread_barrier_destroy(&pool.start_barrier);
    pthread_barrier_destroy(&pool.end_barrier);
}


// ===================== IVF-PQ-SIMD baseline =====================
// 优化思路：
// 1. IVF 阶段：先选择 query 最近的 nprobe 个簇，减少候选数量；
// 2. PQ 阶段：只对这些 IVF 候选做 PQ-ADC 查表粗排；
// 3. Top-p 阶段：保留 PQ 分数最高的 top_p 个候选；
// 4. 精排阶段：对 Top-p 候选使用原始 float NEON 内积精排，返回 Top-k。
// 说明：这里实现的是 IVF + 全局 PQ-ADC 的简化 IVF-PQ，适合本实验阶段使用。

std::priority_queue<std::pair<float, uint32_t> > flat_search_ivf_pq_adc(
    float* base,
    float* query,
    size_t base_number,
    size_t vecdim,
    size_t k,
    size_t nprobe,
    size_t top_p,
    const IVFIndex& ivf,
    const PQIndex& pq
) {
    if (nprobe < 1) {
        nprobe = 1;
    }

    if (nprobe > ivf.nlist) {
        nprobe = ivf.nlist;
    }

    typedef std::pair<float, uint32_t> ScoreId;

    // 1. 选择 query 最近的 nprobe 个 IVF 簇
    std::priority_queue<
        ScoreId,
        std::vector<ScoreId>,
        std::greater<ScoreId>
    > centroid_heap;

    for (uint32_t c = 0; c < (uint32_t)ivf.nlist; ++c) {
        const float* center = ivf.centroids.data() + (size_t)c * vecdim;
        float score = inner_product_neon(center, query, vecdim);

        if (centroid_heap.size() < nprobe) {
            centroid_heap.push({score, c});
        } else {
            if (score > centroid_heap.top().first) {
                centroid_heap.pop();
                centroid_heap.push({score, c});
            }
        }
    }

    std::vector<uint32_t> probe_lists;
    probe_lists.reserve(nprobe);

    while (!centroid_heap.empty()) {
        probe_lists.push_back(centroid_heap.top().second);
        centroid_heap.pop();
    }

    // 2. 收集 IVF 候选向量 id
    std::vector<uint32_t> candidates;

    for (size_t p = 0; p < probe_lists.size(); ++p) {
        uint32_t cid = probe_lists[p];
        const std::vector<uint32_t>& list = ivf.lists[cid];
        candidates.insert(candidates.end(), list.begin(), list.end());
    }

    if (candidates.empty()) {
        return std::priority_queue<std::pair<float, uint32_t> >();
    }

    if (top_p < k) {
        top_p = k;
    }

    if (top_p > candidates.size()) {
        top_p = candidates.size();
    }

    // 3. 构建 PQ-ADC LUT
    std::vector<float> lut;
    build_adc_lut(query, pq, lut);

    // 4. 在 IVF 候选集合内部做 PQ-ADC 粗排
    PQMinHeap coarse_heap;

    for (size_t t = 0; t < candidates.size(); ++t) {
        uint32_t idx = candidates[t];

        float score = 0.0f;
        const uint8_t* code = pq.codes.data() + (size_t)idx * pq.M;

        for (size_t m = 0; m < pq.M; ++m) {
            uint8_t cid = code[m];
            score += lut[m * pq.Ks + cid];
        }

        if (coarse_heap.size() < top_p) {
            coarse_heap.push({score, idx});
        } else {
            if (score > coarse_heap.top().first) {
                coarse_heap.pop();
                coarse_heap.push({score, idx});
            }
        }
    }

    // 5. 收集 PQ 粗排 Top-p 候选
    std::vector<uint32_t> rerank_candidates;
    rerank_candidates.reserve(top_p);

    while (!coarse_heap.empty()) {
        rerank_candidates.push_back(coarse_heap.top().second);
        coarse_heap.pop();
    }

    // 6. 对 Top-p 候选使用原始 float NEON 精排
    std::priority_queue<std::pair<float, uint32_t> > final_q;

    for (size_t t = 0; t < rerank_candidates.size(); ++t) {
        uint32_t idx = rerank_candidates[t];
        const float* base_vec = base + (size_t)idx * vecdim;

        float dis = 1.0f - inner_product_neon(base_vec, query, vecdim);

        if (final_q.size() < k) {
            final_q.push({dis, idx});
        } else {
            if (dis < final_q.top().first) {
                final_q.pop();
                final_q.push({dis, idx});
            }
        }
    }

    return final_q;
}

// ===================== IVF-PQ-SIMD + 静态 Pthread 线程池 =====================
// 优化思路：
// 1. 主线程选择 query 最近的 nprobe 个 IVF 簇；
// 2. 收集这些簇中的候选向量 id；
// 3. 主线程构建 PQ-ADC LUT；
// 4. Pthread 工作线程并行扫描 candidates，做 PQ-ADC 粗排；
// 5. 每个线程维护局部 Top-p 小根堆；
// 6. 主线程合并局部 Top-p，得到全局 Top-p；
// 7. 对全局 Top-p 使用原始 float NEON 精排，返回 Top-k。

struct IVFPQPthreadStaticPool;

struct IVFPQPthreadStaticParam {
    int tid;
    IVFPQPthreadStaticPool* pool;
};

struct IVFPQPthreadStaticPool {
    int num_threads;

    float* base;
    size_t base_number;
    size_t vecdim;
    size_t k;

    const PQIndex* pq;

    const float* current_query;
    const std::vector<uint32_t>* current_candidates;
    std::vector<float> lut;
    size_t current_top_p;

    std::atomic<bool> stop;

    pthread_barrier_t start_barrier;
    pthread_barrier_t end_barrier;

    std::vector<pthread_t> threads;
    std::vector<IVFPQPthreadStaticParam> params;

    std::vector<std::vector<PQScoreId> > local_candidates;
};


void* ivf_pq_static_worker(void* arg)
{
    IVFPQPthreadStaticParam* param = (IVFPQPthreadStaticParam*)arg;
    int tid = param->tid;
    IVFPQPthreadStaticPool* pool = param->pool;

    while (true) {
        pthread_barrier_wait(&pool->start_barrier);

        if (pool->stop.load()) {
            break;
        }

        const PQIndex& pq = *(pool->pq);
        const std::vector<uint32_t>& candidates = *(pool->current_candidates);
        const std::vector<float>& lut = pool->lut;

        size_t cand_num = candidates.size();
        size_t top_p = pool->current_top_p;
        int num_threads = pool->num_threads;

        size_t begin = cand_num * (size_t)tid / (size_t)num_threads;
        size_t end = cand_num * (size_t)(tid + 1) / (size_t)num_threads;

        PQMinHeap local_heap;

        for (size_t t = begin; t < end; ++t) {
            uint32_t idx = candidates[t];

            float score = 0.0f;
            const uint8_t* code = pq.codes.data() + (size_t)idx * pq.M;

            for (size_t m = 0; m < pq.M; ++m) {
                uint8_t cid = code[m];
                score += lut[m * pq.Ks + cid];
            }

            if (local_heap.size() < top_p) {
                local_heap.push({score, idx});
            } else {
                if (score > local_heap.top().first) {
                    local_heap.pop();
                    local_heap.push({score, idx});
                }
            }
        }

        std::vector<PQScoreId>& out = pool->local_candidates[tid];
        out.clear();
        out.reserve(local_heap.size());

        while (!local_heap.empty()) {
            out.push_back(local_heap.top());
            local_heap.pop();
        }

        pthread_barrier_wait(&pool->end_barrier);
    }

    return NULL;
}


void init_ivf_pq_pthread_static_pool(
    IVFPQPthreadStaticPool& pool,
    float* base,
    size_t base_number,
    size_t vecdim,
    size_t k,
    const PQIndex& pq,
    int num_threads
) {
    if (num_threads < 1) {
        num_threads = 1;
    }

    if ((size_t)num_threads > base_number) {
        num_threads = (int)base_number;
    }

    pool.num_threads = num_threads;
    pool.base = base;
    pool.base_number = base_number;
    pool.vecdim = vecdim;
    pool.k = k;
    pool.pq = &pq;
    pool.current_query = NULL;
    pool.current_candidates = NULL;
    pool.current_top_p = 0;
    pool.stop.store(false);

    pool.threads.resize(num_threads);
    pool.params.resize(num_threads);
    pool.local_candidates.resize(num_threads);

    pthread_barrier_init(&pool.start_barrier, NULL, num_threads + 1);
    pthread_barrier_init(&pool.end_barrier, NULL, num_threads + 1);

    for (int t = 0; t < num_threads; ++t) {
        pool.params[t].tid = t;
        pool.params[t].pool = &pool;
        pthread_create(&pool.threads[t], NULL, ivf_pq_static_worker, (void*)&pool.params[t]);
    }
}


std::priority_queue<std::pair<float, uint32_t> > flat_search_ivf_pq_adc_pthread_static(
    IVFPQPthreadStaticPool& pool,
    float* query,
    size_t nprobe,
    size_t top_p,
    const IVFIndex& ivf
) {
    if (nprobe < 1) {
        nprobe = 1;
    }

    if (nprobe > ivf.nlist) {
        nprobe = ivf.nlist;
    }

    typedef std::pair<float, uint32_t> ScoreId;

    // 1. 选择 query 最近的 nprobe 个 IVF 簇
    std::priority_queue<
        ScoreId,
        std::vector<ScoreId>,
        std::greater<ScoreId>
    > centroid_heap;

    for (uint32_t c = 0; c < (uint32_t)ivf.nlist; ++c) {
        const float* center = ivf.centroids.data() + (size_t)c * pool.vecdim;
        float score = inner_product_neon(center, query, pool.vecdim);

        if (centroid_heap.size() < nprobe) {
            centroid_heap.push({score, c});
        } else {
            if (score > centroid_heap.top().first) {
                centroid_heap.pop();
                centroid_heap.push({score, c});
            }
        }
    }

    std::vector<uint32_t> probe_lists;
    probe_lists.reserve(nprobe);

    while (!centroid_heap.empty()) {
        probe_lists.push_back(centroid_heap.top().second);
        centroid_heap.pop();
    }

    // 2. 收集 IVF 候选向量 id
    std::vector<uint32_t> candidates;

    for (size_t p = 0; p < probe_lists.size(); ++p) {
        uint32_t cid = probe_lists[p];
        const std::vector<uint32_t>& list = ivf.lists[cid];
        candidates.insert(candidates.end(), list.begin(), list.end());
    }

    if (candidates.empty()) {
        return std::priority_queue<std::pair<float, uint32_t> >();
    }

    if (top_p < pool.k) {
        top_p = pool.k;
    }

    if (top_p > candidates.size()) {
        top_p = candidates.size();
    }

    // 3. 构建 PQ-ADC LUT
    build_adc_lut(query, *(pool.pq), pool.lut);

    for (int t = 0; t < pool.num_threads; ++t) {
        pool.local_candidates[t].clear();
    }

    pool.current_query = query;
    pool.current_candidates = &candidates;
    pool.current_top_p = top_p;

    // 4. 唤醒工作线程并行做 PQ-ADC 粗排
    pthread_barrier_wait(&pool.start_barrier);

    // 5. 等待所有线程完成粗排
    pthread_barrier_wait(&pool.end_barrier);

    // 6. 合并各线程局部 Top-p，得到全局 Top-p
    PQMinHeap global_heap;

    for (int tid = 0; tid < pool.num_threads; ++tid) {
        for (size_t i = 0; i < pool.local_candidates[tid].size(); ++i) {
            const PQScoreId& item = pool.local_candidates[tid][i];

            if (global_heap.size() < top_p) {
                global_heap.push(item);
            } else {
                if (item.first > global_heap.top().first) {
                    global_heap.pop();
                    global_heap.push(item);
                }
            }
        }
    }

    std::vector<uint32_t> rerank_candidates;
    rerank_candidates.reserve(top_p);

    while (!global_heap.empty()) {
        rerank_candidates.push_back(global_heap.top().second);
        global_heap.pop();
    }

    // 7. 对全局 Top-p 候选使用原始 float NEON 精排
    std::priority_queue<std::pair<float, uint32_t> > final_q;

    for (size_t t = 0; t < rerank_candidates.size(); ++t) {
        uint32_t idx = rerank_candidates[t];
        const float* base_vec = pool.base + (size_t)idx * pool.vecdim;

        float dis = 1.0f - inner_product_neon(base_vec, query, pool.vecdim);

        if (final_q.size() < pool.k) {
            final_q.push({dis, idx});
        } else {
            if (dis < final_q.top().first) {
                final_q.pop();
                final_q.push({dis, idx});
            }
        }
    }

    return final_q;
}


void destroy_ivf_pq_pthread_static_pool(IVFPQPthreadStaticPool& pool)
{
    pool.stop.store(true);

    pthread_barrier_wait(&pool.start_barrier);

    for (int t = 0; t < pool.num_threads; ++t) {
        pthread_join(pool.threads[t], NULL);
    }

    pthread_barrier_destroy(&pool.start_barrier);
    pthread_barrier_destroy(&pool.end_barrier);
}

// ===================== IVF-PQ-SIMD + OpenMP =====================
// 优化思路：
// 1. 主线程选择 query 最近的 nprobe 个 IVF 簇；
// 2. 收集这些簇中的候选向量 id；
// 3. 构建 PQ-ADC LUT；
// 4. 使用 OpenMP 并行扫描 candidates，做 PQ-ADC 粗排；
// 5. 每个线程维护局部 Top-p；
// 6. 主线程合并局部 Top-p；
// 7. 对全局 Top-p 使用原始 float NEON 精排。

std::priority_queue<std::pair<float, uint32_t> > flat_search_ivf_pq_adc_openmp(
    float* base,
    float* query,
    size_t base_number,
    size_t vecdim,
    size_t k,
    size_t nprobe,
    size_t top_p,
    const IVFIndex& ivf,
    const PQIndex& pq
) {
    if (nprobe < 1) {
        nprobe = 1;
    }

    if (nprobe > ivf.nlist) {
        nprobe = ivf.nlist;
    }

    typedef std::pair<float, uint32_t> ScoreId;

    // 1. 选择 query 最近的 nprobe 个 IVF 簇
    std::priority_queue<
        ScoreId,
        std::vector<ScoreId>,
        std::greater<ScoreId>
    > centroid_heap;

    for (uint32_t c = 0; c < (uint32_t)ivf.nlist; ++c) {
        const float* center = ivf.centroids.data() + (size_t)c * vecdim;
        float score = inner_product_neon(center, query, vecdim);

        if (centroid_heap.size() < nprobe) {
            centroid_heap.push({score, c});
        } else {
            if (score > centroid_heap.top().first) {
                centroid_heap.pop();
                centroid_heap.push({score, c});
            }
        }
    }

    std::vector<uint32_t> probe_lists;
    probe_lists.reserve(nprobe);

    while (!centroid_heap.empty()) {
        probe_lists.push_back(centroid_heap.top().second);
        centroid_heap.pop();
    }

    // 2. 收集 IVF 候选
    std::vector<uint32_t> candidates;

    for (size_t p = 0; p < probe_lists.size(); ++p) {
        uint32_t cid = probe_lists[p];
        const std::vector<uint32_t>& list = ivf.lists[cid];
        candidates.insert(candidates.end(), list.begin(), list.end());
    }

    if (candidates.empty()) {
        return std::priority_queue<std::pair<float, uint32_t> >();
    }

    if (top_p < k) {
        top_p = k;
    }

    if (top_p > candidates.size()) {
        top_p = candidates.size();
    }

    // 3. 构建 PQ-ADC LUT
    std::vector<float> lut;
    build_adc_lut(query, pq, lut);

    int num_threads = OMP_NUM_THREADS;
    if (num_threads < 1) {
        num_threads = 1;
    }

    if ((size_t)num_threads > candidates.size()) {
        num_threads = (int)candidates.size();
    }

    std::vector<std::vector<PQScoreId> > local_candidates(num_threads);

    // 4. OpenMP 并行 PQ-ADC 粗排
    #pragma omp parallel num_threads(num_threads)
    {
        int tid = omp_get_thread_num();

        PQMinHeap local_heap;

        #pragma omp for schedule(static)
        for (long long t = 0; t < (long long)candidates.size(); ++t) {
            uint32_t idx = candidates[(size_t)t];

            float score = 0.0f;
            const uint8_t* code = pq.codes.data() + (size_t)idx * pq.M;

            for (size_t m = 0; m < pq.M; ++m) {
                uint8_t cid = code[m];
                score += lut[m * pq.Ks + cid];
            }

            if (local_heap.size() < top_p) {
                local_heap.push({score, idx});
            } else {
                if (score > local_heap.top().first) {
                    local_heap.pop();
                    local_heap.push({score, idx});
                }
            }
        }

        local_candidates[tid].reserve(local_heap.size());

        while (!local_heap.empty()) {
            local_candidates[tid].push_back(local_heap.top());
            local_heap.pop();
        }
    }

    // 5. 合并局部 Top-p
    PQMinHeap global_heap;

    for (int tid = 0; tid < num_threads; ++tid) {
        for (size_t i = 0; i < local_candidates[tid].size(); ++i) {
            const PQScoreId& item = local_candidates[tid][i];

            if (global_heap.size() < top_p) {
                global_heap.push(item);
            } else {
                if (item.first > global_heap.top().first) {
                    global_heap.pop();
                    global_heap.push(item);
                }
            }
        }
    }

    std::vector<uint32_t> rerank_candidates;
    rerank_candidates.reserve(top_p);

    while (!global_heap.empty()) {
        rerank_candidates.push_back(global_heap.top().second);
        global_heap.pop();
    }

    // 6. 原始 float NEON 精排
    std::priority_queue<std::pair<float, uint32_t> > final_q;

    for (size_t t = 0; t < rerank_candidates.size(); ++t) {
        uint32_t idx = rerank_candidates[t];
        const float* base_vec = base + (size_t)idx * vecdim;

        float dis = 1.0f - inner_product_neon(base_vec, query, vecdim);

        if (final_q.size() < k) {
            final_q.push({dis, idx});
        } else {
            if (dis < final_q.top().first) {
                final_q.pop();
                final_q.push({dis, idx});
            }
        }
    }

    return final_q;
}

// ===================== HNSW baseline =====================
// 使用 hnswlib 构建 HNSW 索引，并调用 searchKnn 进行查询。
// HNSW_M 控制图中每个点的邻居数量规模，HNSW_EF_CONSTRUCTION 控制构图质量，
// HNSW_EF_SEARCH 控制查询时候选搜索宽度，ef 越大 recall 通常越高，但 latency 也越高。

HierarchicalNSW<float>* build_hnsw_index_in_memory(
    float* base,
    size_t base_number,
    size_t vecdim,
    InnerProductSpace*& ipspace
) {
    ipspace = new InnerProductSpace(vecdim);

    HierarchicalNSW<float>* hnsw_index =
        new HierarchicalNSW<float>(
            ipspace,
            base_number,
            HNSW_M,
            HNSW_EF_CONSTRUCTION
        );

    for (uint32_t i = 0; i < (uint32_t)base_number; ++i) {
        hnsw_index->addPoint(base + (size_t)i * vecdim, i);
    }

    hnsw_index->setEf(HNSW_EF_SEARCH);

    std::cerr << "HNSW index finished: M=" << HNSW_M
              << " efConstruction=" << HNSW_EF_CONSTRUCTION
              << " efSearch=" << HNSW_EF_SEARCH << "\n";

    return hnsw_index;
}


std::priority_queue<std::pair<float, uint32_t> > hnsw_search_baseline(
    HierarchicalNSW<float>* hnsw_index,
    float* query,
    size_t k
) {
    std::priority_queue<std::pair<float, labeltype> > raw_res =
        hnsw_index->searchKnn(query, k);

    std::priority_queue<std::pair<float, uint32_t> > res;

    while (!raw_res.empty()) {
        float dis = raw_res.top().first;
        uint32_t idx = (uint32_t)raw_res.top().second;
        res.push({dis, idx});
        raw_res.pop();
    }

    return res;
}

// ===================== IVF-HNSW baseline =====================
// 思路：
// 1. 先使用已有 IVFIndex 的 lists，将 base 划分为多个簇；
// 2. 对每个非空 IVF list 单独构建一个 HNSW；
// 3. 查询时先选择最近的 nprobe 个 IVF 簇；
// 4. 在这些簇对应的小 HNSW 中分别 searchKnn；
// 5. 合并所有小 HNSW 的结果，得到最终 Top-k。

struct IVFHNSWIndex {
    InnerProductSpace* space;
    std::vector<HierarchicalNSW<float>* > hnsw_lists;
    std::vector<size_t> list_sizes;
};


void build_ivf_hnsw_index(
    float* base,
    size_t base_number,
    size_t vecdim,
    const IVFIndex& ivf,
    IVFHNSWIndex& ivf_hnsw
) {
    ivf_hnsw.space = new InnerProductSpace(vecdim);
    ivf_hnsw.hnsw_lists.resize(ivf.nlist, NULL);
    ivf_hnsw.list_sizes.resize(ivf.nlist, 0);

    for (size_t cid = 0; cid < ivf.nlist; ++cid) {
        const std::vector<uint32_t>& list = ivf.lists[cid];
        size_t list_size = list.size();
        ivf_hnsw.list_sizes[cid] = list_size;

        if (list_size == 0) {
            continue;
        }

        HierarchicalNSW<float>* local_hnsw =
            new HierarchicalNSW<float>(
                ivf_hnsw.space,
                list_size,
                IVF_HNSW_M,
                IVF_HNSW_EF_CONSTRUCTION
            );

        for (size_t i = 0; i < list_size; ++i) {
            uint32_t global_id = list[i];
            local_hnsw->addPoint(
                base + (size_t)global_id * vecdim,
                global_id
            );
        }

        local_hnsw->setEf(IVF_HNSW_EF_SEARCH);
        ivf_hnsw.hnsw_lists[cid] = local_hnsw;

        std::cerr << "IVF-HNSW list " << cid
                  << " built, size=" << list_size << "\n";
    }

    std::cerr << "IVF-HNSW index finished: nlist=" << ivf.nlist
              << " efSearch=" << IVF_HNSW_EF_SEARCH << "\n";
}


std::priority_queue<std::pair<float, uint32_t> > search_ivf_hnsw(
    float* query,
    size_t vecdim,
    size_t k,
    size_t nprobe,
    const IVFIndex& ivf,
    const IVFHNSWIndex& ivf_hnsw
) {
    if (nprobe < 1) {
        nprobe = 1;
    }

    if (nprobe > ivf.nlist) {
        nprobe = ivf.nlist;
    }

    typedef std::pair<float, uint32_t> ScoreId;

    // 1. 先选择 query 最近的 nprobe 个 IVF 簇
    std::priority_queue<
        ScoreId,
        std::vector<ScoreId>,
        std::greater<ScoreId>
    > centroid_heap;

    for (uint32_t c = 0; c < (uint32_t)ivf.nlist; ++c) {
        const float* center = ivf.centroids.data() + (size_t)c * vecdim;
        float score = inner_product_neon(center, query, vecdim);

        if (centroid_heap.size() < nprobe) {
            centroid_heap.push({score, c});
        } else {
            if (score > centroid_heap.top().first) {
                centroid_heap.pop();
                centroid_heap.push({score, c});
            }
        }
    }

    std::vector<uint32_t> probe_lists;
    probe_lists.reserve(nprobe);

    while (!centroid_heap.empty()) {
        probe_lists.push_back(centroid_heap.top().second);
        centroid_heap.pop();
    }

    // 2. 在每个被选中的 IVF 簇对应的小 HNSW 中搜索
    std::priority_queue<std::pair<float, uint32_t> > final_q;

    for (size_t p = 0; p < probe_lists.size(); ++p) {
        uint32_t cid = probe_lists[p];

        HierarchicalNSW<float>* local_hnsw = ivf_hnsw.hnsw_lists[cid];
        size_t list_size = ivf_hnsw.list_sizes[cid];

        if (local_hnsw == NULL || list_size == 0) {
            continue;
        }

        size_t local_k = k;
        if (local_k > list_size) {
            local_k = list_size;
        }

        std::priority_queue<std::pair<float, labeltype> > local_res =
            local_hnsw->searchKnn(query, local_k);

        while (!local_res.empty()) {
            float dis = local_res.top().first;
            uint32_t idx = (uint32_t)local_res.top().second;

            if (final_q.size() < k) {
                final_q.push({dis, idx});
            } else {
                if (dis < final_q.top().first) {
                    final_q.pop();
                    final_q.push({dis, idx});
                }
            }

            local_res.pop();
        }
    }

    return final_q;
}


void destroy_ivf_hnsw_index(IVFHNSWIndex& ivf_hnsw)
{
    for (size_t i = 0; i < ivf_hnsw.hnsw_lists.size(); ++i) {
        if (ivf_hnsw.hnsw_lists[i] != NULL) {
            delete ivf_hnsw.hnsw_lists[i];
            ivf_hnsw.hnsw_lists[i] = NULL;
        }
    }

    delete ivf_hnsw.space;
    ivf_hnsw.space = NULL;
}


int main(int argc, char *argv[])
{
    //openmp
    omp_set_dynamic(0);
    omp_set_num_threads(OMP_NUM_THREADS);

    size_t test_number = 0, base_number = 0;
    size_t test_gt_d = 0, vecdim = 0;

    std::string data_path = "/anndata/"; 
    auto test_query = LoadData<float>(data_path + "DEEP100K.query.fbin", test_number, vecdim);
    auto test_gt = LoadData<int>(data_path + "DEEP100K.gt.query.100k.top100.bin", test_number, test_gt_d);
    auto base = LoadData<float>(data_path + "DEEP100K.base.100k.fbin", base_number, vecdim);
    //float base_scale = compute_symmetric_scale(base, base_number * vecdim);
    //int8_t* base_i8 = quantize_base_int8(base, base_number, vecdim, base_scale);
    //std::cerr << "SQ base quantization finished, base_scale: " << base_scale << "\n";
    
    // //PQ索引构建
    // PQIndex pq_index;
    // build_pq_index(base, base_number, vecdim, pq_index);

    // //IVF索引构建
    // IVFIndex ivf_index;
    // build_ivf_index(base, base_number, vecdim, ivf_index);

    //HNSW-baseline
    InnerProductSpace* hnsw_space = NULL;
    HierarchicalNSW<float>* hnsw_index =
        build_hnsw_index_in_memory(base, base_number, vecdim, hnsw_space);

    // //HNSW-IVF
    // IVFIndex ivf_index;
    // build_ivf_index(base, base_number, vecdim, ivf_index);

    // IVFHNSWIndex ivf_hnsw_index;
    // build_ivf_hnsw_index(
    //     base,
    //     base_number,
    //     vecdim,
    //     ivf_index,
    //     ivf_hnsw_index
    // );
    
    
    // 只测试前2000条查询
    test_number = 2000;

    const size_t k = 10;

    // //IVF-PQ-pthread
    // IVFPQPthreadStaticPool ivf_pq_static_pool;
    // init_ivf_pq_pthread_static_pool(
    //     ivf_pq_static_pool,
    //     base,
    //     base_number,
    //     vecdim,
    //     k,
    //     pq_index,
    //     PTHREAD_NUM_THREADS
    // );

    // //IVF-SIMD + pthread
    // IVFPthreadStaticPool ivf_static_pool;
    // init_ivf_pthread_static_pool(
    //     ivf_static_pool,
    //     base,
    //     base_number,
    //     vecdim,
    //     k,
    //     PTHREAD_NUM_THREADS
    // );

//     FlatPthreadStaticPool flat_static_pool;
//     init_flat_pthread_static_pool(
//     flat_static_pool,
//     base,
//     base_number,
//     vecdim,
//     k,
//     PTHREAD_NUM_THREADS
// );

    //const size_t sq_p = 50; // SQ粗排保留候选数，可后续测试 100/200/500/1000/2000
    const size_t pq_p =800; // 先用 1000，跑通后再调 50/100/200/500/2000

    std::vector<SearchResult> results;
    results.resize(test_number);


    //静态优化 PQ索引构建
    // PQPthreadStaticPool pq_static_pool;
    // init_pq_pthread_static_pool(
    //     pq_static_pool,
    //     base,
    //     base_number,
    //     vecdim,
    //     k,
    //     pq_p,
    //     pq_index,
    //     PTHREAD_NUM_THREADS
    // );

    
    // // 查询测试代码
    // for(int i = 0; i < test_number; ++i) {
    //     const unsigned long Converter = 1000 * 1000;
    //     struct timeval val;
    //     int ret = gettimeofday(&val, NULL);

    //     // 该文件已有代码中你只能修改该函数的调用方式
    //     // 可以任意修改函数名，函数参数或者改为调用成员函数，但是不能修改函数返回值。
        
    //     //auto res = flat_search_neon(base, test_query + i*vecdim, base_number, vecdim, k);
    //     //auto res = flat_search_sq(base, base_i8, test_query + i*vecdim, base_number, vecdim, k, sq_p);
    //     //auto res = flat_search_pq_adc(base, test_query + i*vecdim, base_number, vecdim, k, pq_p, pq_index);
    //     //auto res = flat_search_pq_adc_soa_block(base, test_query + i*vecdim, base_number, vecdim, k, pq_p, pq_index);
        
    //     // auto res = flat_search_pq_adc_soa_block_pthread(
    //     //     base,
    //     //     test_query + i*vecdim,
    //     //     base_number,
    //     //     vecdim,
    //     //     k,
    //     //     pq_p,
    //     //     pq_index
    //     // );

    //     //PQ静态pthread优化调用
    //     // auto res = flat_search_pq_adc_soa_block_pthread_static(
    //     //     pq_static_pool,
    //     //     test_query + i*vecdim
    //     // );

    //     //flat-simd pthread优化调用
    // //         auto res = flat_search_neon_pthread_static(
    // //     flat_static_pool,
    // //     test_query + i*vecdim
    // // );

    //     //openmp PQ-SIMD
    // //     auto res = flat_search_pq_adc_soa_block_openmp(
    // //     base,
    // //     test_query + i * vecdim,
    // //     base_number,
    // //     vecdim,
    // //     k,
    // //     pq_p,
    // //     pq_index
    // // );

    // // //openmp flat-SIMD
    // // auto res = flat_search_neon_openmp(
    // //     base,
    // //     test_query + i * vecdim,
    // //     base_number,
    // //     vecdim,
    // //     k
    // // );

    // // //IVF-SIMD优化调用
    // // auto res = flat_search_ivf_simd(
    // //     base,
    // //     test_query + i * vecdim,
    // //     base_number,
    // //     vecdim,
    // //     k,
    // //     IVF_NPROBE,
    // //     ivf_index
    // // );

    // // //IVF-SIMD openmp
    // // auto res = flat_search_ivf_simd(
    // //     base,
    // //     test_query + i * vecdim,
    // //     base_number,
    // //     vecdim,
    // //     k,
    // //     IVF_NPROBE,
    // //     ivf_index
    // // );

    // // //IVF-pthread
    // //     auto res = flat_search_ivf_simd_pthread_static(
    // //         ivf_static_pool,
    // //         test_query + i * vecdim,
    // //         IVF_NPROBE,
    // //         ivf_index
    // //     );

    // // //IVF-PQ baseline
    // // auto res = flat_search_ivf_pq_adc(
    // //     base,
    // //     test_query + i * vecdim,
    // //     base_number,
    // //     vecdim,
    // //     k,
    // //     IVF_NPROBE,
    // //     IVFPQ_TOP_P,
    // //     ivf_index,
    // //     pq_index
    // // );

    // // //IVF-PQ-pthread
    // // auto res = flat_search_ivf_pq_adc(
    // //     base,
    // //     test_query + i * vecdim,
    // //     base_number,
    // //     vecdim,
    // //     k,
    // //     IVF_NPROBE,
    // //     IVFPQ_TOP_P,
    // //     ivf_index,
    // //     pq_index
    // // );

    // // //IVF-PQ openmp
    // // auto res = flat_search_ivf_pq_adc_openmp(
    // //     base,
    // //     test_query + i * vecdim,
    // //     base_number,
    // //     vecdim,
    // //     k,
    // //     IVF_NPROBE,
    // //     IVFPQ_TOP_P,
    // //     ivf_index,
    // //     pq_index
    // // );

    // // //HNSW
    // // auto res = hnsw_search_baseline(
    // //     hnsw_index,
    // //     test_query + i * vecdim,
    // //     k
    // // );

    // //IVF-HNSW
    // auto res = search_ivf_hnsw(
    //     test_query + i * vecdim,
    //     vecdim,
    //     k,
    //     IVF_NPROBE,
    //     ivf_index,
    //     ivf_hnsw_index
    // );


    //     struct timeval newVal;
    //     ret = gettimeofday(&newVal, NULL);
    //     int64_t diff = (newVal.tv_sec * Converter + newVal.tv_usec) - (val.tv_sec * Converter + val.tv_usec);

    //     std::set<uint32_t> gtset;
    //     for(int j = 0; j < k; ++j){
    //         int t = test_gt[j + i*test_gt_d];
    //         gtset.insert(t);
    //     }

    //     size_t acc = 0;
    //     while (res.size()) {   
    //         int x = res.top().second;
    //         if(gtset.find(x) != gtset.end()){
    //             ++acc;
    //         }
    //         res.pop();
    //     }
    //     float recall = (float)acc/k;

    //     results[i] = {recall, diff};
    // }



    //query间并行
    const unsigned long Converter = 1000 * 1000;
    struct timeval batch_start, batch_end;

    gettimeofday(&batch_start, NULL);

    #pragma omp parallel for num_threads(OMP_NUM_THREADS) schedule(static)
    for (int i = 0; i < (int)test_number; ++i) {
        auto res = hnsw_search_baseline(
            hnsw_index,
            test_query + (size_t)i * vecdim,
            k
        );

        std::set<uint32_t> gtset;
        for (int j = 0; j < (int)k; ++j) {
            int t = test_gt[j + i * test_gt_d];
            gtset.insert(t);
        }

        size_t acc = 0;
        while (res.size()) {
            int x = res.top().second;
            if (gtset.find(x) != gtset.end()) {
                ++acc;
            }
            res.pop();
        }

        float recall = (float)acc / k;

        results[i] = {recall, 0};
    }

    gettimeofday(&batch_end, NULL);

    int64_t batch_diff =
        (batch_end.tv_sec * Converter + batch_end.tv_usec) -
        (batch_start.tv_sec * Converter + batch_start.tv_usec);

    //destroy_pq_pthread_static_pool(pq_static_pool);

    //destroy_flat_pthread_static_pool(flat_static_pool);

    //destroy_ivf_pthread_static_pool(ivf_static_pool);

    //destroy_ivf_pq_pthread_static_pool(ivf_pq_static_pool);

    delete hnsw_index;
    delete hnsw_space;

    //destroy_ivf_hnsw_index(ivf_hnsw_index);

    // float avg_recall = 0, avg_latency = 0;
    // for(int i = 0; i < test_number; ++i) {
    //     avg_recall += results[i].recall;
    //     avg_latency += results[i].latency;
    // }

    // // 浮点误差可能导致一些精确算法平均recall不是1
    // std::cout << "average recall: "<<avg_recall / test_number<<"\n";
    // std::cout << "average latency (us): "<<avg_latency / test_number<<"\n";
    // //delete[] base_i8;


    //query间并行
    float avg_recall = 0;
    for (int i = 0; i < (int)test_number; ++i) {
        avg_recall += results[i].recall;
    }

    double avg_latency = (double)batch_diff / test_number;

    std::cout << "average recall: " << avg_recall / test_number << "\n";
    std::cout << "average latency (us): " << avg_latency << "\n";
    std::cout << "batch total latency (us): " << batch_diff << "\n";
    std::cout << "query parallel threads: " << OMP_NUM_THREADS << "\n";


    return 0;
}
