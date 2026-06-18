#include <cuda_runtime.h>

#include <algorithm>
#include <cfloat>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr int kDefaultTopK = 10;
constexpr int kDefaultQueryCount = 200;
constexpr int kDefaultBatchSize = 32;
constexpr int kDistanceThreads = 128;
constexpr int kTopkThreads = 256;
constexpr int kMaxTopK = 10;
constexpr uint32_t kInvalidId = UINT32_MAX;
constexpr uint32_t kDefaultIvfNlist = 64;
constexpr uint32_t kDefaultIvfNprobe = 8;
constexpr uint32_t kDefaultIvfTrainSample = 4096;
constexpr uint32_t kDefaultIvfIters = 5;
constexpr uint32_t kDefaultPqM = 8;
constexpr uint32_t kDefaultPqKs = 256;
constexpr uint32_t kDefaultPqTrainSample = 4096;
constexpr uint32_t kDefaultPqIters = 4;
constexpr uint32_t kDefaultIvfpqTopP = 8192;

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err__ = (call);                                            \
        if (err__ != cudaSuccess) {                                            \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__     \
                      << " " << cudaGetErrorString(err__) << "\n";           \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

int InitializeCudaDevice() {
    const char* visible_devices = std::getenv("CUDA_VISIBLE_DEVICES");
    if (visible_devices != nullptr) {
        std::string value = visible_devices;
        if (value.empty() || value == "-1") {
            std::cerr << "CUDA_VISIBLE_DEVICES hides all GPUs; clearing it "
                         "for this process.\n";
#ifdef _WIN32
            _putenv_s("CUDA_VISIBLE_DEVICES", "");
#else
            unsetenv("CUDA_VISIBLE_DEVICES");
#endif
        }
    }

    int device_count = 0;
    cudaError_t count_status = cudaGetDeviceCount(&device_count);
    if (count_status != cudaSuccess) {
        std::cerr << "cudaGetDeviceCount failed: "
                  << cudaGetErrorString(count_status) << "\n";
        std::exit(EXIT_FAILURE);
    }

    if (device_count <= 0) {
        std::cerr << "No CUDA device is visible to this process.\n";
        std::exit(EXIT_FAILURE);
    }

    int device = 0;
    CUDA_CHECK(cudaSetDevice(device));

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    std::cout << "GPU: " << prop.name << "\n";

    return device;
}

template <typename T>
std::vector<T> LoadData(const std::string& path, uint32_t& n, uint32_t& d) {
    std::ifstream fin(path, std::ios::binary);
    if (!fin) {
        std::cerr << "failed to open " << path << "\n";
        std::exit(EXIT_FAILURE);
    }

    fin.read(reinterpret_cast<char*>(&n), 4);
    fin.read(reinterpret_cast<char*>(&d), 4);
    std::vector<T> data(static_cast<size_t>(n) * d);
    fin.read(reinterpret_cast<char*>(data.data()),
             static_cast<std::streamsize>(data.size() * sizeof(T)));

    if (!fin) {
        std::cerr << "failed to read complete data from " << path << "\n";
        std::exit(EXIT_FAILURE);
    }

    std::cerr << "load " << path << " n=" << n << " d=" << d
              << " bytes/element=" << sizeof(T) << "\n";
    return data;
}

__device__ __forceinline__ void insert_topk(float dist,
                                            uint32_t id,
                                            float local_dist[kMaxTopK],
                                            uint32_t local_id[kMaxTopK],
                                            int k) {
    if (dist >= local_dist[k - 1]) {
        return;
    }

    int pos = k - 1;
    while (pos > 0 && dist < local_dist[pos - 1]) {
        local_dist[pos] = local_dist[pos - 1];
        local_id[pos] = local_id[pos - 1];
        --pos;
    }
    local_dist[pos] = dist;
    local_id[pos] = id;
}

// Matrix-multiplication style baseline:
// base is n x d, query batch is m x d, output distances are stored as m x n.
__global__ void compute_ip_distance_kernel(const float* __restrict__ base,
                                           const float* __restrict__ queries,
                                           float* __restrict__ distances,
                                           uint32_t n,
                                           uint32_t d,
                                           uint32_t batch_size) {
    uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t q = blockIdx.y;

    if (row >= n || q >= batch_size) {
        return;
    }

    const float* base_vec = base + static_cast<size_t>(row) * d;
    const float* query_vec = queries + static_cast<size_t>(q) * d;

    float dot = 0.0f;
    for (uint32_t dim = 0; dim < d; ++dim) {
        dot += base_vec[dim] * query_vec[dim];
    }

    distances[static_cast<size_t>(q) * n + row] = 1.0f - dot;
}

__global__ void exact_topk_kernel(const float* __restrict__ distances,
                                  float* __restrict__ out_dist,
                                  uint32_t* __restrict__ out_id,
                                  uint32_t n,
                                  int k) {
    __shared__ float shared_dist[kTopkThreads * kMaxTopK];
    __shared__ uint32_t shared_id[kTopkThreads * kMaxTopK];

    uint32_t q = blockIdx.x;
    int tid = threadIdx.x;

    float local_dist[kMaxTopK];
    uint32_t local_id[kMaxTopK];
    for (int i = 0; i < k; ++i) {
        local_dist[i] = FLT_MAX;
        local_id[i] = UINT32_MAX;
    }

    const float* query_dist = distances + static_cast<size_t>(q) * n;
    for (uint32_t row = tid; row < n; row += blockDim.x) {
        insert_topk(query_dist[row], row, local_dist, local_id, k);
    }

    for (int i = 0; i < k; ++i) {
        shared_dist[tid * kMaxTopK + i] = local_dist[i];
        shared_id[tid * kMaxTopK + i] = local_id[i];
    }
    __syncthreads();

    if (tid == 0) {
        float best_dist[kMaxTopK];
        uint32_t best_id[kMaxTopK];
        for (int i = 0; i < k; ++i) {
            best_dist[i] = FLT_MAX;
            best_id[i] = UINT32_MAX;
        }

        for (int t = 0; t < blockDim.x; ++t) {
            for (int i = 0; i < k; ++i) {
                insert_topk(shared_dist[t * kMaxTopK + i],
                            shared_id[t * kMaxTopK + i],
                            best_dist,
                            best_id,
                            k);
            }
        }

        for (int i = 0; i < k; ++i) {
            out_dist[static_cast<size_t>(q) * k + i] = best_dist[i];
            out_id[static_cast<size_t>(q) * k + i] = best_id[i];
        }
    }
}

// Exact matrix-baseline optimization:
// Fuse distance computation and top-k reduction so the full batch x base matrix
// is not written to and read back from global memory.
__global__ void fused_exact_topk_kernel(const float* __restrict__ base,
                                        const float* __restrict__ queries,
                                        float* __restrict__ out_dist,
                                        uint32_t* __restrict__ out_id,
                                        uint32_t n,
                                        uint32_t d,
                                        int k) {
    __shared__ float shared_dist[kTopkThreads * kMaxTopK];
    __shared__ uint32_t shared_id[kTopkThreads * kMaxTopK];

    uint32_t q = blockIdx.x;
    int tid = threadIdx.x;

    float local_dist[kMaxTopK];
    uint32_t local_id[kMaxTopK];
    for (int i = 0; i < k; ++i) {
        local_dist[i] = FLT_MAX;
        local_id[i] = kInvalidId;
    }

    const float* query_vec = queries + static_cast<size_t>(q) * d;
    for (uint32_t row = tid; row < n; row += blockDim.x) {
        const float* base_vec = base + static_cast<size_t>(row) * d;

        float dot = 0.0f;
        for (uint32_t dim = 0; dim < d; ++dim) {
            dot += base_vec[dim] * query_vec[dim];
        }

        insert_topk(1.0f - dot, row, local_dist, local_id, k);
    }

    for (int i = 0; i < k; ++i) {
        shared_dist[tid * kMaxTopK + i] = local_dist[i];
        shared_id[tid * kMaxTopK + i] = local_id[i];
    }
    __syncthreads();

    if (tid == 0) {
        float best_dist[kMaxTopK];
        uint32_t best_id[kMaxTopK];
        for (int i = 0; i < k; ++i) {
            best_dist[i] = FLT_MAX;
            best_id[i] = kInvalidId;
        }

        for (int t = 0; t < blockDim.x; ++t) {
            for (int i = 0; i < k; ++i) {
                if (shared_id[t * kMaxTopK + i] != kInvalidId) {
                    insert_topk(shared_dist[t * kMaxTopK + i],
                                shared_id[t * kMaxTopK + i],
                                best_dist,
                                best_id,
                                k);
                }
            }
        }

        for (int i = 0; i < k; ++i) {
            out_dist[static_cast<size_t>(q) * k + i] = best_dist[i];
            out_id[static_cast<size_t>(q) * k + i] = best_id[i];
        }
    }
}

__global__ void compute_candidate_ip_distance_kernel(
    const float* __restrict__ base,
    const float* __restrict__ queries,
    const uint32_t* __restrict__ candidate_ids,
    float* __restrict__ distances,
    uint32_t d,
    uint32_t candidate_bound,
    uint32_t batch_size) {
    uint32_t slot = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t q = blockIdx.y;

    if (slot >= candidate_bound || q >= batch_size) {
        return;
    }

    size_t pos = static_cast<size_t>(q) * candidate_bound + slot;
    uint32_t id = candidate_ids[pos];
    if (id == kInvalidId) {
        distances[pos] = FLT_MAX;
        return;
    }

    const float* base_vec = base + static_cast<size_t>(id) * d;
    const float* query_vec = queries + static_cast<size_t>(q) * d;

    float dot = 0.0f;
    for (uint32_t dim = 0; dim < d; ++dim) {
        dot += base_vec[dim] * query_vec[dim];
    }

    distances[pos] = 1.0f - dot;
}

__global__ void exact_topk_candidates_kernel(
    const float* __restrict__ distances,
    const uint32_t* __restrict__ candidate_ids,
    float* __restrict__ out_dist,
    uint32_t* __restrict__ out_id,
    uint32_t candidate_bound,
    int k) {
    __shared__ float shared_dist[kTopkThreads * kMaxTopK];
    __shared__ uint32_t shared_id[kTopkThreads * kMaxTopK];

    uint32_t q = blockIdx.x;
    int tid = threadIdx.x;

    float local_dist[kMaxTopK];
    uint32_t local_id[kMaxTopK];
    for (int i = 0; i < k; ++i) {
        local_dist[i] = FLT_MAX;
        local_id[i] = kInvalidId;
    }

    const float* query_dist = distances + static_cast<size_t>(q) * candidate_bound;
    const uint32_t* query_ids = candidate_ids + static_cast<size_t>(q) * candidate_bound;
    for (uint32_t slot = tid; slot < candidate_bound; slot += blockDim.x) {
        uint32_t id = query_ids[slot];
        if (id != kInvalidId) {
            insert_topk(query_dist[slot], id, local_dist, local_id, k);
        }
    }

    for (int i = 0; i < k; ++i) {
        shared_dist[tid * kMaxTopK + i] = local_dist[i];
        shared_id[tid * kMaxTopK + i] = local_id[i];
    }
    __syncthreads();

    if (tid == 0) {
        float best_dist[kMaxTopK];
        uint32_t best_id[kMaxTopK];
        for (int i = 0; i < k; ++i) {
            best_dist[i] = FLT_MAX;
            best_id[i] = kInvalidId;
        }

        for (int t = 0; t < blockDim.x; ++t) {
            for (int i = 0; i < k; ++i) {
                if (shared_id[t * kMaxTopK + i] != kInvalidId) {
                    insert_topk(shared_dist[t * kMaxTopK + i],
                                shared_id[t * kMaxTopK + i],
                                best_dist,
                                best_id,
                                k);
                }
            }
        }

        for (int i = 0; i < k; ++i) {
            out_dist[static_cast<size_t>(q) * k + i] = best_dist[i];
            out_id[static_cast<size_t>(q) * k + i] = best_id[i];
        }
    }
}

__global__ void compute_pq_candidate_distance_kernel(
    const uint8_t* __restrict__ pq_codes,
    const float* __restrict__ pq_lut,
    const uint32_t* __restrict__ candidate_ids,
    float* __restrict__ distances,
    uint32_t pq_m,
    uint32_t pq_ks,
    uint32_t candidate_bound,
    uint32_t batch_size) {
    uint32_t slot = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t q = blockIdx.y;

    if (slot >= candidate_bound || q >= batch_size) {
        return;
    }

    size_t pos = static_cast<size_t>(q) * candidate_bound + slot;
    uint32_t id = candidate_ids[pos];
    if (id == kInvalidId) {
        distances[pos] = FLT_MAX;
        return;
    }

    const uint8_t* code = pq_codes + static_cast<size_t>(id) * pq_m;
    const float* query_lut = pq_lut + static_cast<size_t>(q) * pq_m * pq_ks;

    float score = 0.0f;
    for (uint32_t m = 0; m < pq_m; ++m) {
        uint8_t cid = code[m];
        score += query_lut[static_cast<size_t>(m) * pq_ks + cid];
    }

    distances[pos] = 1.0f - score;
}

struct IVFIndex {
    uint32_t nlist = kDefaultIvfNlist;
    uint32_t nprobe = kDefaultIvfNprobe;
    uint32_t vecdim = 0;
    std::vector<float> centers;
    std::vector<std::vector<uint32_t> > lists;
    uint32_t candidate_bound = 0;
};

struct IVFQueryPlan {
    uint32_t query_id = 0;
    uint32_t primary_center = 0;
    uint32_t candidate_count = 0;
    std::vector<uint32_t> probes;
};

struct PQIndex {
    uint32_t M = kDefaultPqM;
    uint32_t Ks = kDefaultPqKs;
    uint32_t subdim = 0;
    std::vector<float> codebook;
    std::vector<uint8_t> codes;
};

float inner_product_cpu(const float* a, const float* b, uint32_t d) {
    float sum = 0.0f;
    for (uint32_t i = 0; i < d; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

void normalize_vector(float* v, uint32_t d) {
    float norm2 = 0.0f;
    for (uint32_t i = 0; i < d; ++i) {
        norm2 += v[i] * v[i];
    }
    if (norm2 <= 1e-20f) {
        return;
    }

    float inv_norm = 1.0f / std::sqrt(norm2);
    for (uint32_t i = 0; i < d; ++i) {
        v[i] *= inv_norm;
    }
}

uint32_t nearest_center_ip(const float* vec,
                           const std::vector<float>& centers,
                           uint32_t nlist,
                           uint32_t d) {
    uint32_t best = 0;
    float best_score = -FLT_MAX;
    for (uint32_t c = 0; c < nlist; ++c) {
        float score = inner_product_cpu(vec, centers.data() + static_cast<size_t>(c) * d, d);
        if (score > best_score) {
            best_score = score;
            best = c;
        }
    }
    return best;
}

float l2_distance_cpu(const float* a, const float* b, uint32_t d) {
    float sum = 0.0f;
    for (uint32_t i = 0; i < d; ++i) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

uint8_t nearest_pq_centroid_l2(const float* vec,
                               const float* centroids,
                               uint32_t ks,
                               uint32_t subdim) {
    uint32_t best = 0;
    float best_dist = FLT_MAX;

    for (uint32_t c = 0; c < ks; ++c) {
        float dist = l2_distance_cpu(vec,
                                     centroids + static_cast<size_t>(c) * subdim,
                                     subdim);
        if (dist < best_dist) {
            best_dist = dist;
            best = c;
        }
    }

    return static_cast<uint8_t>(best);
}

std::vector<uint32_t> top_centers_ip(const float* query,
                                     const std::vector<float>& centers,
                                     uint32_t nlist,
                                     uint32_t nprobe,
                                     uint32_t d) {
    std::vector<std::pair<float, uint32_t> > scores;
    scores.reserve(nlist);

    for (uint32_t c = 0; c < nlist; ++c) {
        float score = inner_product_cpu(query, centers.data() + static_cast<size_t>(c) * d, d);
        scores.push_back(std::make_pair(score, c));
    }

    uint32_t probe = std::min(nprobe, nlist);
    std::partial_sort(scores.begin(),
                      scores.begin() + probe,
                      scores.end(),
                      [](const auto& a, const auto& b) {
                          return a.first > b.first;
                      });

    std::vector<uint32_t> result;
    result.reserve(probe);
    for (uint32_t i = 0; i < probe; ++i) {
        result.push_back(scores[i].second);
    }
    return result;
}

uint32_t compute_candidate_bound(const std::vector<std::vector<uint32_t> >& lists,
                                 uint32_t nprobe) {
    std::vector<uint32_t> sizes;
    sizes.reserve(lists.size());
    for (const auto& list : lists) {
        sizes.push_back(static_cast<uint32_t>(list.size()));
    }

    uint32_t probe = std::min<uint32_t>(nprobe, static_cast<uint32_t>(sizes.size()));
    std::partial_sort(sizes.begin(),
                      sizes.begin() + probe,
                      sizes.end(),
                      [](uint32_t a, uint32_t b) {
                          return a > b;
                      });

    uint32_t bound = 0;
    for (uint32_t i = 0; i < probe; ++i) {
        bound += sizes[i];
    }
    return std::max<uint32_t>(bound, kMaxTopK);
}

IVFIndex build_ivf_index_cpu(const std::vector<float>& base,
                             uint32_t base_n,
                             uint32_t d,
                             uint32_t nlist,
                             uint32_t nprobe) {
    IVFIndex index;
    index.nlist = nlist;
    index.nprobe = std::min(nprobe, nlist);
    index.vecdim = d;
    index.centers.assign(static_cast<size_t>(nlist) * d, 0.0f);
    index.lists.assign(nlist, std::vector<uint32_t>());

    uint32_t sample_n = std::min<uint32_t>(kDefaultIvfTrainSample, base_n);
    std::vector<uint32_t> sample_ids(sample_n);
    for (uint32_t i = 0; i < sample_n; ++i) {
        sample_ids[i] = static_cast<uint32_t>(
            (static_cast<uint64_t>(i) * base_n) / sample_n);
    }

    for (uint32_t c = 0; c < nlist; ++c) {
        uint32_t sample_pos = static_cast<uint32_t>(
            (static_cast<uint64_t>(c) * sample_n) / nlist);
        uint32_t id = sample_ids[sample_pos];
        std::copy(base.data() + static_cast<size_t>(id) * d,
                  base.data() + static_cast<size_t>(id + 1) * d,
                  index.centers.data() + static_cast<size_t>(c) * d);
        normalize_vector(index.centers.data() + static_cast<size_t>(c) * d, d);
    }

    std::vector<uint32_t> assignment(sample_n, 0);
    for (uint32_t iter = 0; iter < kDefaultIvfIters; ++iter) {
        for (uint32_t s = 0; s < sample_n; ++s) {
            const float* vec = base.data() + static_cast<size_t>(sample_ids[s]) * d;
            assignment[s] = nearest_center_ip(vec, index.centers, nlist, d);
        }

        std::vector<float> sums(static_cast<size_t>(nlist) * d, 0.0f);
        std::vector<uint32_t> counts(nlist, 0);

        for (uint32_t s = 0; s < sample_n; ++s) {
            uint32_t c = assignment[s];
            const float* vec = base.data() + static_cast<size_t>(sample_ids[s]) * d;
            float* sum = sums.data() + static_cast<size_t>(c) * d;
            for (uint32_t j = 0; j < d; ++j) {
                sum[j] += vec[j];
            }
            counts[c]++;
        }

        for (uint32_t c = 0; c < nlist; ++c) {
            if (counts[c] == 0) {
                continue;
            }
            float* center = index.centers.data() + static_cast<size_t>(c) * d;
            const float* sum = sums.data() + static_cast<size_t>(c) * d;
            float inv = 1.0f / static_cast<float>(counts[c]);
            for (uint32_t j = 0; j < d; ++j) {
                center[j] = sum[j] * inv;
            }
            normalize_vector(center, d);
        }
    }

    for (uint32_t i = 0; i < base_n; ++i) {
        const float* vec = base.data() + static_cast<size_t>(i) * d;
        uint32_t c = nearest_center_ip(vec, index.centers, nlist, d);
        index.lists[c].push_back(i);
    }

    index.candidate_bound = compute_candidate_bound(index.lists, index.nprobe);

    uint64_t total = 0;
    uint32_t non_empty = 0;
    uint32_t max_size = 0;
    for (const auto& list : index.lists) {
        total += list.size();
        if (!list.empty()) {
            non_empty++;
        }
        max_size = std::max<uint32_t>(max_size, static_cast<uint32_t>(list.size()));
    }

    std::cout << "IVF built: nlist=" << index.nlist
              << " nprobe=" << index.nprobe
              << " non_empty_lists=" << non_empty
              << " avg_list_size=" << (static_cast<double>(total) / index.nlist)
              << " max_list_size=" << max_size
              << " candidate_bound=" << index.candidate_bound << "\n";

    return index;
}

PQIndex build_pq_index_cpu(const std::vector<float>& base,
                           uint32_t base_n,
                           uint32_t d) {
    if (d % kDefaultPqM != 0) {
        std::cerr << "vecdim must be divisible by PQ M\n";
        std::exit(EXIT_FAILURE);
    }

    PQIndex pq;
    pq.M = kDefaultPqM;
    pq.Ks = kDefaultPqKs;
    pq.subdim = d / pq.M;
    pq.codebook.assign(static_cast<size_t>(pq.M) * pq.Ks * pq.subdim, 0.0f);
    pq.codes.assign(static_cast<size_t>(base_n) * pq.M, 0);

    uint32_t sample_n = std::min<uint32_t>(kDefaultPqTrainSample, base_n);
    std::vector<uint32_t> sample_ids(sample_n);
    for (uint32_t i = 0; i < sample_n; ++i) {
        sample_ids[i] = static_cast<uint32_t>(
            (static_cast<uint64_t>(i) * base_n) / sample_n);
    }

    for (uint32_t m = 0; m < pq.M; ++m) {
        float* centroids =
            pq.codebook.data() + static_cast<size_t>(m) * pq.Ks * pq.subdim;

        for (uint32_t c = 0; c < pq.Ks; ++c) {
            uint32_t sample_pos = static_cast<uint32_t>(
                (static_cast<uint64_t>(c) * sample_n) / pq.Ks);
            uint32_t id = sample_ids[sample_pos];
            const float* src =
                base.data() + static_cast<size_t>(id) * d + m * pq.subdim;
            std::copy(src, src + pq.subdim,
                      centroids + static_cast<size_t>(c) * pq.subdim);
        }

        std::vector<uint8_t> assignment(sample_n, 0);
        for (uint32_t iter = 0; iter < kDefaultPqIters; ++iter) {
            for (uint32_t s = 0; s < sample_n; ++s) {
                const float* vec =
                    base.data() + static_cast<size_t>(sample_ids[s]) * d +
                    m * pq.subdim;
                assignment[s] =
                    nearest_pq_centroid_l2(vec, centroids, pq.Ks, pq.subdim);
            }

            std::vector<float> sums(static_cast<size_t>(pq.Ks) * pq.subdim,
                                    0.0f);
            std::vector<uint32_t> counts(pq.Ks, 0);

            for (uint32_t s = 0; s < sample_n; ++s) {
                uint32_t cid = assignment[s];
                const float* vec =
                    base.data() + static_cast<size_t>(sample_ids[s]) * d +
                    m * pq.subdim;
                float* sum = sums.data() + static_cast<size_t>(cid) * pq.subdim;
                for (uint32_t j = 0; j < pq.subdim; ++j) {
                    sum[j] += vec[j];
                }
                counts[cid]++;
            }

            for (uint32_t c = 0; c < pq.Ks; ++c) {
                if (counts[c] == 0) {
                    continue;
                }
                float* center = centroids + static_cast<size_t>(c) * pq.subdim;
                const float* sum = sums.data() + static_cast<size_t>(c) * pq.subdim;
                float inv = 1.0f / static_cast<float>(counts[c]);
                for (uint32_t j = 0; j < pq.subdim; ++j) {
                    center[j] = sum[j] * inv;
                }
            }
        }

        std::cout << "PQ trained subspace " << m << "\n";
    }

    for (uint32_t i = 0; i < base_n; ++i) {
        for (uint32_t m = 0; m < pq.M; ++m) {
            const float* vec =
                base.data() + static_cast<size_t>(i) * d + m * pq.subdim;
            const float* centroids =
                pq.codebook.data() + static_cast<size_t>(m) * pq.Ks * pq.subdim;
            pq.codes[static_cast<size_t>(i) * pq.M + m] =
                nearest_pq_centroid_l2(vec, centroids, pq.Ks, pq.subdim);
        }
    }

    std::cout << "PQ built: M=" << pq.M
              << " Ks=" << pq.Ks
              << " subdim=" << pq.subdim
              << " code_bytes_per_vector=" << pq.M << "\n";

    return pq;
}

void build_pq_lut_batch(const std::vector<float>& grouped_query_batch,
                        uint32_t current_batch,
                        uint32_t query_d,
                        const PQIndex& pq,
                        std::vector<float>& pq_lut) {
    pq_lut.assign(static_cast<size_t>(current_batch) * pq.M * pq.Ks, 0.0f);

    for (uint32_t q = 0; q < current_batch; ++q) {
        const float* query =
            grouped_query_batch.data() + static_cast<size_t>(q) * query_d;

        for (uint32_t m = 0; m < pq.M; ++m) {
            const float* query_sub = query + m * pq.subdim;
            const float* centroids =
                pq.codebook.data() + static_cast<size_t>(m) * pq.Ks * pq.subdim;
            float* lut =
                pq_lut.data() + (static_cast<size_t>(q) * pq.M + m) * pq.Ks;

            for (uint32_t c = 0; c < pq.Ks; ++c) {
                lut[c] = inner_product_cpu(query_sub,
                                           centroids + static_cast<size_t>(c) * pq.subdim,
                                           pq.subdim);
            }
        }
    }
}

void select_top_p_candidates_from_pq(
    const std::vector<float>& pq_distances,
    const std::vector<uint32_t>& candidate_ids,
    uint32_t current_batch,
    uint32_t candidate_stride,
    uint32_t top_p,
    std::vector<uint32_t>& refine_ids,
    uint64_t& refine_candidate_sum) {
    refine_ids.assign(static_cast<size_t>(current_batch) * top_p, kInvalidId);

    for (uint32_t q = 0; q < current_batch; ++q) {
        std::vector<std::pair<float, uint32_t> > scored;
        scored.reserve(candidate_stride);

        size_t base = static_cast<size_t>(q) * candidate_stride;
        for (uint32_t slot = 0; slot < candidate_stride; ++slot) {
            uint32_t id = candidate_ids[base + slot];
            if (id != kInvalidId) {
                scored.push_back(std::make_pair(pq_distances[base + slot], id));
            }
        }

        uint32_t keep = std::min<uint32_t>(top_p, static_cast<uint32_t>(scored.size()));
        if (keep > 0) {
            std::partial_sort(scored.begin(),
                              scored.begin() + keep,
                              scored.end(),
                              [](const auto& a, const auto& b) {
                                  return a.first < b.first;
                              });
        }

        refine_candidate_sum += keep;
        for (uint32_t i = 0; i < keep; ++i) {
            refine_ids[static_cast<size_t>(q) * top_p + i] = scored[i].second;
        }
    }
}

std::vector<IVFQueryPlan> build_grouped_ivf_query_plans(
    const std::vector<float>& queries,
    uint32_t query_count,
    uint32_t d,
    const IVFIndex& index) {
    std::vector<IVFQueryPlan> plans(query_count);

    for (uint32_t i = 0; i < query_count; ++i) {
        const float* query = queries.data() + static_cast<size_t>(i) * d;
        plans[i].query_id = i;
        plans[i].probes = top_centers_ip(query,
                                         index.centers,
                                         index.nlist,
                                         index.nprobe,
                                         d);
        plans[i].primary_center = plans[i].probes.empty() ? 0 : plans[i].probes[0];

        uint32_t count = 0;
        for (uint32_t list_id : plans[i].probes) {
            count += static_cast<uint32_t>(index.lists[list_id].size());
        }
        plans[i].candidate_count = count;
    }

    std::sort(plans.begin(),
              plans.end(),
              [](const IVFQueryPlan& a, const IVFQueryPlan& b) {
                  if (a.primary_center != b.primary_center) {
                      return a.primary_center < b.primary_center;
                  }
                  if (a.candidate_count != b.candidate_count) {
                      return a.candidate_count < b.candidate_count;
                  }
                  return a.query_id < b.query_id;
              });

    return plans;
}

float recall_at_k(const std::vector<uint32_t>& result_ids,
                  const std::vector<int>& gt,
                  uint32_t gt_dim,
                  uint32_t query_offset,
                  uint32_t batch_size,
                  int k) {
    float sum_recall = 0.0f;

    for (uint32_t q = 0; q < batch_size; ++q) {
        std::set<uint32_t> gt_set;
        for (int j = 0; j < k; ++j) {
            gt_set.insert(static_cast<uint32_t>(
                gt[static_cast<size_t>(query_offset + q) * gt_dim + j]));
        }

        int correct = 0;
        for (int j = 0; j < k; ++j) {
            uint32_t id = result_ids[static_cast<size_t>(q) * k + j];
            if (gt_set.find(id) != gt_set.end()) {
                ++correct;
            }
        }
        sum_recall += static_cast<float>(correct) / k;
    }

    return sum_recall;
}

float recall_at_k_planned(const std::vector<uint32_t>& result_ids,
                          const std::vector<int>& gt,
                          uint32_t gt_dim,
                          const std::vector<IVFQueryPlan>& plans,
                          uint32_t plan_offset,
                          uint32_t batch_size,
                          int k) {
    float sum_recall = 0.0f;

    for (uint32_t q = 0; q < batch_size; ++q) {
        uint32_t query_id = plans[plan_offset + q].query_id;

        std::set<uint32_t> gt_set;
        for (int j = 0; j < k; ++j) {
            gt_set.insert(static_cast<uint32_t>(
                gt[static_cast<size_t>(query_id) * gt_dim + j]));
        }

        int correct = 0;
        for (int j = 0; j < k; ++j) {
            uint32_t id = result_ids[static_cast<size_t>(q) * k + j];
            if (gt_set.find(id) != gt_set.end()) {
                ++correct;
            }
        }
        sum_recall += static_cast<float>(correct) / k;
    }

    return sum_recall;
}

}  // namespace

int main(int argc, char** argv) {
    int query_count = argc > 1 ? std::atoi(argv[1]) : kDefaultQueryCount;
    int batch_size = argc > 2 ? std::atoi(argv[2]) : kDefaultBatchSize;

    bool use_ivf = false;
    bool use_ivfpq = false;
    bool use_fused_exact = false;
    int path_arg = 3;
    if (argc > path_arg) {
        std::string mode = argv[path_arg];
        if (mode == "ivf" || mode == "--ivf") {
            use_ivf = true;
            path_arg++;
        } else if (mode == "ivfpq" || mode == "--ivfpq") {
            use_ivf = true;
            use_ivfpq = true;
            path_arg++;
        } else if (mode == "fused" || mode == "--fused") {
            use_fused_exact = true;
            path_arg++;
        }
    }

    const std::string base_path =
        argc > path_arg ? argv[path_arg] : "DEEP100K.base.100k.fbin";
    const std::string query_path =
        argc > path_arg + 1 ? argv[path_arg + 1] : "DEEP100K.query.fbin";
    const std::string gt_path =
        argc > path_arg + 2 ? argv[path_arg + 2] : "DEEP100K.gt.query.100k.top100.bin";

    if (query_count <= 0 || batch_size <= 0) {
        std::cerr << "query_count and batch_size must be positive\n";
        return EXIT_FAILURE;
    }

    const int k = kDefaultTopK;

    InitializeCudaDevice();

    uint32_t base_n = 0, base_d = 0;
    uint32_t query_n = 0, query_d = 0;
    uint32_t gt_n = 0, gt_d = 0;

    std::vector<float> base = LoadData<float>(base_path, base_n, base_d);
    std::vector<float> queries = LoadData<float>(query_path, query_n, query_d);
    std::vector<int> gt = LoadData<int>(gt_path, gt_n, gt_d);

    if (base_d != query_d || query_n != gt_n || gt_d < static_cast<uint32_t>(k)) {
        std::cerr << "dataset dimensions do not match\n";
        return EXIT_FAILURE;
    }

    query_count = std::min(query_count, static_cast<int>(query_n));
    batch_size = std::min(batch_size, query_count);

    if (use_ivf) {
        if (use_ivfpq) {
            std::cout << "search mode: IVF-PQ GPU approximate baseline\n";
        } else {
            std::cout << "search mode: IVF GPU baseline\n";
        }
    } else if (use_fused_exact) {
        std::cout << "search mode: fused exact GPU top-k baseline\n";
    } else {
        std::cout << "search mode: exact GPU matrix baseline\n";
    }

    IVFIndex ivf_index;
    if (use_ivf) {
        ivf_index = build_ivf_index_cpu(base,
                                        base_n,
                                        base_d,
                                        kDefaultIvfNlist,
                                        kDefaultIvfNprobe);
    }

    PQIndex pq_index;
    if (use_ivfpq) {
        pq_index = build_pq_index_cpu(base, base_n, base_d);
    }

    std::vector<IVFQueryPlan> ivf_plans;
    if (use_ivf) {
        ivf_plans = build_grouped_ivf_query_plans(queries,
                                                  static_cast<uint32_t>(query_count),
                                                  query_d,
                                                  ivf_index);
        std::cout << "IVF grouping: sort queries by primary center, then "
                     "candidate count\n";
    }

    float* d_base = nullptr;
    float* d_queries = nullptr;
    float* d_distances = nullptr;
    float* d_out_dist = nullptr;
    uint32_t* d_out_id = nullptr;
    uint32_t* d_candidate_ids = nullptr;
    uint8_t* d_pq_codes = nullptr;
    float* d_pq_lut = nullptr;

    size_t max_batch = static_cast<size_t>(batch_size);
    size_t distance_cols = use_ivf ? ivf_index.candidate_bound : base_n;

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_base),
                          base.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_queries),
                          max_batch * query_d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_distances),
                          max_batch * distance_cols * sizeof(float)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_out_dist),
                          max_batch * k * sizeof(float)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_out_id),
                          max_batch * k * sizeof(uint32_t)));
    if (use_ivf) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_candidate_ids),
                              max_batch * distance_cols * sizeof(uint32_t)));
    }
    if (use_ivfpq) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_pq_codes),
                              pq_index.codes.size() * sizeof(uint8_t)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_pq_lut),
                              max_batch * pq_index.M * pq_index.Ks * sizeof(float)));
    }

    CUDA_CHECK(cudaMemcpy(d_base,
                          base.data(),
                          base.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    if (use_ivfpq) {
        CUDA_CHECK(cudaMemcpy(d_pq_codes,
                              pq_index.codes.data(),
                              pq_index.codes.size() * sizeof(uint8_t),
                              cudaMemcpyHostToDevice));
    }

    std::vector<uint32_t> result_ids(max_batch * k);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    float recall_sum = 0.0f;
    float gpu_ms_sum = 0.0f;
    uint64_t candidate_sum = 0;
    uint64_t candidate_slot_sum = 0;
    uint64_t refine_candidate_sum = 0;
    uint64_t refine_slot_sum = 0;
    uint32_t processed = 0;
    std::vector<uint32_t> candidate_ids;
    std::vector<float> grouped_query_batch;
    std::vector<float> pq_lut;
    std::vector<float> pq_distances_host;
    std::vector<uint32_t> refine_candidate_ids;
    if (use_ivf) {
        candidate_ids.resize(max_batch * distance_cols, kInvalidId);
        grouped_query_batch.resize(max_batch * query_d);
    }

    for (uint32_t offset = 0; offset < static_cast<uint32_t>(query_count);
         offset += batch_size) {
        uint32_t current_batch =
            std::min<uint32_t>(batch_size, query_count - offset);

        uint32_t batch_distance_cols = static_cast<uint32_t>(distance_cols);
        if (use_ivf) {
            batch_distance_cols = kMaxTopK;
            for (uint32_t q = 0; q < current_batch; ++q) {
                const IVFQueryPlan& plan = ivf_plans[offset + q];
                batch_distance_cols =
                    std::max(batch_distance_cols, plan.candidate_count);
            }
            batch_distance_cols = std::min<uint32_t>(
                batch_distance_cols, static_cast<uint32_t>(distance_cols));

            std::fill(candidate_ids.begin(), candidate_ids.end(), kInvalidId);

            for (uint32_t q = 0; q < current_batch; ++q) {
                const IVFQueryPlan& plan = ivf_plans[offset + q];
                const float* src_query =
                    queries.data() + static_cast<size_t>(plan.query_id) * query_d;

                std::copy(src_query,
                          src_query + query_d,
                          grouped_query_batch.data() + static_cast<size_t>(q) * query_d);

                uint32_t slot = 0;
                for (uint32_t list_id : plan.probes) {
                    const std::vector<uint32_t>& list = ivf_index.lists[list_id];
                    for (uint32_t id : list) {
                        if (slot >= batch_distance_cols) {
                            break;
                        }
                        candidate_ids[static_cast<size_t>(q) * batch_distance_cols + slot] = id;
                        slot++;
                    }
                }
                candidate_sum += slot;
            }

            candidate_slot_sum +=
                static_cast<uint64_t>(current_batch) * batch_distance_cols;

            CUDA_CHECK(cudaMemcpy(d_queries,
                                  grouped_query_batch.data(),
                                  static_cast<size_t>(current_batch) * query_d *
                                      sizeof(float),
                                  cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_candidate_ids,
                                  candidate_ids.data(),
                                  static_cast<size_t>(current_batch) * batch_distance_cols *
                                      sizeof(uint32_t),
                                  cudaMemcpyHostToDevice));
            if (use_ivfpq) {
                build_pq_lut_batch(grouped_query_batch,
                                   current_batch,
                                   query_d,
                                   pq_index,
                                   pq_lut);
                CUDA_CHECK(cudaMemcpy(d_pq_lut,
                                      pq_lut.data(),
                                      pq_lut.size() * sizeof(float),
                                      cudaMemcpyHostToDevice));
            }
        } else {
            CUDA_CHECK(cudaMemcpy(d_queries,
                                  queries.data() + static_cast<size_t>(offset) * query_d,
                                  static_cast<size_t>(current_batch) * query_d *
                                      sizeof(float),
                                  cudaMemcpyHostToDevice));
        }

        dim3 distance_block(kDistanceThreads);
        dim3 distance_grid((batch_distance_cols + kDistanceThreads - 1) / kDistanceThreads,
                           current_batch);

        CUDA_CHECK(cudaEventRecord(start));
        if (use_ivf) {
            if (use_ivfpq) {
                compute_pq_candidate_distance_kernel<<<distance_grid, distance_block>>>(
                    d_pq_codes,
                    d_pq_lut,
                    d_candidate_ids,
                    d_distances,
                    pq_index.M,
                    pq_index.Ks,
                    batch_distance_cols,
                    current_batch);

                CUDA_CHECK(cudaEventRecord(stop));
                CUDA_CHECK(cudaEventSynchronize(stop));
                CUDA_CHECK(cudaGetLastError());

                float pq_batch_ms = 0.0f;
                CUDA_CHECK(cudaEventElapsedTime(&pq_batch_ms, start, stop));
                gpu_ms_sum += pq_batch_ms;

                pq_distances_host.resize(static_cast<size_t>(current_batch) *
                                         batch_distance_cols);
                CUDA_CHECK(cudaMemcpy(pq_distances_host.data(),
                                      d_distances,
                                      pq_distances_host.size() * sizeof(float),
                                      cudaMemcpyDeviceToHost));

                uint32_t refine_top_p =
                    std::min<uint32_t>(kDefaultIvfpqTopP, batch_distance_cols);
                select_top_p_candidates_from_pq(pq_distances_host,
                                                candidate_ids,
                                                current_batch,
                                                batch_distance_cols,
                                                refine_top_p,
                                                refine_candidate_ids,
                                                refine_candidate_sum);

                refine_slot_sum +=
                    static_cast<uint64_t>(current_batch) * refine_top_p;

                CUDA_CHECK(cudaMemcpy(d_candidate_ids,
                                      refine_candidate_ids.data(),
                                      refine_candidate_ids.size() * sizeof(uint32_t),
                                      cudaMemcpyHostToDevice));

                dim3 refine_grid((refine_top_p + kDistanceThreads - 1) /
                                     kDistanceThreads,
                                 current_batch);

                CUDA_CHECK(cudaEventRecord(start));
                compute_candidate_ip_distance_kernel<<<refine_grid, distance_block>>>(
                    d_base,
                    d_queries,
                    d_candidate_ids,
                    d_distances,
                    query_d,
                    refine_top_p,
                    current_batch);
                exact_topk_candidates_kernel<<<current_batch, kTopkThreads>>>(
                    d_distances,
                    d_candidate_ids,
                    d_out_dist,
                    d_out_id,
                    refine_top_p,
                    k);
            } else {
                compute_candidate_ip_distance_kernel<<<distance_grid, distance_block>>>(
                    d_base,
                    d_queries,
                    d_candidate_ids,
                    d_distances,
                    query_d,
                    batch_distance_cols,
                    current_batch);
                exact_topk_candidates_kernel<<<current_batch, kTopkThreads>>>(
                    d_distances,
                    d_candidate_ids,
                    d_out_dist,
                    d_out_id,
                    batch_distance_cols,
                    k);
            }
        } else {
            if (use_fused_exact) {
                fused_exact_topk_kernel<<<current_batch, kTopkThreads>>>(
                    d_base,
                    d_queries,
                    d_out_dist,
                    d_out_id,
                    base_n,
                    query_d,
                    k);
            } else {
                compute_ip_distance_kernel<<<distance_grid, distance_block>>>(
                    d_base, d_queries, d_distances, base_n, query_d, current_batch);
                exact_topk_kernel<<<current_batch, kTopkThreads>>>(
                    d_distances, d_out_dist, d_out_id, base_n, k);
            }
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaGetLastError());

        float batch_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&batch_ms, start, stop));
        gpu_ms_sum += batch_ms;

        CUDA_CHECK(cudaMemcpy(result_ids.data(),
                              d_out_id,
                              static_cast<size_t>(current_batch) * k *
                                  sizeof(uint32_t),
                              cudaMemcpyDeviceToHost));

        if (use_ivf) {
            recall_sum += recall_at_k_planned(result_ids,
                                              gt,
                                              gt_d,
                                              ivf_plans,
                                              offset,
                                              current_batch,
                                              k);
        } else {
            recall_sum += recall_at_k(result_ids,
                                      gt,
                                      gt_d,
                                      offset,
                                      current_batch,
                                      k);
        }
        processed += current_batch;
    }

    std::cout << "base: " << base_n << " x " << base_d << "\n";
    std::cout << "queries tested: " << processed << "\n";
    std::cout << "batch size: " << batch_size << "\n";
    std::cout << "top k: " << k << "\n";
    if (use_ivf) {
        std::cout << "ivf nlist: " << ivf_index.nlist << "\n";
        std::cout << "ivf nprobe: " << ivf_index.nprobe << "\n";
        std::cout << "average IVF candidates/query: "
                  << (static_cast<double>(candidate_sum) / processed) << "\n";
        std::cout << "average launched candidate slots/query: "
                  << (static_cast<double>(candidate_slot_sum) / processed) << "\n";
        std::cout << "candidate slot utilization: "
                  << (static_cast<double>(candidate_sum) /
                      static_cast<double>(candidate_slot_sum)) << "\n";
        if (use_ivfpq) {
            std::cout << "pq M: " << pq_index.M << "\n";
            std::cout << "pq Ks: " << pq_index.Ks << "\n";
            std::cout << "pq subdim: " << pq_index.subdim << "\n";
            std::cout << "pq code bytes/vector: " << pq_index.M << "\n";
            std::cout << "ivfpq top-p refine: " << kDefaultIvfpqTopP << "\n";
            std::cout << "average refined candidates/query: "
                      << (static_cast<double>(refine_candidate_sum) / processed) << "\n";
            std::cout << "average refined candidate slots/query: "
                      << (static_cast<double>(refine_slot_sum) / processed) << "\n";
        }
    }
    std::cout << "average recall@" << k << ": " << (recall_sum / processed)
              << "\n";
    std::cout << "average GPU time/query (us): "
              << (gpu_ms_sum * 1000.0f / processed) << "\n";
    std::cout << "total GPU kernel time (ms): " << gpu_ms_sum << "\n";

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_base));
    CUDA_CHECK(cudaFree(d_queries));
    CUDA_CHECK(cudaFree(d_distances));
    CUDA_CHECK(cudaFree(d_out_dist));
    CUDA_CHECK(cudaFree(d_out_id));
    if (d_candidate_ids != nullptr) {
        CUDA_CHECK(cudaFree(d_candidate_ids));
    }
    if (d_pq_codes != nullptr) {
        CUDA_CHECK(cudaFree(d_pq_codes));
    }
    if (d_pq_lut != nullptr) {
        CUDA_CHECK(cudaFree(d_pq_lut));
    }

    return 0;
}
