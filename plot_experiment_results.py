from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


OUT_DIR = Path("figures")
OUT_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 140,
    "savefig.dpi": 300,
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def annotate_bars(ax, bars, fmt="{:.1f}", dy=3):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            fmt.format(height),
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, dy),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def plot_batch_size():
    batch_sizes = [16, 32, 64, 128]
    time_us = [224.392, 183.491, 173.066, 172.502]

    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.plot(batch_sizes, time_us, marker="o", linewidth=2)
    for x, y in zip(batch_sizes, time_us):
        ax.annotate(f"{y:.1f}", (x, y), xytext=(0, 7),
                    textcoords="offset points", ha="center", fontsize=8)

    ax.set_title("Exact GPU Baseline: Batch Size vs Latency")
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Average GPU time / query (us)")
    ax.set_xticks(batch_sizes)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "batch_size_exact_latency.png")
    plt.close(fig)


def plot_method_comparison():
    methods = [
        "Exact",
        "Fused exact",
        "IVF grouped",
        "IVF-PQ\nP=2048",
        "IVF-PQ\nP=4096",
        "IVF-PQ\nP=8192",
    ]
    time_us = [195.955, 281.346, 44.3081, 12.0312, 17.6531, 27.1984]
    recall = [1.0, 1.0, 0.96285, 0.96035, 0.96265, 0.9628]

    fig, ax1 = plt.subplots(figsize=(8.0, 4.2))
    x = range(len(methods))
    bars = ax1.bar(x, time_us, color="#4c78a8", alpha=0.86)
    annotate_bars(ax1, bars, fmt="{:.1f}")
    ax1.set_ylabel("Average GPU time / query (us)")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(methods)
    ax1.set_title("Method Comparison: Latency and Recall")

    ax2 = ax1.twinx()
    ax2.plot(x, recall, color="#f58518", marker="o", linewidth=2)
    ax2.set_ylabel("Recall@10")
    ax2.set_ylim(0.90, 1.01)
    ax2.grid(False)
    for i, r in enumerate(recall):
        ax2.annotate(f"{r:.4f}", (i, r), xytext=(0, -14),
                     textcoords="offset points", ha="center",
                     color="#8a4b08", fontsize=8)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "method_latency_recall.png")
    plt.close(fig)


def plot_ivf_grouping_utilization():
    labels = ["Before grouping", "After grouping"]
    real_candidates = [13632.7, 13632.7]
    launched_slots = [23268.0, 17326.6]
    wasted_slots = [launched_slots[i] - real_candidates[i] for i in range(2)]
    utilization = [
        real_candidates[i] / launched_slots[i] for i in range(2)
    ]

    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    x = range(len(labels))
    b1 = ax.bar(x, real_candidates, label="Useful candidate slots",
                color="#54a24b", alpha=0.9)
    b2 = ax.bar(x, wasted_slots, bottom=real_candidates,
                label="Padding / wasted slots", color="#e45756", alpha=0.82)

    ax.set_title("IVF Query Grouping Reduces Padding Waste")
    ax.set_ylabel("Average candidate slots / query")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.legend(loc="upper right")

    for i, u in enumerate(utilization):
        ax.annotate(f"utilization={u * 100:.1f}%",
                    xy=(i, launched_slots[i]),
                    xytext=(0, 8),
                    textcoords="offset points",
                    ha="center",
                    fontsize=9)

    annotate_bars(ax, b1, fmt="{:.0f}", dy=-16)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "ivf_grouping_slot_utilization.png")
    plt.close(fig)


def plot_ivfpq_top_p_tradeoff():
    top_p = [2048, 4096, 8192]
    time_us = [12.0312, 17.6531, 27.1984]
    recall = [0.96035, 0.96265, 0.9628]

    fig, ax1 = plt.subplots(figsize=(6.6, 3.9))
    ax1.plot(top_p, time_us, color="#4c78a8", marker="o", linewidth=2)
    ax1.set_xlabel("IVF-PQ top-p refine candidates")
    ax1.set_ylabel("Average GPU time / query (us)", color="#4c78a8")
    ax1.tick_params(axis="y", labelcolor="#4c78a8")
    ax1.set_xticks(top_p)
    for x, y in zip(top_p, time_us):
        ax1.annotate(f"{y:.1f}", (x, y), xytext=(0, 7),
                     textcoords="offset points", ha="center",
                     color="#2f5f87", fontsize=8)

    ax2 = ax1.twinx()
    ax2.plot(top_p, recall, color="#f58518", marker="s", linewidth=2)
    ax2.set_ylabel("Recall@10", color="#f58518")
    ax2.tick_params(axis="y", labelcolor="#f58518")
    ax2.set_ylim(0.958, 0.964)
    ax2.grid(False)
    for x, y in zip(top_p, recall):
        ax2.annotate(f"{y:.5f}", (x, y), xytext=(0, -16),
                     textcoords="offset points", ha="center",
                     color="#9a520d", fontsize=8)

    ax1.set_title("IVF-PQ top-p Trade-off")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "ivfpq_top_p_tradeoff.png")
    plt.close(fig)


def plot_candidate_reduction():
    labels = ["IVF candidates", "IVF-PQ refine\nP=2048",
              "IVF-PQ refine\nP=4096", "IVF-PQ refine\nP=8192"]
    candidates = [13632.7, 2048, 4096, 8190.68]

    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    bars = ax.bar(labels, candidates, color=["#72b7b2", "#4c78a8",
                                             "#4c78a8", "#4c78a8"])
    annotate_bars(ax, bars, fmt="{:.0f}")
    ax.set_title("IVF-PQ Reduces Exact Re-ranking Work")
    ax.set_ylabel("Average exact/refine candidates / query")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "ivfpq_candidate_reduction.png")
    plt.close(fig)


def main():
    plot_batch_size()
    plot_method_comparison()
    plot_ivf_grouping_utilization()
    plot_ivfpq_top_p_tradeoff()
    plot_candidate_reduction()
    print(f"Saved figures to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
