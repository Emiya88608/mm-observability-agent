from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/ood_eval_v4"
IMAGE_DIR = OUT / "images"
rng = np.random.default_rng(20261005)

def answer_for(label):
    if label == "latency_spike":
        return {
            "incident_type": label,
            "affected_metric": "latency_ms",
            "recommended_action": "inspect_network_path",
        }
    if label == "packet_loss_spike":
        return {
            "incident_type": label,
            "affected_metric": "packet_loss_pct",
            "recommended_action": "inspect_packet_loss",
        }
    return {
        "incident_type": "normal",
        "affected_metric": "none",
        "recommended_action": "continue_monitoring",
    }

def render(sample_id, label, level):
    x = np.arange(60)
    latency = 38 + rng.normal(0, 3.5, 60)
    loss = np.clip(0.45 + rng.normal(0, 0.12, 60), 0, None)
    point = int(rng.integers(18, 38))

    if label == "latency_spike":
        target = rng.uniform(81, 95) if level == "boundary" else rng.uniform(100, 140)
        latency[point:point + 7] = target + rng.normal(0, 2.0, 7)

    elif label == "packet_loss_spike":
        target = rng.uniform(3.05, 3.8) if level == "boundary" else rng.uniform(4.5, 12)
        loss[point:point + 7] = target + rng.normal(0, 0.08, 7)

    else:
        if level == "boundary":
            # 近阈值但未越界。
            # 加噪声后仍严格低于告警阈值，避免标签与工具规则冲突。
            latency[point:point + 7] = np.minimum(
                rng.uniform(68, 78.5) + rng.normal(0, 0.3, 7),
                79.5,
            )
            loss[point:point + 7] = np.minimum(
                rng.uniform(2.0, 2.8) + rng.normal(0, 0.03, 7),
                2.95,
            )
        else:
            # 真正常规正常样本，远离两个告警阈值。
            latency = 32 + rng.normal(0, 2.5, 60)
            loss = np.clip(0.25 + rng.normal(0, 0.08, 60), 0, None)

    fig, axes = plt.subplots(2, 1, figsize=(10, 5.8), sharex=True)
    fig.patch.set_facecolor("#111827")

    for ax in axes:
        ax.set_facecolor("#1F2937")
        ax.grid(alpha=0.15, color="#E5E7EB")
        ax.tick_params(colors="#D1D5DB")
        for spine in ax.spines.values():
            spine.set_color("#6B7280")

    axes[0].plot(x, latency, color="#38BDF8", linewidth=2.2)
    axes[0].axhline(80, color="#FB7185", linestyle="--", linewidth=1.3)
    axes[0].set_ylabel("Latency (ms)", color="#D1D5DB")

    axes[1].plot(x, loss, color="#FBBF24", linewidth=2.2)
    axes[1].axhline(3, color="#FB7185", linestyle="--", linewidth=1.3)
    axes[1].set_ylabel("Packet loss (%)", color="#D1D5DB")
    axes[1].set_xlabel("Minute", color="#D1D5DB")

    fig.suptitle("Production Network Health Dashboard", color="#F9FAFB")
    fig.tight_layout()

    image_path = IMAGE_DIR / f"{sample_id}.png"
    fig.savefig(image_path, dpi=130, facecolor=fig.get_facecolor())
    plt.close(fig)

    return {
        "latency_max_ms": round(float(latency.max()), 2),
        "packet_loss_max_pct": round(float(loss.max()), 2),
    }

IMAGE_DIR.mkdir(parents=True, exist_ok=True)
rows = []
tool_metrics = {}

# 60 条：每类 20 条；其中每类 10 条是阈值边界样本。
plan = (
    [("normal", "boundary")] * 10
    + [("normal", "regular")] * 10
    + [("latency_spike", "boundary")] * 10
    + [("latency_spike", "regular")] * 10
    + [("packet_loss_spike", "boundary")] * 10
    + [("packet_loss_spike", "regular")] * 10
)

for index, (label, level) in enumerate(plan):
    sample_id = f"oodv4_{index:03d}"
    metrics = render(sample_id, label, level)

    # 工具数值与标签必须一致。
    by_rule = (
        "latency_spike" if metrics["latency_max_ms"] > 80
        else "packet_loss_spike" if metrics["packet_loss_max_pct"] > 3
        else "normal"
    )
    assert by_rule == label, (sample_id, label, metrics)

    rows.append({
        "id": sample_id,
        "image": f"ood_eval_v4/images/{sample_id}.png",
        "question": "请分析该监控截图。仅输出 JSON：incident_type、affected_metric、recommended_action。",
        "answer": answer_for(label),
        "distribution": "ood_dark_dashboard_v4",
        "difficulty": level,
    })
    tool_metrics[sample_id] = metrics

(OUT / "test.jsonl").write_text(
    "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
    encoding="utf-8",
)
(OUT / "tool_metrics.json").write_text(
    json.dumps(tool_metrics, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print("========== OOD 测试集构建完成 ==========")
print("样本数：", len(rows))
print("测试集：", OUT / "test.jsonl")
print("工具真值：", OUT / "tool_metrics.json")
print("图片目录：", IMAGE_DIR)
