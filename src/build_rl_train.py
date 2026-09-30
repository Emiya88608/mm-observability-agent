from pathlib import Path
from collections import Counter
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/rl_train"
IMAGE_DIR = OUT / "images"
rng = np.random.default_rng(20261003)

def answer_for(label):
    mapping = {
        "normal": {
            "incident_type": "normal",
            "affected_metric": "none",
            "recommended_action": "continue_monitoring",
        },
        "latency_spike": {
            "incident_type": "latency_spike",
            "affected_metric": "latency_ms",
            "recommended_action": "inspect_network_path",
        },
        "packet_loss_spike": {
            "incident_type": "packet_loss_spike",
            "affected_metric": "packet_loss_pct",
            "recommended_action": "inspect_packet_loss",
        },
    }
    return mapping[label]

def rule_label(tool):
    if tool["latency_max_ms"] > 80:
        return "latency_spike"
    if tool["packet_loss_max_pct"] > 3:
        return "packet_loss_spike"
    return "normal"

def render(sample_id, label, theme):
    x = np.arange(60)
    latency = 35 + rng.normal(0, 2.8, 60)
    loss = np.clip(0.35 + rng.normal(0, 0.09, 60), 0, None)
    point = int(rng.integers(18, 38))

    if label == "normal":
        latency[point:point + 7] = np.minimum(
            rng.uniform(68, 78.5) + rng.normal(0, 0.25, 7), 79.5
        )
        loss[point:point + 7] = np.minimum(
            rng.uniform(2.0, 2.8) + rng.normal(0, 0.03, 7), 2.95
        )
    elif label == "latency_spike":
        latency[point:point + 7] = rng.uniform(80.5, 94) + rng.normal(0, 0.7, 7)
    else:
        loss[point:point + 7] = rng.uniform(3.05, 4.3) + rng.normal(0, 0.04, 7)

    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    image_path = IMAGE_DIR / f"{sample_id}.png"

    dark = theme == "dark"
    fig, axes = plt.subplots(2, 1, figsize=(9.6, 5.6), sharex=True)
    if dark:
        fig.patch.set_facecolor("#111827")
        for ax in axes:
            ax.set_facecolor("#1F2937")
            ax.tick_params(colors="#D1D5DB")
            ax.grid(alpha=0.15, color="#E5E7EB")
            for spine in ax.spines.values():
                spine.set_color("#6B7280")
        text_color = "#F9FAFB"
        latency_color = "#38BDF8"
        loss_color = "#FBBF24"
    else:
        for ax in axes:
            ax.grid(alpha=0.25)
        text_color = "#111827"
        latency_color = "#2D6CDF"
        loss_color = "#E28413"

    axes[0].plot(x, latency, color=latency_color, linewidth=2.1)
    axes[0].axhline(80, color="#EF4444", linestyle="--", linewidth=1.2)
    axes[0].set_ylabel("Latency (ms)", color=text_color)

    axes[1].plot(x, loss, color=loss_color, linewidth=2.1)
    axes[1].axhline(3, color="#EF4444", linestyle="--", linewidth=1.2)
    axes[1].set_ylabel("Packet loss (%)", color=text_color)
    axes[1].set_xlabel("Minute", color=text_color)

    fig.suptitle("Network Health Dashboard", color=text_color)
    fig.tight_layout()
    fig.savefig(image_path, dpi=130, facecolor=fig.get_facecolor())
    plt.close(fig)

    return {
        "latency_max_ms": round(float(latency.max()), 2),
        "packet_loss_max_pct": round(float(loss.max()), 2),
    }

# 防止 ID 与任何既有训练/测试集合重叠。
existing_ids = set()
for path in [
    ROOT / "data/splits/train.jsonl",
    ROOT / "data/splits/validation.jsonl",
    ROOT / "data/splits/test.jsonl",
    ROOT / "data/ood_eval/test.jsonl",
    ROOT / "data/ood_eval_v2/test.jsonl",
    ROOT / "data/sft/train_trajectories.jsonl",
]:
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                existing_ids.add(json.loads(line)["id"])

labels = (
    ["normal"] * 24
    + ["latency_spike"] * 24
    + ["packet_loss_spike"] * 24
)

rows = []
tools = {}

for i, label in enumerate(labels):
    sample_id = f"rl_{i:03d}"
    assert sample_id not in existing_ids, f"ID 重叠：{sample_id}"

    theme = "dark" if i % 2 == 0 else "light"
    tool = render(sample_id, label, theme)
    assert rule_label(tool) == label, (sample_id, label, tool)

    tool_call = {
        "name": "query_metric_window",
        "arguments": {
            "window_minutes": 60,
            "metrics": ["latency_ms", "packet_loss_pct"],
        },
    }

    rows.append({
        "id": sample_id,
        "image": f"rl_train/images/{sample_id}.png",
        "question": "请分析该监控截图。仅输出 JSON：incident_type、affected_metric、recommended_action。",
        "answer": answer_for(label),
        "tool_call": tool_call,
        "tool_result": tool,
        "difficulty": "threshold_boundary",
        "theme": theme,
    })
    tools[sample_id] = tool

OUT.mkdir(parents=True, exist_ok=True)
(OUT / "train.jsonl").write_text(
    "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
    encoding="utf-8",
)
(OUT / "tool_metrics.json").write_text(
    json.dumps(tools, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

summary = {
    "total": len(rows),
    "label_distribution": dict(Counter(row["answer"]["incident_type"] for row in rows)),
    "theme_distribution": dict(Counter(row["theme"] for row in rows)),
    "test_leakage_check": False,
}
(OUT / "summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print("========== RL 训练数据构建完成 ==========")
print(json.dumps(summary, ensure_ascii=False, indent=2))
