from pathlib import Path
import json
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parents[1]
image_dir = root / "data/images"
split_dir = root / "data/splits"
rng = np.random.default_rng(20260929)
records = []

for i in range(150):
    incident = ["normal", "latency_spike", "packet_loss_spike"][i % 3]
    t = np.arange(60)
    latency = 32 + rng.normal(0, 2.5, 60)
    loss = np.clip(0.25 + rng.normal(0, 0.08, 60), 0, None)
    start = int(rng.integers(18, 43))

    if incident == "latency_spike":
        latency[start:start + 7] += rng.uniform(70, 120)
        answer = {"incident_type": incident, "affected_metric": "latency_ms",
                  "recommended_action": "inspect_network_path"}
    elif incident == "packet_loss_spike":
        loss[start:start + 7] += rng.uniform(5, 12)
        answer = {"incident_type": incident, "affected_metric": "packet_loss_pct",
                  "recommended_action": "inspect_packet_loss"}
    else:
        answer = {"incident_type": incident, "affected_metric": "none",
                  "recommended_action": "continue_monitoring"}

    image = image_dir / f"case_{i:04d}.png"
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(t, latency, color="#2563eb")
    axes[0].axhline(80, color="#dc2626", linestyle="--")
    axes[0].set_ylabel("Latency (ms)")
    axes[0].grid(alpha=0.25)
    axes[1].plot(t, loss, color="#ea580c")
    axes[1].axhline(3, color="#dc2626", linestyle="--")
    axes[1].set_ylabel("Packet loss (%)")
    axes[1].set_xlabel("Minute")
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(image, dpi=140)
    plt.close(fig)

    records.append({
        "id": f"monitor_{i:04d}",
        "image": f"images/{image.name}",
        "question": "请分析该监控截图。仅输出 JSON：incident_type、affected_metric、recommended_action。",
        "answer": answer,
        "expected_tool_path": ["crop_chart", "query_metric_window", "lookup_runbook"],
    })

rng.shuffle(records)
splits = {"train": records[:105], "validation": records[105:125], "test": records[125:]}

for name, rows in splits.items():
    with (split_dir / f"{name}.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

print("完成：", root)
print("样本分布：", Counter(x["answer"]["incident_type"] for x in records))
print("数据划分：", {k: len(v) for k, v in splits.items()})
