from pathlib import Path
import json
import numpy as np

root = Path(__file__).resolve().parents[1]
rng = np.random.default_rng(20260929)
tool_data = {}

for i in range(150):
    incident = ["normal", "latency_spike", "packet_loss_spike"][i % 3]

    latency = 32 + rng.normal(0, 2.5, 60)
    loss = np.clip(0.25 + rng.normal(0, 0.08, 60), 0, None)
    start = int(rng.integers(18, 43))

    if incident == "latency_spike":
        latency[start:start + 7] += rng.uniform(70, 120)
    elif incident == "packet_loss_spike":
        loss[start:start + 7] += rng.uniform(5, 12)

    tool_data[f"monitor_{i:04d}"] = {
        "latency_max_ms": round(float(latency.max()), 2),
        "packet_loss_max_pct": round(float(loss.max()), 2),
    }

output = root / "data/tool_metrics.json"
output.write_text(
    json.dumps(tool_data, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
print("工具指标库已生成：", output)
print("样例：", tool_data["monitor_0000"])
