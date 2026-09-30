from pathlib import Path
from collections import defaultdict
import json

ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS = ROOT / "artifacts/agent_loop_v4/sft_predictions.jsonl"
METRICS = ROOT / "data/ood_eval_v4/tool_metrics.json"
OUTPUT = ROOT / "artifacts/agent_loop_v4/rule_baseline_summary.json"

def classify(metrics):
    latency = float(metrics["latency_max_ms"])
    loss = float(metrics["packet_loss_max_pct"])

    if latency > 80:
        label = "latency_spike"
    elif loss > 3:
        label = "packet_loss_spike"
    else:
        label = "normal"

    result = {
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
        "normal": {
            "incident_type": "normal",
            "affected_metric": "none",
            "recommended_action": "continue_monitoring",
        },
    }
    return result[label]

rows = [
    json.loads(line)
    for line in PREDICTIONS.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
metrics_by_id = json.loads(METRICS.read_text(encoding="utf-8"))

assert all(row["id"] in metrics_by_id for row in rows), "有测试样本缺少工具指标"

matrix = defaultdict(lambda: defaultdict(int))
class_totals = defaultdict(int)
class_correct = defaultdict(int)
full_correct = 0

for row in rows:
    expected = row["answer"]
    predicted = classify(metrics_by_id[row["id"]])
    truth = expected["incident_type"]
    guess = predicted["incident_type"]

    matrix[truth][guess] += 1
    class_totals[truth] += 1
    class_correct[truth] += int(truth == guess)
    full_correct += int(all(predicted.get(key) == value for key, value in expected.items()))

total = len(rows)
summary = {
    "method": "forced_tool_query_then_deterministic_threshold_rule",
    "dataset": "ood_dark_dashboard_v4",
    "total": total,
    "incident_accuracy": round(sum(class_correct.values()) / total, 4),
    "full_json_accuracy": round(full_correct / total, 4),
    "per_class_recall": {
        label: round(class_correct[label] / count, 4)
        for label, count in class_totals.items()
    },
    "confusion_matrix": {label: dict(values) for label, values in matrix.items()},
    "validity_note": (
        "Diagnostic only: synthetic labels use the same metric thresholds as this rule; "
        "not an independent generalization result."
    ),
}

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

print("========== 规则工具基线已复现 ==========")
print(json.dumps(summary, ensure_ascii=False, indent=2))
print("\n证据文件：", OUTPUT)
