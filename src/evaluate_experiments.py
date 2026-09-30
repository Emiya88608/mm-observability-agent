from pathlib import Path
from collections import Counter, defaultdict
import json
import re

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
CLASSES = ["normal", "latency_spike", "packet_loss_spike"]
REQUIRED_KEYS = ["incident_type", "affected_metric", "recommended_action"]

FILES = {
    "baseline_vlm": ARTIFACTS / "baseline_qwen25vl3b_25_predictions.jsonl",
    "tool_augmented_vlm": ARTIFACTS / "tool_augmented_25_predictions.jsonl",
}

def read_jsonl(path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows

def parse_json(value):
    if isinstance(value, dict):
        return value, True
    if not isinstance(value, str):
        return {}, False

    text = re.sub(r"^```(?:json)?\s*", "", value.strip(), flags=re.I)
    text = re.sub(r"\s*```$", "", text)
    candidates = [text]

    match = re.search(r"\{.*\}", text, flags=re.S)
    if match:
        candidates.append(match.group(0))

    for candidate in candidates:
        try:
            return json.loads(candidate), True
        except json.JSONDecodeError:
            continue
    return {}, False

def normalize(data):
    return {key: data.get(key) for key in REQUIRED_KEYS}

def evaluate(name, path):
    if not path.exists():
        raise FileNotFoundError(f"找不到预测文件：{path}")

    rows = read_jsonl(path)
    ids = [row["id"] for row in rows]
    assert len(ids) == len(set(ids)), f"{name} 存在重复样本 ID"

    json_valid = 0
    incident_correct = 0
    full_correct = 0
    confusion = {true: Counter() for true in CLASSES}
    bad_cases = []
    tool_called = 0

    for row in rows:
        truth = normalize(row["answer"])
        pred, valid = parse_json(row.get("prediction", row.get("raw_output", "")))

        if valid:
            json_valid += 1

        pred_label = pred.get("incident_type", "INVALID")
        true_label = truth["incident_type"]

        if "tool_result" in row:
            tool_called += 1

        if true_label in CLASSES:
            confusion[true_label][pred_label] += 1

        if pred_label == true_label:
            incident_correct += 1

        if normalize(pred) == truth:
            full_correct += 1
        else:
            bad_cases.append({
                "id": row["id"],
                "true": truth,
                "prediction": normalize(pred),
                "tool_result": row.get("tool_result"),
            })

    total = len(rows)
    per_class = {}
    for label in CLASSES:
        tp = confusion[label][label]
        actual = sum(confusion[label].values())
        predicted = sum(confusion[t][label] for t in CLASSES)
        recall = tp / actual if actual else 0
        precision = tp / predicted if predicted else 0
        per_class[label] = {
            "support": actual,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
        }

    return {
        "method": name,
        "file": str(path),
        "total": total,
        "ids": ids,
        "json_valid_rate": round(json_valid / total, 4),
        "incident_accuracy": round(incident_correct / total, 4),
        "full_json_accuracy": round(full_correct / total, 4),
        "tool_called_rate": round(tool_called / total, 4),
        "prediction_distribution": dict(
            Counter(
                item
                for true_label in CLASSES
                for item, count in confusion[true_label].items()
                for _ in range(count)
            )
        ),
        "per_class": per_class,
        "confusion_matrix": {
            true: {pred: confusion[true][pred] for pred in CLASSES + ["INVALID"]}
            for true in CLASSES
        },
        "bad_cases": bad_cases,
    }

results = {}
for name, path in FILES.items():
    results[name] = evaluate(name, path)

baseline_ids = results["baseline_vlm"]["ids"]
tool_ids = results["tool_augmented_vlm"]["ids"]
assert set(baseline_ids) == set(tool_ids), "两组实验测试样本不一致，禁止直接比较。"

for item in results.values():
    item.pop("ids")

results["comparison"] = {
    "same_test_set": True,
    "full_json_accuracy_delta": round(
        results["tool_augmented_vlm"]["full_json_accuracy"]
        - results["baseline_vlm"]["full_json_accuracy"], 4
    ),
    "incident_accuracy_delta": round(
        results["tool_augmented_vlm"]["incident_accuracy"]
        - results["baseline_vlm"]["incident_accuracy"], 4
    ),
}

output = ARTIFACTS / "evaluation_summary.json"
output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

print("========== 正式评测结果 ==========")
for name, item in results.items():
    if name == "comparison":
        continue
    print(f"\n--- {name} ---")
    print("样本数：", item["total"])
    print("JSON 合法率：", item["json_valid_rate"])
    print("类别准确率：", item["incident_accuracy"])
    print("完整 JSON 准确率：", item["full_json_accuracy"])
    print("工具调用率：", item["tool_called_rate"])
    print("每类指标：", json.dumps(item["per_class"], ensure_ascii=False))
    print("混淆矩阵：", json.dumps(item["confusion_matrix"], ensure_ascii=False))
    print("Bad Case 数：", len(item["bad_cases"]))

print("\n========== 工具增强收益 ==========")
print(json.dumps(results["comparison"], ensure_ascii=False, indent=2))
print(f"\n证据文件已保存：{output}")
