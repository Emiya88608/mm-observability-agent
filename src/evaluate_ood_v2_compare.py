import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import torch
from PIL import Image
from peft import PeftModel
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2_5_VLForConditionalGeneration,
)

ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"
ADAPTER_DIR = ROOT / "artifacts/qlora_runs/sft_tool_grounded_100/adapter"
TEST_FILE = ROOT / "data/ood_eval_v2/test.jsonl"
TOOL_FILE = ROOT / "data/ood_eval_v2/tool_metrics.json"
OUT_DIR = ROOT / "artifacts/ood_eval_v2"
CLASSES = ["normal", "latency_spike", "packet_loss_spike"]
KEYS = ["incident_type", "affected_metric", "recommended_action"]

SYSTEM = """你是网络监控异常分析 Agent。
截图仅提供初步线索；精确数值以 query_metric_window 的工具结果为准。
判定规则：
- latency_max_ms > 80：latency_spike；
- 否则 packet_loss_max_pct > 3：packet_loss_spike；
- 否则：normal。
最终只能输出 incident_type、affected_metric、recommended_action 三个字段组成的 JSON。"""

def parse_json(text):
    text = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.I)
    text = re.sub(r"\s*```$", "", text)
    match = re.search(r"\{.*\}", text, flags=re.S)
    for candidate in [text, match.group(0) if match else ""]:
        try:
            return json.loads(candidate), True
        except (json.JSONDecodeError, TypeError):
            pass
    return {}, False

def norm(data):
    return {key: data.get(key) for key in KEYS}

def make_messages(row, tool_result):
    tool_call = {
        "name": "query_metric_window",
        "arguments": {
            "window_minutes": 60,
            "metrics": ["latency_ms", "packet_loss_pct"],
        },
    }
    text = (
        f"{row['question']}\n\n"
        f"已调用工具 query_metric_window："
        f"{json.dumps(tool_call, ensure_ascii=False)}\n"
        f"工具返回的精确数值："
        f"{json.dumps(tool_result, ensure_ascii=False)}\n"
        "请依据工具数值给出最终结论。"
    )
    return [
        {"role": "system", "content": SYSTEM},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(ROOT / "data" / row["image"])},
                {"type": "text", "text": text},
            ],
        },
    ]

def evaluate(model, processor, rows, tools, method):
    predictions = []
    valid_count = incident_correct = full_correct = 0
    confusion = {label: Counter() for label in CLASSES}
    difficulty_stats = defaultdict(lambda: {"total": 0, "correct": 0})

    with torch.inference_mode():
        for i, row in enumerate(rows, 1):
            image = Image.open(ROOT / "data" / row["image"]).convert("RGB")
            messages = make_messages(row, tools[row["id"]])
            prompt = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = processor(
                text=[prompt],
                images=[image],
                return_tensors="pt",
            ).to("cuda")

            output_ids = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                use_cache=True,
            )
            output_ids = output_ids[:, inputs["input_ids"].shape[1]:]
            raw = processor.batch_decode(
                output_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]

            pred, valid = parse_json(raw)
            truth = norm(row["answer"])
            prediction = norm(pred)
            true_label = truth["incident_type"]
            pred_label = prediction["incident_type"] or "INVALID"

            valid_count += int(valid)
            incident_correct += int(pred_label == true_label)
            full_correct += int(prediction == truth)
            confusion[true_label][pred_label] += 1

            difficulty = row["difficulty"]
            difficulty_stats[difficulty]["total"] += 1
            difficulty_stats[difficulty]["correct"] += int(prediction == truth)

            predictions.append({
                "id": row["id"],
                "difficulty": difficulty,
                "answer": row["answer"],
                "tool_result": tools[row["id"]],
                "prediction": prediction,
                "raw_output": raw,
            })
            print(f"[{method} {i}/{len(rows)}] {row['id']} true={true_label} pred={pred_label}")

    per_class = {}
    for label in CLASSES:
        tp = confusion[label][label]
        support = sum(confusion[label].values())
        predicted = sum(confusion[t][label] for t in CLASSES)
        per_class[label] = {
            "support": support,
            "precision": round(tp / predicted, 4) if predicted else 0,
            "recall": round(tp / support, 4) if support else 0,
        }

    difficulty_metrics = {
        key: {
            "total": value["total"],
            "full_json_accuracy": round(value["correct"] / value["total"], 4),
        }
        for key, value in difficulty_stats.items()
    }

    result = {
        "method": method,
        "total": len(rows),
        "json_valid_rate": round(valid_count / len(rows), 4),
        "incident_accuracy": round(incident_correct / len(rows), 4),
        "full_json_accuracy": round(full_correct / len(rows), 4),
        "tool_called_rate": 1.0,
        "per_class": per_class,
        "difficulty_metrics": difficulty_metrics,
        "confusion_matrix": {
            true: {pred: confusion[true][pred] for pred in CLASSES + ["INVALID"]}
            for true in CLASSES
        },
    }
    return result, predictions

OUT_DIR.mkdir(parents=True, exist_ok=True)
rows = [json.loads(x) for x in TEST_FILE.read_text(encoding="utf-8").splitlines() if x.strip()]
tools = json.loads(TOOL_FILE.read_text(encoding="utf-8"))

assert len(rows) == 60, f"预期 60 条 OOD 样本，实际 {len(rows)}"
assert ADAPTER_DIR.exists(), f"找不到 adapter：{ADAPTER_DIR}"

quant = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
processor = AutoProcessor.from_pretrained(
    MODEL_NAME,
    min_pixels=256 * 28 * 28,
    max_pixels=512 * 28 * 28,
)

base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    quantization_config=quant,
    torch_dtype=torch.bfloat16,
    device_map={"": 0},
)
base.eval()

base_result, base_predictions = evaluate(
    base, processor, rows, tools, "base_tool_grounded_ood"
)

adapter_model = PeftModel.from_pretrained(base, ADAPTER_DIR)
adapter_model.eval()

sft_result, sft_predictions = evaluate(
    adapter_model, processor, rows, tools, "sft_tool_grounded_ood"
)

for name, predictions in {
    "base_tool_grounded_ood": base_predictions,
    "sft_tool_grounded_ood": sft_predictions,
}.items():
    with (OUT_DIR / f"{name}_predictions.jsonl").open("w", encoding="utf-8") as f:
        for row in predictions:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

summary = {
    "dataset": "ood_dark_dashboard_v2_60",
    "base": base_result,
    "sft": sft_result,
    "full_json_accuracy_delta": round(
        sft_result["full_json_accuracy"] - base_result["full_json_accuracy"], 4
    ),
    "incident_accuracy_delta": round(
        sft_result["incident_accuracy"] - base_result["incident_accuracy"], 4
    ),
}
(OUT_DIR / "ood_comparison_summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print("\n========== OOD 基座 vs SFT 对比 ==========")
print(json.dumps(summary, ensure_ascii=False, indent=2))
