import json
import re
from collections import Counter
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
TEST_FILE = ROOT / "data/splits/test.jsonl"
METRIC_FILE = ROOT / "data/tool_metrics.json"
OUTPUT_FILE = ROOT / "artifacts/sft_tool_grounded_100_predictions.jsonl"
METRICS_FILE = ROOT / "artifacts/sft_tool_grounded_100_metrics.json"

SYSTEM_PROMPT = """你是网络监控异常分析 Agent。
截图仅提供初步线索；精确数值以 query_metric_window 的工具结果为准。
判定规则：
- latency_max_ms > 80：latency_spike；
- 否则 packet_loss_max_pct > 3：packet_loss_spike；
- 否则：normal。
最终只能输出 incident_type、affected_metric、recommended_action 三个字段组成的 JSON。"""

REQUIRED_KEYS = [
    "incident_type",
    "affected_metric",
    "recommended_action",
]
CLASSES = ["normal", "latency_spike", "packet_loss_spike"]

def parse_json(text):
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
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

def messages_for(row, tool_result):
    tool_call = {
        "name": "query_metric_window",
        "arguments": {
            "window_minutes": 60,
            "metrics": ["latency_ms", "packet_loss_pct"],
        },
    }
    user_text = (
        f"{row['question']}\n\n"
        f"已调用工具 query_metric_window："
        f"{json.dumps(tool_call, ensure_ascii=False)}\n"
        f"工具返回的精确数值："
        f"{json.dumps(tool_result, ensure_ascii=False)}\n"
        "请依据工具数值给出最终结论。"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(ROOT / 'data' / row["image"])},
                {"type": "text", "text": user_text},
            ],
        },
    ]

if not ADAPTER_DIR.exists():
    raise FileNotFoundError(f"找不到 SFT adapter：{ADAPTER_DIR}")

rows = [
    json.loads(line)
    for line in TEST_FILE.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
metrics_by_id = json.loads(METRIC_FILE.read_text(encoding="utf-8"))

quant_config = BitsAndBytesConfig(
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

base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    quantization_config=quant_config,
    torch_dtype=torch.bfloat16,
    device_map={"": 0},
)
model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
model.eval()

predictions = []
json_valid = 0
incident_correct = 0
full_correct = 0
confusion = {label: Counter() for label in CLASSES}

with torch.inference_mode():
    for index, row in enumerate(rows, 1):
        tool_result = metrics_by_id[row["id"]]
        image = Image.open(ROOT / "data" / row["image"]).convert("RGB")
        messages = messages_for(row, tool_result)

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
        generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
        raw_output = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        prediction, valid = parse_json(raw_output)
        if valid:
            json_valid += 1

        truth = normalize(row["answer"])
        predicted = normalize(prediction)
        true_label = truth["incident_type"]
        pred_label = predicted["incident_type"] or "INVALID"

        confusion[true_label][pred_label] += 1
        incident_correct += int(pred_label == true_label)
        full_correct += int(predicted == truth)

        predictions.append({
            "id": row["id"],
            "answer": row["answer"],
            "tool_result": tool_result,
            "prediction": predicted,
            "raw_output": raw_output,
        })

        print(
            f"[{index}/{len(rows)}] {row['id']} | "
            f"true={true_label} | pred={pred_label}"
        )

with OUTPUT_FILE.open("w", encoding="utf-8") as f:
    for row in predictions:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

per_class = {}
for label in CLASSES:
    tp = confusion[label][label]
    support = sum(confusion[label].values())
    predicted_as_label = sum(confusion[true][label] for true in CLASSES)
    per_class[label] = {
        "support": support,
        "precision": round(tp / predicted_as_label, 4) if predicted_as_label else 0,
        "recall": round(tp / support, 4) if support else 0,
    }

summary = {
    "method": "qwen25vl3b_qlora_tool_grounded_sft",
    "adapter": str(ADAPTER_DIR),
    "total": len(rows),
    "json_valid_rate": round(json_valid / len(rows), 4),
    "incident_accuracy": round(incident_correct / len(rows), 4),
    "full_json_accuracy": round(full_correct / len(rows), 4),
    "tool_called_rate": 1.0,
    "per_class": per_class,
    "confusion_matrix": {
        true: {
            pred: confusion[true][pred]
            for pred in CLASSES + ["INVALID"]
        }
        for true in CLASSES
    },
    "prediction_file": str(OUTPUT_FILE),
}

METRICS_FILE.write_text(
    json.dumps(summary, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print("\n========== SFT 测试集指标 ==========")
print(json.dumps(summary, ensure_ascii=False, indent=2))
