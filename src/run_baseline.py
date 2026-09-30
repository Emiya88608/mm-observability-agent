from pathlib import Path
import argparse
import json
import re

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

parser = argparse.ArgumentParser()
parser.add_argument("--limit", type=int, default=5)
args = parser.parse_args()

root = Path(__file__).resolve().parents[1]
test_path = root / "data/splits/test.jsonl"
output_path = root / f"artifacts/baseline_qwen25vl3b_{args.limit}_predictions.jsonl"
metrics_path = root / f"artifacts/baseline_qwen25vl3b_{args.limit}_metrics.json"

records = [json.loads(line) for line in test_path.read_text(encoding="utf-8").splitlines()]
records = records[:args.limit]

model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
print("加载模型：", model_name)

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(model_name)

def parse_json(text):
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None

rows = []
valid_json = 0
correct_incident = 0
fully_correct = 0

for index, record in enumerate(records, start=1):
    image_path = root / "data" / record["image"]
    prompt = """
你是网络监控分析助手。请根据截图判断主要异常。
只能输出一行 JSON，不要 Markdown，不要解释。
JSON 必须严格包含：
incident_type：normal / latency_spike / packet_loss_spike
affected_metric：none / latency_ms / packet_loss_pct
recommended_action：continue_monitoring / inspect_network_path / inspect_packet_loss
"""

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": str(image_path)},
            {"type": "text", "text": prompt},
        ],
    }]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=100)

    trimmed_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
    ]
    raw_output = processor.batch_decode(
        trimmed_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    prediction = parse_json(raw_output)
    answer = record["answer"]

    if prediction is not None:
        valid_json += 1
        if prediction.get("incident_type") == answer["incident_type"]:
            correct_incident += 1
        if prediction == answer:
            fully_correct += 1

    row = {
        "id": record["id"],
        "answer": answer,
        "prediction": prediction,
        "raw_output": raw_output,
    }
    rows.append(row)
    print(f"[{index}/{len(records)}] {record['id']} -> {raw_output}")

with output_path.open("w", encoding="utf-8") as f:
    for row in rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

total = len(records)
metrics = {
    "total": total,
    "json_valid_rate": round(valid_json / total, 4),
    "incident_accuracy": round(correct_incident / total, 4),
    "full_json_accuracy": round(fully_correct / total, 4),
    "prediction_file": str(output_path),
}
metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

print("\n========== 实验 1 指标 ==========")
print(json.dumps(metrics, ensure_ascii=False, indent=2))
