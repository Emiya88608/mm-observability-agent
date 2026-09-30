from pathlib import Path
import argparse
import json
import re

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

parser = argparse.ArgumentParser()
parser.add_argument("--limit", type=int, default=25)
args = parser.parse_args()

root = Path(__file__).resolve().parents[1]
records = [
    json.loads(line)
    for line in (root / "data/splits/test.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
][:args.limit]

tool_metrics = json.loads(
    (root / "data/tool_metrics.json").read_text(encoding="utf-8")
)

def parse_json(text):
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None

model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
print("加载模型：", model_name)

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(model_name)

rows = []
correct = 0
valid_json = 0

for number, record in enumerate(records, start=1):
    image_path = root / "data" / record["image"]

    # 模拟 Agent 调用了 query_metric_window 工具
    tool_result = tool_metrics[record["id"]]

    prompt = f"""
你是网络监控分析 Agent。你已经调用了 query_metric_window 工具。

工具返回：
- latency_max_ms = {tool_result["latency_max_ms"]}
- packet_loss_max_pct = {tool_result["packet_loss_max_pct"]}

严格决策规则：
1. latency_max_ms > 80：incident_type 为 latency_spike，
   affected_metric 为 latency_ms，
   recommended_action 为 inspect_network_path。
2. 否则 packet_loss_max_pct > 3：incident_type 为 packet_loss_spike，
   affected_metric 为 packet_loss_pct，
   recommended_action 为 inspect_packet_loss。
3. 否则：incident_type 为 normal，
   affected_metric 为 none，
   recommended_action 为 continue_monitoring。

只能输出一行 JSON；不要 Markdown、不要解释。
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
        output_ids = model.generate(**inputs, max_new_tokens=100)

    trimmed = [
        output[len(input_ids):]
        for input_ids, output in zip(inputs.input_ids, output_ids)
    ]
    raw_output = processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    prediction = parse_json(raw_output)
    answer = record["answer"]

    if prediction is not None:
        valid_json += 1
        if prediction == answer:
            correct += 1

    rows.append({
        "id": record["id"],
        "answer": answer,
        "tool_result": tool_result,
        "prediction": prediction,
        "raw_output": raw_output,
    })

    print(f"[{number}/{len(records)}] {record['id']} -> {raw_output}")

output_path = root / f"artifacts/tool_augmented_{args.limit}_predictions.jsonl"
with output_path.open("w", encoding="utf-8") as f:
    for row in rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

metrics = {
    "method": "vlm_with_query_metric_window",
    "total": len(records),
    "tool_called_rate": 1.0,
    "json_valid_rate": round(valid_json / len(records), 4),
    "full_json_accuracy": round(correct / len(records), 4),
    "prediction_file": str(output_path),
}

metrics_path = root / f"artifacts/tool_augmented_{args.limit}_metrics.json"
metrics_path.write_text(
    json.dumps(metrics, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print("\n========== 工具增强实验指标 ==========")
print(json.dumps(metrics, ensure_ascii=False, indent=2))
