import gc
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
TEST_FILE = ROOT / "data/ood_eval_v3/test.jsonl"
TOOL_FILE = ROOT / "data/ood_eval_v3/tool_metrics.json"
SFT_ADAPTER = ROOT / "artifacts/qlora_runs/sft_tool_grounded_100/adapter"
GRPO_ADAPTER = ROOT / "artifacts/grpo_runs/grpo_smoke_10/adapter"
OUT_DIR = ROOT / "artifacts/ood_final_v3"
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
            value = json.loads(candidate)
            if isinstance(value, dict):
                return value, True
        except (json.JSONDecodeError, TypeError):
            pass
    return {}, False

def normalize(data):
    return {key: data.get(key) for key in KEYS}

def make_messages(row, tool):
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
        f"工具返回的精确数值：{json.dumps(tool, ensure_ascii=False)}\n"
        "请依据工具数值给出最终结论。"
    )
    return [
        {"role": "system", "content": SYSTEM},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(ROOT / "data" / row["image"])},
                {"type": "text", "text": user_text},
            ],
        },
    ]

def evaluate(adapter_path, name, rows, tools, processor):
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        quantization_config=quant,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
    )
    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()

    valid_count = incident_correct = full_correct = 0
    confusion = {label: Counter() for label in CLASSES}
    predictions = []

    with torch.inference_mode():
        for index, row in enumerate(rows, 1):
            image = Image.open(ROOT / "data" / row["image"]).convert("RGB")
            prompt = processor.apply_chat_template(
                make_messages(row, tools[row["id"]]),
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = processor(
                text=[prompt],
                images=[image],
                return_tensors="pt",
            ).to("cuda")

            output = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                use_cache=True,
            )
            generated = output[:, inputs["input_ids"].shape[1]:]
            raw = processor.batch_decode(
                generated,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]

            prediction, valid = parse_json(raw)
            truth = normalize(row["answer"])
            pred = normalize(prediction)
            true_label = truth["incident_type"]
            pred_label = pred["incident_type"] or "INVALID"

            valid_count += int(valid)
            incident_correct += int(pred_label == true_label)
            full_correct += int(pred == truth)
            confusion[true_label][pred_label] += 1

            predictions.append({
                "id": row["id"],
                "difficulty": row["difficulty"],
                "answer": row["answer"],
                "tool_result": tools[row["id"]],
                "prediction": pred,
                "raw_output": raw,
            })
            print(f"[{name} {index}/{len(rows)}] {row['id']} true={true_label} pred={pred_label}")

    per_class = {}
    for label in CLASSES:
        tp = confusion[label][label]
        support = sum(confusion[label].values())
        predicted = sum(confusion[true][label] for true in CLASSES)
        per_class[label] = {
            "support": support,
            "precision": round(tp / predicted, 4) if predicted else 0,
            "recall": round(tp / support, 4) if support else 0,
        }

    result = {
        "method": name,
        "total": len(rows),
        "json_valid_rate": round(valid_count / len(rows), 4),
        "incident_accuracy": round(incident_correct / len(rows), 4),
        "full_json_accuracy": round(full_correct / len(rows), 4),
        "per_class": per_class,
        "confusion_matrix": {
            true: {
                pred: confusion[true][pred]
                for pred in CLASSES + ["INVALID"]
            }
            for true in CLASSES
        },
    }

    output_path = OUT_DIR / f"{name}_predictions.jsonl"
    with output_path.open("w", encoding="utf-8") as f:
        for row in predictions:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    del model, base
    gc.collect()
    torch.cuda.empty_cache()
    return result

if not TEST_FILE.exists() or not TOOL_FILE.exists():
    raise FileNotFoundError("OOD v3 测试文件缺失。")
if not SFT_ADAPTER.exists() or not GRPO_ADAPTER.exists():
    raise FileNotFoundError("SFT 或 GRPO adapter 不存在。")

rows = [
    json.loads(line)
    for line in TEST_FILE.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
tools = json.loads(TOOL_FILE.read_text(encoding="utf-8"))
assert len(rows) == 60, f"预期 60 条 v3 样本，实际 {len(rows)}"
assert len({row["id"] for row in rows}) == 60, "v3 样本 ID 重复"

OUT_DIR.mkdir(parents=True, exist_ok=True)
processor = AutoProcessor.from_pretrained(
    MODEL_NAME,
    min_pixels=256 * 28 * 28,
    max_pixels=512 * 28 * 28,
)

sft = evaluate(SFT_ADAPTER, "sft", rows, tools, processor)
grpo = evaluate(GRPO_ADAPTER, "grpo_smoke_10", rows, tools, processor)

summary = {
    "dataset": "ood_dark_dashboard_v3_60",
    "sft": sft,
    "grpo_smoke_10": grpo,
    "grpo_minus_sft_incident_accuracy": round(
        grpo["incident_accuracy"] - sft["incident_accuracy"], 4
    ),
    "grpo_minus_sft_full_json_accuracy": round(
        grpo["full_json_accuracy"] - sft["full_json_accuracy"], 4
    ),
}
summary_path = OUT_DIR / "sft_vs_grpo_summary.json"
summary_path.write_text(
    json.dumps(summary, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print("\n========== OOD v3：SFT vs GRPO ==========")
print(json.dumps(summary, ensure_ascii=False, indent=2))
print("\n结果文件：", summary_path)
