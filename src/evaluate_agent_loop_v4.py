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
TEST_FILE = ROOT / "data/ood_eval_v4/test.jsonl"
TOOL_FILE = ROOT / "data/ood_eval_v4/tool_metrics.json"
ADAPTERS = {
    "sft": ROOT / "artifacts/qlora_runs/sft_tool_grounded_100/adapter",
    "grpo_10step": ROOT / "artifacts/grpo_runs/grpo_smoke_10/adapter",
}
OUT_DIR = ROOT / "artifacts/agent_loop_v4"
LABELS = ["normal", "latency_spike", "packet_loss_spike"]
KEYS = ["incident_type", "affected_metric", "recommended_action"]

SYSTEM = """你是网络监控诊断 Agent，可以使用 query_metric_window 工具。
先看监控图。如果需要精确数值才能可靠判断，就请求调用工具。
第一步必须只输出以下 JSON 之一：
调用工具：{"action":"call_tool","tool":"query_metric_window","arguments":{"window_minutes":60}}
直接回答：{"action":"final","answer":{"incident_type":"...","affected_metric":"...","recommended_action":"..."}}
不要输出其他文字。"""

FINAL_SYSTEM = """你是网络监控诊断 Agent。根据监控截图和工具精确数值判断。
规则：latency_max_ms > 80 判为 latency_spike；否则 packet_loss_max_pct > 3 判为 packet_loss_spike；否则判为 normal。
最终仅输出 JSON，字段为 incident_type、affected_metric、recommended_action。"""

def parse_json(text):
    text = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.I)
    text = re.sub(r"\s*```$", "", text)
    match = re.search(r"\{.*\}", text, flags=re.S)
    for candidate in [text, match.group(0) if match else ""]:
        try:
            value = json.loads(candidate)
            if isinstance(value, dict):
                return value
        except (json.JSONDecodeError, TypeError):
            pass
    return {}

def normalized(value):
    return {key: value.get(key) for key in KEYS}

def infer(model, processor, image, messages, max_tokens=128):
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
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            use_cache=True,
        )
    generated = output[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(
        generated,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

def run_one_model(name, adapter_path, rows, tool_db, processor):
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

    correct = valid_count = tool_calls = fallback_calls = 0
    confusion = {label: Counter() for label in LABELS}
    outputs = []

    for i, row in enumerate(rows, 1):
        image = Image.open(ROOT / "data" / row["image"]).convert("RGB")
        first_messages = [
            {"role": "system", "content": SYSTEM},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(ROOT / "data" / row["image"])},
                    {"type": "text", "text": row["question"]},
                ],
            },
        ]

        action_raw = infer(model, processor, image, first_messages)
        action = parse_json(action_raw)
        action_type = action.get("action")
        tool_result = None
        final_raw = None
        fallback = False

        if action_type == "final" and isinstance(action.get("answer"), dict):
            prediction = normalized(action["answer"])
            final_valid = True
        else:
            # 仅识别到明确 call_tool 时记作模型调用；格式错误时由程序安全回退调用。
            if action_type == "call_tool" and action.get("tool") == "query_metric_window":
                tool_calls += 1
            else:
                fallback_calls += 1
                fallback = True

            # 本地模拟工具：按样本 ID 返回预先生成的监控数值。
            tool_result = tool_db[row["id"]]
            tool_message = (
                f"工具 query_metric_window 返回："
                f"{json.dumps(tool_result, ensure_ascii=False)}。"
                "请按阈值规则给出最终诊断 JSON。"
            )
            final_messages = [
                {"role": "system", "content": FINAL_SYSTEM},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": str(ROOT / "data" / row["image"])},
                        {
                            "type": "text",
                            "text": f"{row['question']}\n\n{tool_message}",
                        },
                    ],
                },
            ]
            final_raw = infer(model, processor, image, final_messages)
            parsed_final = parse_json(final_raw)
            prediction = normalized(parsed_final)
            final_valid = all(prediction.get(key) is not None for key in KEYS)

        truth = normalized(row["answer"])
        label = prediction.get("incident_type") or "INVALID"
        valid_count += int(final_valid)
        correct += int(prediction == truth)
        confusion[truth["incident_type"]][label] += 1

        outputs.append({
            "id": row["id"],
            "answer": row["answer"],
            "action_raw": action_raw,
            "action": action,
            "fallback_tool_call": fallback,
            "tool_result": tool_result,
            "final_raw": final_raw,
            "prediction": prediction,
        })

        print(
            f"[{name} {i}/{len(rows)}] {row['id']} "
            f"true={truth['incident_type']} pred={label} "
            f"tool={'yes' if tool_result is not None else 'no'}"
        )

    result = {
        "method": name,
        "total": len(rows),
        "full_json_accuracy": round(correct / len(rows), 4),
        "final_json_valid_rate": round(valid_count / len(rows), 4),
        "model_tool_call_rate": round(tool_calls / len(rows), 4),
        "fallback_tool_call_rate": round(fallback_calls / len(rows), 4),
        "per_class_recall": {
            label: round(
                confusion[label][label] / sum(confusion[label].values()), 4
            )
            for label in LABELS
        },
        "confusion_matrix": {
            true: {
                pred: confusion[true][pred]
                for pred in LABELS + ["INVALID"]
            }
            for true in LABELS
        },
    }

    with (OUT_DIR / f"{name}_predictions.jsonl").open("w", encoding="utf-8") as f:
        for item in outputs:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    del model, base
    gc.collect()
    torch.cuda.empty_cache()
    return result

if not torch.cuda.is_available():
    raise RuntimeError("CUDA 不可用。")
rows = [
    json.loads(line)
    for line in TEST_FILE.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
tool_db = json.loads(TOOL_FILE.read_text(encoding="utf-8"))
assert len(rows) == 60, f"v4 应有 60 条样本，实际 {len(rows)}"
assert len({row["id"] for row in rows}) == 60
for name, path in ADAPTERS.items():
    if not path.exists():
        raise FileNotFoundError(f"{name} adapter 不存在：{path}")

OUT_DIR.mkdir(parents=True, exist_ok=True)
processor = AutoProcessor.from_pretrained(
    MODEL_NAME,
    min_pixels=256 * 28 * 28,
    max_pixels=512 * 28 * 28,
)

results = {}
for name, path in ADAPTERS.items():
    results[name] = run_one_model(name, path, rows, tool_db, processor)

summary = {
    "dataset": "ood_dark_dashboard_v4_60",
    "tool_backend": "local deterministic simulator",
    "results": results,
    "grpo_minus_sft_accuracy": round(
        results["grpo_10step"]["full_json_accuracy"]
        - results["sft"]["full_json_accuracy"],
        4,
    ),
}
summary_path = OUT_DIR / "agent_loop_summary.json"
summary_path.write_text(
    json.dumps(summary, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print("\n========== Agent 工具调用闭环评测 ==========")
print(json.dumps(summary, ensure_ascii=False, indent=2))
print("\n结果文件：", summary_path)
