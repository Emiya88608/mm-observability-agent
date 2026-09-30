import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from peft import PeftModel
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data/rl_train/train.jsonl"
ADAPTER = ROOT / "artifacts/qlora_runs/sft_tool_grounded_100/adapter"
OUT_DIR = ROOT / "artifacts/rl_viability"
MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"
KEYS = ["incident_type", "affected_metric", "recommended_action"]

SYSTEM = """你是网络监控异常分析 Agent。
截图仅提供初步线索；精确数值以 query_metric_window 的工具结果为准。
判定规则：
- latency_max_ms > 80：latency_spike；
- 否则 packet_loss_max_pct > 3：packet_loss_spike；
- 否则：normal。
最终只能输出 incident_type、affected_metric、recommended_action 三个字段组成的 JSON。"""

def parse_json(text):
    clean = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.I)
    clean = re.sub(r"\s*```$", "", clean)
    match = re.search(r"\{.*\}", clean, flags=re.S)
    for candidate in [clean, match.group(0) if match else ""]:
        try:
            return json.loads(candidate), True
        except (json.JSONDecodeError, TypeError):
            pass
    return {}, False

def normalized(data):
    return {key: data.get(key) for key in KEYS}

def reward(prediction, valid, answer):
    # 完全基于确定性标签和工具规则，不使用 LLM-as-a-judge。
    score = 0.0
    if valid:
        score += 0.10
    if prediction.get("incident_type") == answer["incident_type"]:
        score += 0.50
    if prediction.get("affected_metric") == answer["affected_metric"]:
        score += 0.20
    if prediction.get("recommended_action") == answer["recommended_action"]:
        score += 0.20
    return round(score, 2)

def make_messages(row):
    tool_call = json.dumps(row["tool_call"], ensure_ascii=False)
    tool_result = json.dumps(row["tool_result"], ensure_ascii=False)
    text = (
        f"{row['question']}\n\n"
        f"已调用工具 query_metric_window：{tool_call}\n"
        f"工具返回的精确数值：{tool_result}\n"
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

parser = argparse.ArgumentParser()
parser.add_argument("--per_class", type=int, default=4)
parser.add_argument("--candidates", type=int, default=4)
args = parser.parse_args()

rows = [
    json.loads(line)
    for line in DATA_FILE.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
by_label = defaultdict(list)
for row in rows:
    by_label[row["answer"]["incident_type"]].append(row)

selected = []
for label in ["normal", "latency_spike", "packet_loss_spike"]:
    selected.extend(by_label[label][:args.per_class])

assert len(selected) == args.per_class * 3
assert ADAPTER.exists(), f"找不到 adapter：{ADAPTER}"

torch.manual_seed(20261003)
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
model = PeftModel.from_pretrained(base, ADAPTER)
model.eval()

records = []
groups = []

with torch.inference_mode():
    for group_index, row in enumerate(selected):
        image = Image.open(ROOT / "data" / row["image"]).convert("RGB")
        prompt = processor.apply_chat_template(
            make_messages(row),
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = processor(
            text=[prompt],
            images=[image],
            return_tensors="pt",
        ).to("cuda")

        rewards = []
        for candidate_index in range(args.candidates):
            output = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=True,
                temperature=0.8,
                top_p=0.95,
                use_cache=True,
            )
            generated = output[:, inputs["input_ids"].shape[1]:]
            raw = processor.batch_decode(
                generated,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
            prediction, valid = parse_json(raw)
            value = reward(prediction, valid, row["answer"])
            rewards.append(value)

            records.append({
                "group_id": group_index,
                "id": row["id"],
                "true_answer": row["answer"],
                "tool_result": row["tool_result"],
                "candidate_index": candidate_index,
                "prediction": normalized(prediction),
                "json_valid": valid,
                "reward": value,
                "raw_output": raw,
            })

        groups.append({
            "group_id": group_index,
            "id": row["id"],
            "label": row["answer"]["incident_type"],
            "rewards": rewards,
            "has_full_correct": 1.0 in rewards,
            "has_reward_variance": len(set(rewards)) > 1,
        })
        print(f"[{group_index + 1}/{len(selected)}] {row['id']} rewards={rewards}")

OUT_DIR.mkdir(parents=True, exist_ok=True)
with (OUT_DIR / "candidate_rollouts.jsonl").open("w", encoding="utf-8") as f:
    for record in records:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

summary = {
    "groups": len(groups),
    "candidates_per_group": args.candidates,
    "groups_with_reward_variance": sum(x["has_reward_variance"] for x in groups),
    "groups_with_full_correct_candidate": sum(x["has_full_correct"] for x in groups),
    "groups_with_correct_and_wrong_candidates": sum(
        x["has_full_correct"] and x["has_reward_variance"] for x in groups
    ),
    "mean_candidate_reward": round(
        sum(x["reward"] for x in records) / len(records), 4
    ),
    "grpo_viable": (
        sum(x["has_full_correct"] and x["has_reward_variance"] for x in groups) >= 3
    ),
    "groups_detail": groups,
}
(OUT_DIR / "viability_summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print("\n========== GRPO 奖励可行性结果 ==========")
print(json.dumps(summary, ensure_ascii=False, indent=2))
