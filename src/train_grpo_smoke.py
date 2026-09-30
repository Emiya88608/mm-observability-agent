import json
import re
from pathlib import Path

import torch
from datasets import Dataset, Features, Image, Value
from peft import PeftModel, prepare_model_for_kbit_training
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2_5_VLForConditionalGeneration,
)
from trl import GRPOConfig, GRPOTrainer

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data/rl_train/grpo_informative.jsonl"
BASE_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
SFT_ADAPTER = ROOT / "artifacts/qlora_runs/sft_tool_grounded_100/adapter"
OUTPUT_DIR = ROOT / "artifacts/grpo_runs/grpo_smoke_10"

SYSTEM = """你是网络监控异常分析 Agent。
截图仅提供初步线索；精确数值以 query_metric_window 的工具结果为准。
判定规则：
- latency_max_ms > 80：latency_spike；
- 否则 packet_loss_max_pct > 3：packet_loss_spike；
- 否则：normal。
最终只能输出 incident_type、affected_metric、recommended_action 三个字段组成的 JSON。"""

KEYS = ["incident_type", "affected_metric", "recommended_action"]

def extract_text(completion):
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        parts = []
        for item in completion:
            if isinstance(item, dict):
                parts.append(str(item.get("content", "")))
        return "\n".join(parts)
    if isinstance(completion, dict):
        return str(completion.get("content", ""))
    return str(completion)

def parse_json(text):
    text = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.I)
    text = re.sub(r"\s*```$", "", text)
    match = re.search(r"\{.*\}", text, flags=re.S)
    for candidate in [text, match.group(0) if match else ""]:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
    return {}

def exact_reward(completions, answer_json, **kwargs):
    rewards = []
    for completion, expected in zip(completions, answer_json):
        predicted = parse_json(extract_text(completion))
        try:
            truth = json.loads(expected)
        except (TypeError, json.JSONDecodeError):
            truth = {}

        # 部分奖励让 GRPO 能区分字段级进步；满分要求三个字段完全正确。
        score = 0.0
        if predicted:
            score += 0.10
            score += 0.50 * (
                predicted.get("incident_type") == truth.get("incident_type")
            )
            score += 0.20 * (
                predicted.get("affected_metric") == truth.get("affected_metric")
            )
            score += 0.20 * (
                predicted.get("recommended_action") == truth.get("recommended_action")
            )
        rewards.append(float(score))
    return rewards

if not torch.cuda.is_available():
    raise RuntimeError("CUDA 不可用，停止。")
if not DATA_FILE.exists():
    raise FileNotFoundError(DATA_FILE)
if not SFT_ADAPTER.exists():
    raise FileNotFoundError(SFT_ADAPTER)

raw_rows = [
    json.loads(line)
    for line in DATA_FILE.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
if len(raw_rows) != 11:
    raise RuntimeError(f"预期 11 条有效组样本，实际 {len(raw_rows)}")

data = []
for row in raw_rows:
    tool_call = json.dumps(row["tool_call"], ensure_ascii=False)
    tool_result = json.dumps(row["tool_result"], ensure_ascii=False)
    user_content = (
        f"{row['question']}\n\n"
        f"已调用工具 query_metric_window：{tool_call}\n"
        f"工具返回的精确数值：{tool_result}\n"
        "请依据工具数值给出最终结论。"
    )
    data.append({
        "prompt": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user_content},
        ],
        "image": str(ROOT / "data" / row["image"]),
        "answer_json": json.dumps(row["answer"], ensure_ascii=False),
    })

features = Features({
    "prompt": [{"role": Value("string"), "content": Value("string")}],
    "image": Image(),
    "answer_json": Value("string"),
})
dataset = Dataset.from_list(data, features=features)

processor = AutoProcessor.from_pretrained(
    BASE_MODEL,
    min_pixels=256 * 28 * 28,
    max_pixels=512 * 28 * 28,
)
processor.tokenizer.padding_side = "left"

quantization = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    BASE_MODEL,
    quantization_config=quantization,
    torch_dtype=torch.bfloat16,
    device_map={"": 0},
)
base.config.use_cache = False
base = prepare_model_for_kbit_training(
    base,
    use_gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
)
model = PeftModel.from_pretrained(
    base,
    SFT_ADAPTER,
    is_trainable=True,
)

config = GRPOConfig(
    output_dir=str(OUTPUT_DIR),
    max_steps=10,
    learning_rate=5e-6,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
    num_generations=4,
    max_completion_length=128,
    temperature=0.8,
    top_p=0.95,
    bf16=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    logging_steps=1,
    save_strategy="steps",
    save_steps=5,
    save_total_limit=2,
    report_to="none",
    remove_unused_columns=False,
    use_vllm=False,
    seed=20261003,
)

trainer = GRPOTrainer(
    model=model,
    reward_funcs=exact_reward,
    args=config,
    train_dataset=dataset,
    processing_class=processor,
)

print("训练样本数：", len(dataset))
print("生成组大小：", config.num_generations)
print("计划优化步数：", config.max_steps)
print("开始 GRPO 冒烟训练……")

result = trainer.train()
trainer.save_model(str(OUTPUT_DIR / "adapter"))
processor.save_pretrained(OUTPUT_DIR / "processor")

summary = {
    "run": "grpo_smoke_10",
    "steps": config.max_steps,
    "train_metrics": result.metrics,
    "adapter": str(OUTPUT_DIR / "adapter"),
    "peak_gpu_memory_gib": round(
        torch.cuda.max_memory_allocated() / 1024**3, 3
    ),
}
(OUTPUT_DIR / "summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2, default=str),
    encoding="utf-8",
)
print("\n========== GRPO 冒烟训练完成 ==========")
print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
