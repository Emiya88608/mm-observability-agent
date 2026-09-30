import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2_5_VLForConditionalGeneration,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data/sft/train_trajectories.jsonl"
MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"

SYSTEM_PROMPT = """你是网络监控异常分析 Agent。
截图仅提供初步线索；精确数值以 query_metric_window 的工具结果为准。
判定规则：
- latency_max_ms > 80：latency_spike；
- 否则 packet_loss_max_pct > 3：packet_loss_spike；
- 否则：normal。
最终只能输出 incident_type、affected_metric、recommended_action 三个字段组成的 JSON。"""

class TrajectoryDataset(Dataset):
    def __init__(self, path):
        self.rows = [
            json.loads(line)
            for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]

def make_messages(row, include_answer):
    image_path = ROOT / "data" / row["image"]
    tool_call = json.dumps(row["tool_call"], ensure_ascii=False)
    tool_result = json.dumps(row["tool_result"], ensure_ascii=False)

    user_text = (
        f"{row['question']}\n\n"
        f"已调用工具 query_metric_window：{tool_call}\n"
        f"工具返回的精确数值：{tool_result}\n"
        "请依据工具数值给出最终结论。"
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": user_text},
            ],
        },
    ]

    if include_answer:
        messages.append({
            "role": "assistant",
            "content": json.dumps(row["answer"], ensure_ascii=False),
        })

    return messages, image_path

class VisionSFTCollator:
    def __init__(self, processor):
        self.processor = processor
        self.processor.tokenizer.padding_side = "right"

    def __call__(self, rows):
        images = []
        full_texts = []
        prompt_lengths = []

        for row in rows:
            full_messages, image_path = make_messages(row, include_answer=True)
            prompt_messages, _ = make_messages(row, include_answer=False)

            image = Image.open(image_path).convert("RGB")
            full_text = self.processor.apply_chat_template(
                full_messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            prompt_text = self.processor.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            prompt_inputs = self.processor(
                text=[prompt_text],
                images=[image],
                return_tensors="pt",
            )

            images.append(image)
            full_texts.append(full_text)
            prompt_lengths.append(prompt_inputs["input_ids"].shape[1])

        batch = self.processor(
            text=full_texts,
            images=images,
            padding=True,
            return_tensors="pt",
        )

        labels = batch["input_ids"].clone()
        labels[batch["attention_mask"] == 0] = -100

        for row_index, prompt_length in enumerate(prompt_lengths):
            labels[row_index, :prompt_length] = -100

        batch["labels"] = labels
        return batch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_steps", type=int, default=10)
    parser.add_argument("--run_name", type=str, default="smoke_10")
    args = parser.parse_args()

    output_dir = ROOT / "artifacts/qlora_runs" / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        raise RuntimeError("未检测到 CUDA，停止训练。")

    free, total = torch.cuda.mem_get_info()
    print(f"GPU 可用显存：{free / 1024**3:.2f} / {total / 1024**3:.2f} GiB")

    dataset = TrajectoryDataset(DATA_FILE)
    print(f"SFT 训练样本数：{len(dataset)}")

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # 限制视觉 token 数，避免监控截图在 24GB 显存上占用过大。
    processor = AutoProcessor.from_pretrained(
        MODEL_NAME,
        min_pixels=256 * 28 * 28,
        max_pixels=512 * 28 * 28,
    )

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        max_steps=args.max_steps,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=1,
        logging_first_step=True,
        save_strategy="no",
        report_to="none",
        bf16=True,
        fp16=False,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        remove_unused_columns=False,
        dataloader_num_workers=0,
        seed=20260930,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=VisionSFTCollator(processor),
    )

    result = trainer.train()

    adapter_dir = output_dir / "adapter"
    model.save_pretrained(adapter_dir)
    processor.save_pretrained(output_dir / "processor")

    metrics = {
        "run_name": args.run_name,
        "max_steps": args.max_steps,
        "train_metrics": result.metrics,
        "peak_gpu_memory_gib": round(
            torch.cuda.max_memory_allocated() / 1024**3, 3
        ),
        "adapter_dir": str(adapter_dir),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    print("\n========== QLoRA 训练完成 ==========")
    print(json.dumps(metrics, ensure_ascii=False, indent=2, default=str))

if __name__ == "__main__":
    main()
