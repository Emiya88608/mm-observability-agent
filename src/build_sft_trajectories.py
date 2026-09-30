from pathlib import Path
from collections import Counter
import json
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
TRAIN_FILE = DATA / "splits/train.jsonl"
TEST_FILE = DATA / "splits/test.jsonl"
METRIC_FILE = DATA / "tool_metrics.json"
OUT_DIR = DATA / "sft"
HARD_IMAGE_DIR = DATA / "images/sft_hard"

rng = np.random.default_rng(20260930)

def read_jsonl(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

def answer_for(incident):
    if incident == "latency_spike":
        return {
            "incident_type": "latency_spike",
            "affected_metric": "latency_ms",
            "recommended_action": "inspect_network_path",
        }
    if incident == "packet_loss_spike":
        return {
            "incident_type": "packet_loss_spike",
            "affected_metric": "packet_loss_pct",
            "recommended_action": "inspect_packet_loss",
        }
    return {
        "incident_type": "normal",
        "affected_metric": "none",
        "recommended_action": "continue_monitoring",
    }

def trajectory(sample_id, image, question, answer, tool_result, source, difficulty):
    tool_call = {
        "name": "query_metric_window",
        "arguments": {
            "window_minutes": 60,
            "metrics": ["latency_ms", "packet_loss_pct"],
        },
    }
    return {
        "id": sample_id,
        "image": image,
        "question": question,
        "answer": answer,
        "tool_call": tool_call,
        "tool_result": tool_result,
        "source": source,
        "difficulty": difficulty,
        "messages": [
            {
                "role": "system",
                "content": (
                    "你是网络监控异常分析 Agent。截图只能提供初步线索。"
                    "需要精确判断时必须调用 query_metric_window。"
                    "工具结果高于阈值时必须优先相信工具数值："
                    "latency_max_ms > 80 判为 latency_spike；"
                    "否则 packet_loss_max_pct > 3 判为 packet_loss_spike；"
                    "否则判为 normal。最终仅输出 JSON。"
                ),
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question},
                ],
            },
            {
                "role": "assistant",
                "content": json.dumps(tool_call, ensure_ascii=False),
            },
            {
                "role": "tool",
                "name": "query_metric_window",
                "content": json.dumps(tool_result, ensure_ascii=False),
            },
            {
                "role": "assistant",
                "content": json.dumps(answer, ensure_ascii=False),
            },
        ],
    }

def render_hard_case(sample_id, incident):
    minutes = np.arange(60)
    latency = 32 + rng.normal(0, 2.2, 60)
    loss = np.clip(0.25 + rng.normal(0, 0.07, 60), 0, None)

    if incident == "packet_loss_spike":
        # 覆盖阈值附近与明显丢包两类，专门针对现有 7 条 Bad Case。
        target_loss = float(rng.uniform(3.2, 5.5) if int(sample_id[-3:]) % 2 == 0
                            else rng.uniform(6.0, 12.0))
        point = int(rng.integers(20, 35))
        loss[point:point + 7] = target_loss + rng.normal(0, 0.12, 7)

    elif incident == "latency_spike":
        target_latency = float(rng.uniform(85, 130))
        point = int(rng.integers(20, 35))
        latency[point:point + 7] = target_latency + rng.normal(0, 2.0, 7)

    else:
        # 正常类也靠近丢包阈值，但严格小于 3%，作为负样本。
        target_loss = float(rng.uniform(1.8, 2.75))
        point = int(rng.integers(20, 35))
        loss[point:point + 7] = target_loss + rng.normal(0, 0.05, 7)

    HARD_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    path = HARD_IMAGE_DIR / f"{sample_id}.png"

    fig, axes = plt.subplots(2, 1, figsize=(9, 5.4), sharex=True)
    axes[0].plot(minutes, latency, color="#2D6CDF", linewidth=2)
    axes[0].axhline(80, color="#D64545", linestyle="--", linewidth=1)
    axes[0].set_ylabel("Latency (ms)")
    axes[0].grid(alpha=0.25)

    axes[1].plot(minutes, loss, color="#E28413", linewidth=2)
    axes[1].axhline(3, color="#D64545", linestyle="--", linewidth=1)
    axes[1].set_ylabel("Packet loss (%)")
    axes[1].set_xlabel("Minute")
    axes[1].grid(alpha=0.25)

    fig.suptitle("Network Observability Dashboard")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)

    return {
        "latency_max_ms": round(float(latency.max()), 2),
        "packet_loss_max_pct": round(float(loss.max()), 2),
    }

train_rows = read_jsonl(TRAIN_FILE)
test_rows = read_jsonl(TEST_FILE)
metrics = json.loads(METRIC_FILE.read_text(encoding="utf-8"))

train_ids = {row["id"] for row in train_rows}
test_ids = {row["id"] for row in test_rows}

assert not train_ids & test_ids, "原始 train/test 已发生泄漏，停止。"

question = "请分析该监控截图。仅输出 JSON：incident_type、affected_metric、recommended_action。"
records = []

# 保留全部原始训练样本，转为工具调用轨迹。
for row in train_rows:
    sample_id = row["id"]
    records.append(
        trajectory(
            sample_id=sample_id,
            image=row["image"],
            question=row["question"],
            answer=row["answer"],
            tool_result=metrics[sample_id],
            source="original_train",
            difficulty="standard",
        )
    )

# 新增 80 张完全独立的合成图，不碰验证集与测试集。
hard_plan = (
    ["packet_loss_spike"] * 40
    + ["normal"] * 20
    + ["latency_spike"] * 20
)

for i, incident in enumerate(hard_plan):
    sample_id = f"sft_hard_{i:03d}"
    tool_result = render_hard_case(sample_id, incident)
    answer = answer_for(incident)

    # 生成数据自身的一致性校验。
    if tool_result["latency_max_ms"] > 80:
        expected = "latency_spike"
    elif tool_result["packet_loss_max_pct"] > 3:
        expected = "packet_loss_spike"
    else:
        expected = "normal"

    assert expected == incident, (
        f"{sample_id} 标签与工具数值不一致：{incident} vs {tool_result}"
    )

    records.append(
        trajectory(
            sample_id=sample_id,
            image=f"images/sft_hard/{sample_id}.png",
            question=question,
            answer=answer,
            tool_result=tool_result,
            source="synthetic_hard_example",
            difficulty="packet_loss_focus" if incident == "packet_loss_spike" else "contrastive",
        )
    )

all_ids = {row["id"] for row in records}
assert not all_ids & test_ids, "SFT 数据混入测试集，停止。"
assert len(all_ids) == len(records), "SFT 数据存在重复 ID，停止。"

OUT_DIR.mkdir(parents=True, exist_ok=True)
out_file = OUT_DIR / "train_trajectories.jsonl"
with out_file.open("w", encoding="utf-8") as f:
    for row in records:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

summary = {
    "total": len(records),
    "original_train": len(train_rows),
    "synthetic_hard_examples": len(hard_plan),
    "label_distribution": dict(Counter(row["answer"]["incident_type"] for row in records)),
    "test_leakage_check": False,
    "output": str(out_file),
}

(OUT_DIR / "dataset_summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print("========== SFT 轨迹数据构建完成 ==========")
print(json.dumps(summary, ensure_ascii=False, indent=2))
print("新增图片目录：", HARD_IMAGE_DIR)
