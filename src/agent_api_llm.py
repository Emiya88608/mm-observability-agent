from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, unquote
from threading import Lock
import json
import torch

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
ADAPTER_DIR = ROOT / "artifacts/qlora_runs/sft_tool_grounded_100/adapter"
METRICS_FILE = ROOT / "data/ood_eval_v4/tool_metrics.json"

if not METRICS_FILE.exists():
    raise FileNotFoundError(f"找不到工具指标文件：{METRICS_FILE}")
if not ADAPTER_DIR.is_dir():
    raise FileNotFoundError(f"找不到 SFT adapter：{ADAPTER_DIR}")

METRICS = json.loads(METRICS_FILE.read_text(encoding="utf-8"))
MODEL = None
PROCESSOR = None
MODEL_ERROR = None
GENERATION_LOCK = Lock()

def load_model():
    global MODEL, PROCESSOR, MODEL_ERROR

    try:
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        from peft import PeftModel

        if not torch.cuda.is_available():
            raise RuntimeError("未检测到 CUDA；为避免意外占用 CPU，不加载模型。")

        print("正在 GPU 0 加载基础模型和 SFT adapter……", flush=True)
        PROCESSOR = AutoProcessor.from_pretrained(
            MODEL_ID,
            local_files_only=True,
        )
        base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            device_map={"": 0},
            local_files_only=True,
        )
        MODEL = PeftModel.from_pretrained(
            base_model,
            str(ADAPTER_DIR),
            local_files_only=True,
        )
        MODEL.eval()
        print("模型加载完成。", flush=True)
    except Exception as exc:
        MODEL = None
        PROCESSOR = None
        MODEL_ERROR = f"{type(exc).__name__}: {exc}"
        print("模型未能加载，将使用规则模板解释：", MODEL_ERROR, flush=True)

def query_metric_window(sample_id):
    if sample_id not in METRICS:
        raise KeyError(f"未知样本 ID：{sample_id}")
    return METRICS[sample_id]

def classify_metrics(metrics):
    latency = float(metrics["latency_max_ms"])
    packet_loss = float(metrics["packet_loss_max_pct"])

    if latency > 80:
        return {
            "incident_type": "latency_spike",
            "affected_metric": "latency_ms",
            "recommended_action": "inspect_network_path",
        }
    if packet_loss > 3:
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

def fallback_explanation(metrics, diagnosis):
    return (
        f"后端读取到最大时延 {metrics['latency_max_ms']} ms、"
        f"最大丢包率 {metrics['packet_loss_max_pct']}%。"
        f"按配置阈值，当前判定为 {diagnosis['incident_type']}；"
        f"建议执行 {diagnosis['recommended_action']}。"
    )

def generate_explanation(metrics, diagnosis):
    if MODEL is None or PROCESSOR is None:
        return fallback_explanation(metrics, diagnosis), "fallback"

    prompt = f"""
后端已完成确定性诊断，请只用中文写一到两句话解释依据和建议。
不要更改诊断结论，不要输出 JSON，不要猜测没有证据支持的根因。
诊断结论：{diagnosis["incident_type"]}
受影响指标：{diagnosis["affected_metric"]}
建议动作：{diagnosis["recommended_action"]}
最大时延：{metrics["latency_max_ms"]} ms（时延告警阈值：80 ms）
最大丢包率：{metrics["packet_loss_max_pct"]}%（丢包告警阈值：3%）
"""

    messages = [{
        "role": "user",
        "content": [{"type": "text", "text": prompt}],
    }]

    try:
        with GENERATION_LOCK, torch.inference_mode():
            text = PROCESSOR.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = PROCESSOR(
                text=[text],
                padding=True,
                return_tensors="pt",
            ).to(MODEL.device)

            output_ids = MODEL.generate(
                **inputs,
                max_new_tokens=120,
                do_sample=False,
            )
            new_tokens = output_ids[:, inputs.input_ids.shape[1]:]
            explanation = PROCESSOR.batch_decode(
                new_tokens,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()

        if not explanation:
            return fallback_explanation(metrics, diagnosis), "fallback_empty"
        return explanation, "qwen_sft_adapter"
    except Exception as exc:
        print("生成解释失败：", repr(exc), flush=True)
        return fallback_explanation(metrics, diagnosis), "fallback_error"

def analyze(sample_id):
    metrics = query_metric_window(sample_id)
    diagnosis = classify_metrics(metrics)
    explanation, explanation_source = generate_explanation(metrics, diagnosis)

    return {
        "sample_id": sample_id,
        "tool": {
            "name": "query_metric_window",
            "status": "success",
            "result": metrics,
        },
        "diagnosis": diagnosis,
        "explanation": explanation,
        "explanation_source": explanation_source,
        "model_changes_diagnosis": False,
    }

class Handler(BaseHTTPRequestHandler):
    def send_json(self, status, payload):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path.rstrip("/")

        if path == "/health":
            return self.send_json(200, {
                "status": "ok",
                "explanation_model_loaded": MODEL is not None,
                "model_error": MODEL_ERROR,
            })

        if path == "/samples":
            return self.send_json(200, {
                "total": len(METRICS),
                "sample_ids": list(METRICS.keys())[:20],
            })

        if path.startswith("/analyze/"):
            sample_id = unquote(path.rsplit("/", 1)[-1])
            try:
                return self.send_json(200, analyze(sample_id))
            except KeyError as exc:
                return self.send_json(404, {"error": str(exc)})
            except Exception as exc:
                return self.send_json(500, {"error": str(exc)})

        return self.send_json(404, {
            "available": ["/health", "/samples", "/analyze/{sample_id}"],
        })

    def log_message(self, format, *args):
        print("[HTTP]", format % args, flush=True)

if __name__ == "__main__":
    load_model()
    server = ThreadingHTTPServer(("127.0.0.1", 8000), Handler)
    print("API 已启动：http://127.0.0.1:8000", flush=True)
    print("按 Ctrl+C 停止服务。", flush=True)
    server.serve_forever()
