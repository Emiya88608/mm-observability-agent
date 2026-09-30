from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, unquote
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
METRICS_FILE = ROOT / "data/ood_eval_v4/tool_metrics.json"
TEST_FILE = ROOT / "data/ood_eval_v4/test.jsonl"

if not METRICS_FILE.exists():
    raise FileNotFoundError(f"找不到工具指标文件：{METRICS_FILE}")

METRICS = json.loads(METRICS_FILE.read_text(encoding="utf-8"))

def query_metric_window(sample_id):
    """本地工具模拟器：根据样本 ID 查询监控指标。"""
    if sample_id not in METRICS:
        raise KeyError(f"未知样本 ID：{sample_id}")
    return METRICS[sample_id]

def classify_metrics(metrics):
    """按当前实验定义的阈值进行确定性分类。"""
    latency = float(metrics["latency_max_ms"])
    packet_loss = float(metrics["packet_loss_max_pct"])

    if latency > 80:
        label = "latency_spike"
        metric = "latency_ms"
        action = "inspect_network_path"
    elif packet_loss > 3:
        label = "packet_loss_spike"
        metric = "packet_loss_pct"
        action = "inspect_packet_loss"
    else:
        label = "normal"
        metric = "none"
        action = "continue_monitoring"

    return {
        "incident_type": label,
        "affected_metric": metric,
        "recommended_action": action,
    }

def analyze(sample_id):
    metrics = query_metric_window(sample_id)
    diagnosis = classify_metrics(metrics)
    return {
        "sample_id": sample_id,
        "tool": {
            "name": "query_metric_window",
            "status": "success",
            "result": metrics,
        },
        "diagnosis": diagnosis,
        "explanation": (
            f"指标证据：最大时延 {metrics['latency_max_ms']} ms"
            f"（告警阈值 80 ms），最大丢包率 "
            f"{metrics['packet_loss_max_pct']}%（告警阈值 3%）。"
            f"后端规则判定为 {diagnosis['incident_type']}，"
            f"建议动作：{diagnosis['recommended_action']}。"
        ),
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
            return self.send_json(200, {"status": "ok"})

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
            "error": "路径不存在",
            "available": ["/health", "/samples", "/analyze/{sample_id}"],
        })

    def log_message(self, format, *args):
        print("[HTTP]", format % args)

def self_test():
    if not TEST_FILE.exists():
        raise FileNotFoundError(f"找不到 v4 测试集：{TEST_FILE}")

    rows = [
        json.loads(line)
        for line in TEST_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    class_correct = 0
    full_correct = 0

    for row in rows:
        result = analyze(row["id"])["diagnosis"]
        expected = row["answer"]
        class_correct += int(
            result["incident_type"] == expected["incident_type"]
        )
        full_correct += int(
            all(result.get(key) == value for key, value in expected.items())
        )

    print("========== 后端 API 自检 ==========")
    print("工具指标条数：", len(METRICS))
    print("测试样本数：", len(rows))
    print(f"类别准确率：{class_correct}/{len(rows)}")
    print(f"完整字段准确率：{full_correct}/{len(rows)}")
    print("注意：v4 标签与规则阈值同源，这不是独立泛化评测。")

if __name__ == "__main__":
    if "--self-test" in sys.argv:
        self_test()
    else:
        server = ThreadingHTTPServer(("127.0.0.1", 8001), Handler)
        print("API 已启动：http://127.0.0.1:8000")
        print("健康检查：GET /health")
        print("样本列表：GET /samples")
        print("诊断接口：GET /analyze/{sample_id}")
        print("按 Ctrl+C 停止")
        server.serve_forever()
