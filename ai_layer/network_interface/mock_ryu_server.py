"""
Mock Ryu server for testing AI Layer with updated telemetry APIs.

Usage:
    python -m ai_layer.network_interface.mock_ryu_server
"""

import json
import random
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer

CORE_DPID = "0000000000000030"

LINKS = [
    "RAN -> agg",
    "agg -> core",
    "core -> sp1",
    "core -> sp2",
    "sp1 -> lf1",
    "sp2 -> lf1",
]


def _random_flow_stats():
    return {
        CORE_DPID: [
            {
                "priority": 1,
                "match": {},
                "actions": ["OUTPUT:CONTROLLER"],
                "packet_count": random.randint(100, 3000),
                "byte_count": random.randint(10_000, 400_000),
            },
            {
                "priority": 100,
                "match": {"dl_type": 2048, "nw_proto": 17, "tp_dst": 5002},
                "actions": ["OUTPUT:1"],
                "packet_count": random.randint(50, 1200),
                "byte_count": random.randint(5_000, 200_000),
            },
        ]
    }


def _random_link_utilization():
    links = []
    for link in LINKS:
        tx_mbps = round(random.uniform(0.0, 95.0), 2)
        links.append({"link": link, "tx_mbps": tx_mbps})

    return {
        "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "links": links,
    }


def _random_latency(src: str, dst: str):
    latency = round(random.uniform(1.0, 140.0), 3)
    loss_pct = round(random.uniform(0.0, 8.0), 3)
    return {
        "src": src,
        "dst": dst,
        "latency_ms": f"{latency}",
        "packet_loss_percent": f"{loss_pct}",
        "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S.%f"),
    }


class MockRyuHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if "/stats/flow/" in self.path:
            data = _random_flow_stats()
        elif self.path.startswith("/links/utilization"):
            data = _random_link_utilization()
        elif self.path.startswith("/latency/"):
            parts = self.path.strip("/").split("/")
            if len(parts) >= 3:
                src, dst = parts[1], parts[2]
            else:
                src, dst = "G6_D1", "URLLC"
            data = _random_latency(src, dst)
        elif "/qos/queue/" in self.path:
            data = {"queues": []}
        else:
            data = {}

        self._respond(200, data)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length)) if length else {}
        print(f"[POST] {self.path}  body={json.dumps(body, indent=2)}")
        self._respond(200, {"status": "ok"})

    def do_PUT(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length).decode() if length else ""
        print(f"[PUT]  {self.path}  body={body}")
        self._respond(200, {"status": "ok"})

    def _respond(self, code, data):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def log_message(self, fmt, *args):
        print(f'  {args[0]}')


if __name__ == "__main__":
    host, port = "localhost", 8080
    server = HTTPServer((host, port), MockRyuHandler)
    print(f"Mock Ryu controller running on http://{host}:{port}")
    print(f"Core DPID: {CORE_DPID}")
    print("Available telemetry: /links/utilization and /latency/{src}/{dst}")
    print("Press Ctrl+C to stop.\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.server_close()
