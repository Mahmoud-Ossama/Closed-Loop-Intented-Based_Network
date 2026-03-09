"""
Mock Ryu server for testing the AI Layer without a real Ryu controller.

Usage:
    python -m ai_layer.network_interface.mock_ryu_server

Runs on localhost:8080. Responds to the same endpoints as a real Ryu controller
with randomized but realistic fake data. Logs every POST it receives.
"""

import json
import random
from http.server import HTTPServer, BaseHTTPRequestHandler

DPID = "0000000000000001"


def _random_port_stats():
    """Generate realistic fake port stats for 3 ports.
    
    Values simulate a single polling interval (delta counters),
    scaled so utilization falls in a useful 0-1 range.
    link_capacity = 1 Mbps = 125,000 bytes/sec.
    """
    ports = []
    for port_no in range(1, 4):
        tx = random.randint(10_000, 120_000)  # 8%-96% utilization
        ports.append({
            "port_no": port_no,
            "rx_packets": random.randint(50, 500),
            "tx_packets": random.randint(50, 500),
            "rx_bytes": random.randint(10_000, 120_000),
            "tx_bytes": tx,
            "rx_dropped": random.randint(0, 5),
            "tx_dropped": random.randint(0, 3),
            "rx_errors": 0,
            "tx_errors": 0,
        })
    return {DPID: ports}


def _random_flow_stats():
    """Generate fake flow stats."""
    return {
        DPID: [
            {
                "priority": 1,
                "match": {},
                "actions": ["OUTPUT:CONTROLLER"],
                "packet_count": random.randint(100, 2000),
                "byte_count": random.randint(10_000, 200_000),
            },
            {
                "priority": 100,
                "match": {"dl_type": 2048, "nw_proto": 17, "tp_dst": 5002},
                "actions": ["OUTPUT:1"],
                "packet_count": random.randint(50, 1000),
                "byte_count": random.randint(5_000, 100_000),
            },
        ]
    }


class MockRyuHandler(BaseHTTPRequestHandler):

    def do_GET(self):
        if "/stats/port/" in self.path:
            data = _random_port_stats()
        elif "/stats/flow/" in self.path:
            data = _random_flow_stats()
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

    def _respond(self, code, data):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    # Suppress per-request log noise
    def log_message(self, fmt, *args):
        print(f"  {args[0]}")


if __name__ == "__main__":
    HOST, PORT = "localhost", 8080
    server = HTTPServer((HOST, PORT), MockRyuHandler)
    print(f"Mock Ryu controller running on http://{HOST}:{PORT}")
    print(f"DPID: {DPID}")
    print("Press Ctrl+C to stop.\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.server_close()
