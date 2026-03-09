import json
from ai_layer.network_interface.ryu_client import RyuClient
from ai_layer.network_interface.telemetry_parser import TelemetryParser
from ai_layer.network_interface.action_translator import ActionTranslator

cfg = json.load(open("prod.json"))
client = RyuClient(cfg["environment"]["ryu_controller"])
dpid = cfg["environment"]["network"]["switch_dpid"]
link_cap = cfg["environment"]["network"]["link_capacity_bps"]
num_ports = cfg["environment"]["network"]["num_ports"]

parser = TelemetryParser(link_capacity_bps=link_cap, num_ports=num_ports)
translator = ActionTranslator(client, cfg)

# --- 1. Raw port stats from Ryu client ---
raw_ports = client.get_port_stats(dpid)
print("=== Raw port stats ===")
print(json.dumps(raw_ports, indent=2))

# --- 2. Parsed into PortStatistics objects ---
port_stats = parser.parse_port_stats(raw_ports, dpid)
print("\n=== Parsed port stats ===")
for port_no, ps in sorted(port_stats.items()):
    print(f"  Port {port_no}: tx_bytes={ps.tx_bytes}, rx_dropped={ps.rx_dropped}")

# --- 3. Flow stats ---
raw_flows = client.get_flow_stats(dpid)
flows = parser.parse_flow_stats(raw_flows, dpid)
print(f"\n=== Flow entries: {len(flows)} ===")
for f in flows:
    print(f"  priority={f.priority}  packets={f.packet_count}")

# --- 4. State vector (the key output) ---
state = parser.build_state(port_stats)
print("\n=== RL State Vector ===")
labels = ["link1_util", "link2_util", "link3_util", "packet_loss", "traffic_load"]
for label, val in zip(labels, state):
    print(f"  {label:15s} = {val:.4f}")
print(f"\n  raw array: {state}")

# --- 5. One-shot shortcut ---
state2 = parser.response_to_state(raw_ports, dpid)
print(f"  shortcut:  {state2}")

# --- 6. Action Translator ---
print("\n=== Action Translator ===")
for action_id in range(5):
    result = translator.execute(action_id)
    print(f"  Action {result.action_id} ({result.action_name}): "
          f"success={result.success}  msg={result.message}")

# --- 7. Gymnasium Environment (3 steps) ---
from ai_layer.environments.sdn_env import SDNEnv

print("\n=== Gym Environment (3 steps) ===")
env = SDNEnv(cfg)
state, info = env.reset()
print(f"  Reset state: {state}")

for i in range(3):
    action = env.action_space.sample()
    state, reward, terminated, truncated, info = env.step(action)
    print(f"  Step {info['step']}: action={action} ({info['action_name']})  "
          f"reward={reward:.4f}  state={state}")
    env.render()

# --- 8. Mock Environment (offline, no server needed) ---
from ai_layer.environments.mock_env import MockSDNEnv

print("\n=== Mock Environment (5 steps, no server) ===")
mock_env = MockSDNEnv(cfg)
state, info = mock_env.reset(seed=42)
print(f"  Reset state: {state}")

for i in range(5):
    action = mock_env.action_space.sample()
    state, reward, terminated, truncated, _ = mock_env.step(action)
    action_names = ["do_nothing", "route_q0", "route_q1", "rate_limit", "rm_limit"]
    print(f"  Step {i+1}: action={action} ({action_names[action]})  "
          f"reward={reward:.4f}  state={state}")
    mock_env.render()