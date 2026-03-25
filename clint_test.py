import json

from ai_layer.environments.mock_env import MockSDNEnv
from ai_layer.environments.sdn_env import SDNEnv
from ai_layer.network_interface.action_translator import ActionTranslator
from ai_layer.network_interface.ryu_client import RyuClient
from ai_layer.network_interface.telemetry_parser import TelemetryParser

cfg = json.load(open("prod.json"))
client = RyuClient(cfg["environment"]["ryu_controller"])
mon_cfg = cfg["environment"]["monitoring"]

parser = TelemetryParser(
    link_capacity_mbps=cfg["environment"]["network"]["link_capacity_mbps"],
    latency_cap_ms=mon_cfg["latency_cap_ms"],
)
translator = ActionTranslator(client, cfg)

# --- 1. Raw link utilization ---
raw_links = client.get_link_utilization()
print("=== Raw link utilization ===")
print(json.dumps(raw_links, indent=2))

# --- 2. Raw latency ---
lat_src = mon_cfg["main_pair"]["src"]
lat_dst = mon_cfg["main_pair"]["dst"]
raw_latency = client.get_latency(lat_src, lat_dst)
print("\n=== Raw latency ===")
print(json.dumps(raw_latency, indent=2))

# --- 3. Parsed telemetry ---
link_map = parser.parse_link_utilization(raw_links)
lat_map = parser.parse_latency(raw_latency)
print("\n=== Parsed telemetry ===")
for link_name in parser.LINK_ORDER:
    print(f"  {link_name:12s}: tx_mbps={link_map.get(link_name, 0.0):.2f}")
print(f"  latency_ms={lat_map['latency_ms']:.3f}  packet_loss={lat_map['packet_loss']:.5f}")

# --- 4. State vector ---
active_service = mon_cfg["active_service"]
state = parser.build_state(raw_links, raw_latency, active_service)
print("\n=== RL State Vector (12D) ===")
labels = [
    "u_ran_agg", "u_agg_core", "u_core_sp1", "u_core_sp2",
    "u_sp1_lf1", "u_sp2_lf1", "latency", "packet_loss",
    "traffic", "svc_urllc", "svc_embb", "svc_mmtc",
]
for label, val in zip(labels, state):
    print(f"  {label:12s} = {val:.4f}")
print(f"\n  raw array: {state}")

# --- 5. Action Translator ---
print("\n=== Action Translator (Core QoS) ===")
for action_id in range(5):
    result = translator.execute(action_id)
    print(
        f"  Action {result.action_id} ({result.action_name}): "
        f"success={result.success}  msg={result.message}"
    )

# --- 6. Live Gym Environment (3 steps) ---
print("\n=== Gym Environment (3 steps) ===")
env = SDNEnv(cfg)
state, info = env.reset()
print(f"  Reset state: {state}")

for i in range(3):
    action = env.action_space.sample()
    state, reward, terminated, truncated, info = env.step(action)
    print(
        f"  Step {info['step']}: action={action} ({info['action_name']})  "
        f"service={info['service_type']}  reward={reward:.4f}"
    )
    env.render()

# --- 7. Mock Environment (5 steps, no server) ---
print("\n=== Mock Environment (5 steps, no server) ===")
mock_env = MockSDNEnv(cfg)
state, info = mock_env.reset(seed=42, options={"service_type": "eMBB"})
print(f"  Reset state: {state}")

for i in range(5):
    action = mock_env.action_space.sample()
    state, reward, terminated, truncated, info = mock_env.step(action)
    action_names = ["do_nothing", "policy_q0", "policy_q1", "rate_limit", "rm_limit"]
    print(
        f"  Step {i + 1}: action={action} ({action_names[action]})  "
        f"service={info['service_type']}  reward={reward:.4f}"
    )
    mock_env.render()
