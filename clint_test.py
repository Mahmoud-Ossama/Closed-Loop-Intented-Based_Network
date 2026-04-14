import json

from ai_layer.environments.sdn_env import SDNEnv
from ai_layer.network_interface.action_translator import ActionTranslator
from ai_layer.network_interface.ryu_client import RyuClient
from ai_layer.network_interface.telemetry_parser import TelemetryParser
from ai_layer.network_setup import NetworkInitializer

cfg = json.load(open("prod.json", encoding="utf-8"))

# --- 0. Startup Setup (routing + baseline QoS) ---
print("=== Startup setup ===")
setup_summary = NetworkInitializer(cfg).initialize()
print(json.dumps(setup_summary.as_dict(), indent=2))

client = RyuClient(cfg["environment"]["ryu_controller"])
mon_cfg = cfg["environment"]["monitoring"]
net_cfg = cfg["environment"]["network"]

parser = TelemetryParser(
    main_link_capacity_mbps=net_cfg["main_link_capacity_mbps"],
    backup_link_capacity_mbps=net_cfg["backup_link_capacity_mbps"],
    latency_min_ms=mon_cfg["latency_min_ms"],
    latency_max_ms=mon_cfg["latency_max_ms"],
    packet_loss_max_percent=mon_cfg["packet_loss_max_percent"],
)
translator = ActionTranslator(client, cfg)

# --- 1. Raw link utilization ---
raw_links = client.get_link_utilization()
print("\n=== Raw link utilization ===")
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
print(f"  core -> sp1: {link_map.get('core -> sp1', 0.0):.2f} Mbps")
print(f"  sp1 -> lf1 : {link_map.get('sp1 -> lf1', 0.0):.2f} Mbps")
print(f"  core -> sp2: {link_map.get('core -> sp2', 0.0):.2f} Mbps")
print(f"  sp2 -> lf1 : {link_map.get('sp2 -> lf1', 0.0):.2f} Mbps")
print(f"  latency_ms={lat_map['latency_ms']:.3f}  packet_loss_percent={lat_map['packet_loss_percent']:.3f}")

# --- 4. State vector ---
state = parser.build_state(raw_links, raw_latency, failover_active=False)
print("\n=== RL State Vector (6D) ===")
labels = [
    "latency",
    "packet_loss",
    "throughput",
    "main_link_util",
    "backup_link_util",
    "failover_active",
]
for label, val in zip(labels, state):
    print(f"  {label:16s} = {val:.4f}")
print(f"\n  raw array: {state}")

# --- 5. Action Translator ---
print("\n=== Runtime Actions ===")
for action_id in range(cfg["environment"]["action_space"]["dimension"]):
    result = translator.execute(action_id)
    print(
        f"  Action {result.action_id} ({result.action_name}): "
        f"success={result.success} msg={result.message} metadata={result.metadata}"
    )

# --- 6. Live Gym Environment (3 steps) ---
print("\n=== Gym Environment (3 steps) ===")
env = SDNEnv(cfg)
state, info = env.reset()
print(f"  Reset state: {state}")
print(f"  Reset info : {info}")

for _ in range(3):
    action = env.action_space.sample()
    state, reward, terminated, truncated, info = env.step(action)
    print(
        f"  Step {info['step']}: action={action} ({info['action_name']}) "
        f"failover={info['failover_active']} reward={reward:.4f}"
    )
    env.render()
