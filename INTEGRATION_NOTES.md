# Integration Notes (Real Ryu + Real Network)

This repository is now configured for live integration only. Offline mock environment modules were removed.

## 1. Controller and topology values to verify in prod.json

- environment.ryu_controller.base_url
  - Must point to the active Ryu REST server.
- environment.network.switch_dpid
  - Must match the Core switch DPID used for QoS actions.
- environment.network.main_link_capacity_mbps and backup_link_capacity_mbps
  - Must match actual path capacities used for normalization.
- environment.network.stabilization_delay_seconds
  - Tune based on how quickly telemetry reflects policy changes.
- environment.monitoring.main_pair.src/dst
  - Pair used for latency/loss telemetry.
- environment.monitoring.latency_min_ms, latency_max_ms, packet_loss_max_percent
  - Normalization bands for 6D state.
- environment.startup_setup
  - Routing addresses/routes/default gateways and baseline QoS rules/queues.
- environment.action_space
  - Runtime action targets for update_queue/failover/reroute.

## 2. Endpoint contract checklist

Confirm Ryu apps expose exactly the payloads expected by:

- ai_layer/network_interface/ryu_client.py
- ai_layer/network_interface/action_translator.py
- ai_layer/network_interface/telemetry_parser.py

Critical routes:

- PUT /v1.0/conf/switches/{switch_id}/ovsdb_addr
- POST /router/{switch_id}
- GET /links/utilization
- GET /latency/{src}/{dst}
- POST /qos/rules/{switch_id}
- POST /qos/queue/{switch_id}
- POST /network/reset (optional)

If /network/reset is not available, implement one of:

- a custom Ryu endpoint to clear QoS/flow state, or
- an external reset script and call it before each run.

## 3. Telemetry assumptions

- /links/utilization must contain per-link tx_mbps for core -> sp1, core -> sp2, sp1 -> lf1, sp2 -> lf1.
- /latency/{src}/{dst} must include latency_ms and packet_loss_percent.
- If controller returns cumulative counters instead of interval values, parser logic must convert to deltas per poll.

## 4. Smoke-test order

1. Start Mininet topology and traffic generators.
2. Start Ryu apps and confirm REST endpoints are reachable.
3. Run python setup_network.py --config prod.json and verify startup setup summary is successful.
4. Run python clint_test.py and verify:
   - telemetry calls succeed,
   - action translation calls return success,
   - SDNEnv reset/step executes without exceptions.
5. Run short evaluation:
   - python evaluate.py --config prod.json --model-path models/dqn_model.pth --metrics-path logs/evaluation_live_smoke.json
6. Run short training:
   - python train.py --config prod.json --seed 42 --model-path models/dqn_model_live_smoke.pth

## 5. Operational notes

- Keep traffic load active during training/evaluation; static idle links produce low-signal rewards.
- Runtime action semantics are optimization-only:
  - update_queue adjusts queue profile
  - failover changes route to backup gateway
  - reroute restores route to main gateway
- Ensure NTP/time sync if latency telemetry is one-way or timestamp-based.
- Persist Ryu logs and metrics JSON for post-run analysis.
