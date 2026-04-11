# Integration Notes (Real Ryu + Real Network)

This repository is now configured for live integration only. Offline mock environment modules were removed.

## 1. Controller and topology values to verify in prod.json

- environment.ryu_controller.base_url
  - Must point to the active Ryu REST server.
- environment.network.switch_dpid
  - Must match the Core switch DPID used for QoS actions.
- environment.network.link_capacity_mbps
  - Must match your actual Mininet/physical link capacity.
- environment.network.stabilization_delay_seconds
  - Tune based on how quickly telemetry reflects policy changes.
- environment.monitoring.main_pair.src/dst
  - Host pair used for latency/loss telemetry.
- environment.service_policies.*.destination_ip
  - Must match real destination hosts/services.

## 2. Endpoint contract checklist

Confirm Ryu apps expose exactly the payloads expected by:

- ai_layer/network_interface/ryu_client.py
- ai_layer/network_interface/action_translator.py
- ai_layer/network_interface/telemetry_parser.py

Critical routes:

- GET /links/utilization
- GET /latency/{src}/{dst}
- POST/PUT endpoints used for queue and rate-limit actions
- POST /network/reset (optional but strongly recommended)

If /network/reset is not available, implement one of:

- a custom Ryu endpoint to clear QoS/flow state, or
- an external reset script and call it before each run.

## 3. Telemetry assumptions

- Link utilization values must be compatible with parser normalization.
- Latency endpoint must include latency and packet loss fields expected by parser.
- If controller returns cumulative counters instead of interval values, parser logic must convert to deltas per poll.

## 4. Smoke-test order

1. Start Mininet topology and traffic generators.
2. Start Ryu apps and confirm REST endpoints are reachable.
3. Run python clint_test.py and verify:
   - telemetry calls succeed,
   - action translation calls return success,
   - SDNEnv reset/step executes without exceptions.
4. Run short evaluation:
   - python evaluate.py --config prod.json --model-path models/dqn_model.pth --metrics-path logs/evaluation_live_smoke.json
5. Run short training:
   - python train.py --config prod.json --seed 42 --model-path models/dqn_model_live_smoke.pth

## 5. Operational notes

- Keep traffic load active during training/evaluation; static idle links produce low-signal rewards.
- Ensure NTP/time sync if latency telemetry is one-way or timestamp-based.
- Persist Ryu logs and metrics JSON for post-run analysis.
