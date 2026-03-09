# Integration Notes

Things to update once the Ryu controller is connected.

---

## ryu_client.py
- **Tested with:** `mock_ryu_server.py` (fake data on localhost:8080)
- **To verify:** All endpoints return expected JSON structure from real Ryu
- **Potential change:** Endpoint paths may differ if Ryu uses custom REST apps instead of `ofctl_rest` / `rest_qos`

## telemetry_parser.py
- **Current assumption:** Port stats are **delta** values (single polling interval)
- **To check:** If Ryu returns **cumulative** counters, we need to store previous values and compute deltas between polls
- **To check:** Real Ryu port stats key format — currently expects `{dpid: [{port_no, tx_bytes, ...}]}`
- **Potential change:** `link_capacity_bps` division (bits vs bytes) — confirm unit of `tx_bytes` from real controller

## mock_ryu_server.py
- Generates random data scaled for 1 Mbps links
- **Drop** this once real Ryu is available, or keep for offline dev

## prod.json
- `base_url`: Confirm port (currently `localhost:8080`)
- `switch_dpid`: Confirm actual DPID from Mininet topology
- `link_capacity_bps`: Confirm matches real link speed
- `api_endpoints`: Confirm paths match Ryu REST app routes

## action_translator.py
- **To check:** QoS rule JSON body format — may need adjusting for real Ryu `rest_qos` app
- **To check:** Flow entry format for `install_flow` / `delete_flow` if used
- **To check:** `linux-htb` queue type — confirm Ryu supports it in the deployment

## sdn_env.py
- **stabilization_delay:** Currently sleeps 2s between steps — may need tuning with real network
- **reset():** Calls `/network/reset` — confirm this endpoint exists on real Ryu
- **To check:** Whether cumulative vs delta counters affect state between steps

## reward.py
- Reads weights/thresholds from `prod.json` reward_function section
- **To tune:** Loss weight (100.0) and congestion penalty (5.0) after real training runs

---

## Not yet implemented
- `environments/mock_env.py` — Offline simulated environment
- `agent/dqn_agent.py` — DQN agent
- `agent/replay_buffer.py` — Experience replay
- `models/q_network.py` — Neural network
- `training/trainer.py` — Training loop
- `utils/config.py` — Config loader
