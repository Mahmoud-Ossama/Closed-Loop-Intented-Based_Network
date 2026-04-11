# CLAUDE.md

This file provides guidance when working with code in this repository.

## Project Overview

Reinforcement Learning system for SDN traffic optimization.
A DQN agent learns routing/QoS actions by communicating with a real Ryu SDN controller through REST APIs.

## Development Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run live integration smoke test (requires Ryu + network running)
python clint_test.py

# Train DQN on live environment
python train.py --config prod.json --model-path models/dqn_model_live.pth

# Evaluate trained policy and baselines
python evaluate.py --config prod.json --model-path models/dqn_model_live.pth
```

## Architecture

```
ai_layer/
├── network_interface/          # REST communication with Ryu controller
│   ├── ryu_client.py           # HTTP client with retry logic
│   ├── telemetry_parser.py     # JSON -> 12D state conversion
│   └── action_translator.py    # Action ID -> REST API call
├── environments/               # Gymnasium environments
│   └── sdn_env.py              # Live environment (requires Ryu)
├── agent/
│   ├── dqn_agent.py            # DQN implementation
│   └── replay_buffer.py        # Experience replay buffer
├── models/
│   └── q_network.py            # Q-network MLP
├── training/
│   └── trainer.py              # Optional training wrapper
└── utils/
    ├── reward.py               # Service-intent-aware reward
    └── config.py               # Config utilities
```

## State and Action Space

State is a normalized 12D vector:
[util_ran_agg, util_agg_core, util_core_sp1, util_core_sp2, util_sp1_lf1, util_sp2_lf1,
 latency, packet_loss, total_traffic_load, svc_urllc, svc_embb, svc_mmtc]

Actions are 5 discrete controls:

| ID | Name | Effect |
|----|------|--------|
| 0 | do_nothing | No changes |
| 1 | route_to_queue_0 | Service traffic to high-priority queue |
| 2 | route_to_queue_1 | Service traffic to low-priority queue |
| 3 | apply_rate_limit | Apply configured rate limit on target port |
| 4 | remove_rate_limit | Remove/restore default rate |

## Configuration

All settings are driven from prod.json.
Important sections:

- environment.ryu_controller: base URL, retries, timeout
- environment.network: DPID, link capacity, stabilization delay
- environment.monitoring: telemetry pair and caps
- agent: network dimensions and DQN hyperparameters
- training and evaluation: episode counts, checkpointing, baselines

## Integration Focus

Before long training runs, verify:

1. Ryu endpoint paths and payload contracts match ryu_client/action_translator.
2. switch_dpid and port names in prod.json match the real topology.
3. Traffic generation is active so rewards contain meaningful signal.
4. /network/reset exists or an equivalent reset path is available.
