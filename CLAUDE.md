# CLAUDE.md

This file provides guidance when working with code in this repository.

## Project Overview

Reinforcement Learning system for SDN traffic optimization.
A DQN agent learns routing/QoS actions by communicating with a real Ryu SDN controller through REST APIs.

## Development Commands

```bash
# Install dependencies
pip install -r requirements.txt

# One-time startup routing + baseline QoS setup
python setup_network.py --config prod.json

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
│   ├── telemetry_parser.py     # JSON -> 6D operational state conversion
│   └── action_translator.py    # Runtime action ID -> REST API call
├── network_setup/
│   └── network_initializer.py  # One-time routing + baseline QoS startup setup
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
    ├── reward.py               # Operational QoS reward for 6D state
    └── config.py               # Config utilities
```

## State and Action Space

State is a normalized 6D vector:
[latency, packet_loss, throughput, main_link_util, backup_link_util, failover_active]

Actions are 4 discrete runtime controls:

| ID | Name | Effect |
|----|------|--------|
| 0 | do_nothing | No changes |
| 1 | update_queue | Apply queue profile update on configured switch/port |
| 2 | failover | Move selected route(s) to backup path |
| 3 | reroute | Restore selected route(s) to main path |

Startup setup (routing addresses/routes/default gateways + baseline QoS rules/queues)
is executed separately via `setup_network.py` or through train/evaluate setup flags.

## Configuration

All settings are driven from prod.json.
Important sections:

- environment.ryu_controller: base URL, retries, timeout
- environment.network: DPID set and main/backup capacities
- environment.monitoring: telemetry pairs and normalization bands
- environment.startup_setup: one-time routing + baseline QoS orchestration
- environment.action_space: runtime optimization actions
- agent: network dimensions and DQN hyperparameters
- training and evaluation: episode counts, checkpointing, baselines, setup toggles

## Integration Focus

Before long training runs, verify:

1. Startup setup payloads match controller expectations for /v1.0/conf/switches and /router.
2. Runtime action payloads for update_queue/failover/reroute match current topology.
3. /links/utilization and /latency responses include fields used by telemetry_parser.
4. switch_dpid values and port names in prod.json match Mininet topology.
5. Traffic generation is active so rewards contain meaningful signal.

Optional reset behavior:
- By default, env reset does not call /network/reset.
- Enable environment.episode.call_network_reset_on_reset if reset endpoint is available.
