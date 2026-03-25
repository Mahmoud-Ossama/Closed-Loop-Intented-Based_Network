# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Reinforcement Learning (RL) system for SDN (Software-Defined Network) traffic optimization. Implements a Deep Q-Network (DQN) agent that learns to optimize network traffic routing and rate limiting decisions by communicating with a Ryu SDN controller via REST API.

## Development Commands

```bash
# Install dependencies
pip install torch gymnasium requests tensorboard numpy

# Run integration test (requires mock_ryu_server or real Ryu controller running)
python clint_test.py

# Start mock Ryu server for offline testing (port 8080)
python -m ai_layer.network_interface.mock_ryu_server
```

## Architecture

```
ai_layer/
├── network_interface/          # REST communication with Ryu controller
│   ├── ryu_client.py           # HTTP client with retry logic
│   ├── telemetry_parser.py     # JSON → state vector conversion
│   ├── action_translator.py    # Action ID → REST API call
│   └── mock_ryu_server.py      # Fake server for testing
├── environments/              # Gymnasium environments
│   ├── sdn_env.py              # Live environment (requires Ryu)
│   └── mock_env.py             # Offline simulated environment
├── agent/                      # DQN agent (NOT IMPLEMENTED)
├── models/                     # Neural network (NOT IMPLEMENTED)
├── training/                   # Training loop (NOT IMPLEMENTED)
└── utils/
    ├── reward.py               # Multi-objective reward function
    └── config.py               # Config loader (NOT IMPLEMENTED)
```

## Implementation Status

| Module | Status |
|--------|--------|
| ryu_client, telemetry_parser, action_translator, mock_ryu_server | Implemented |
| sdn_env, mock_env, reward | Implemented |
| dqn_agent, replay_buffer, q_network, trainer, config loader | **NOT IMPLEMENTED** (stub files only) |

## Key Patterns

- **Config-driven**: All parameters loaded from `prod.json`
- **Gymnasium interface**: Environments implement `reset()`, `step()`, `render()`
- **Two modes**: LIVE (`SDNEnv` with Ryu) and OFFLINE (`MockSDNEnv` simulated)
- **Dataclasses**: `PortStatistics`, `FlowEntry`, `ActionResult` for structured data
- **Retry logic**: REST client retries on connection/timeout errors

## State/Action Space

**State (5D)**: `[link1_util, link2_util, link3_util, packet_loss_rate, total_traffic_load]`

**Actions (5 discrete)**:
| ID | Name | Effect |
|----|------|--------|
| 0 | do_nothing | No changes |
| 1 | route_to_queue_0 | UDP → high-priority queue |
| 2 | route_to_queue_1 | UDP → low-priority queue |
| 3 | apply_rate_limit | Limit port s1-eth1 to 500kbps |
| 4 | remove_rate_limit | Restore port s1-eth1 to 1Mbps |

## Configuration

All configuration in `prod.json`. Key sections:
- `environment.ryu_controller` - REST API connection (localhost:8080)
- `environment.network` - Switch DPID, link capacity, stabilization delay
- `agent.neural_network` - MLP architecture (5→64→64→5)
- `agent.hyperparameters` - LR, gamma, epsilon decay, batch size
- `training` - Episodes, warmup, checkpoints

## Key Files

| File | Purpose |
|------|---------|
| `prod.json` | Main configuration |
| `PROJECT_SPECIFICATION.md` | Detailed developer documentation |
| `README` | Module overview and data flow |
| `INTEGRATION_NOTES.md` | Notes for real Ryu integration |
| `clint_test.py` | Integration test script |

## Reward Function

Multi-objective: `R = R_util + R_loss + R_balance + R_congestion`
- `R_util = -mean(utilizations)²` - penalize high usage
- `R_loss = -packet_loss × 100` - heavy penalty for drops
- `R_balance = 1/(1+std(utilizations))` - reward balanced load
- `R_congestion = -5 if any link > 80%` - congestion penalty

Final reward clipped to [-10, 10].