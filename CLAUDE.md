# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the **AI Layer** for a Closed-Loop Intent-Based Network system. It implements a DQN agent that optimizes SDN traffic routing and rate limiting decisions via a Ryu controller REST API.

**AI Layer Boundaries:**
- **Responsible for:** Telemetry parsing, state construction, RL decisions, action translation, training/evaluation
- **NOT responsible for:** Direct OpenFlow switch management, Ryu controller implementation, Mininet configuration, traffic generation

## Configuration

All parameters are in `prod.json`. Key sections:

| Section | Purpose |
|---------|---------|
| `environment.ryu_controller` | REST client settings (base_url, timeout, retries) |
| `environment.network` | Switch DPID, link capacity (1Mbps), stabilization delay (2s) |
| `environment.state_space` | 5 features, min-max normalization |
| `environment.action_space` | 5 discrete actions with REST mappings |
| `agent.neural_network` | MLP: 5→64→64→5, ReLU |
| `agent.hyperparameters` | LR=0.001, γ=0.99, ε-decay=0.995 |
| `agent.replay_buffer` | Capacity 10k, min 1k for training |
| `reward_function.components` | Utilization penalty, loss penalty (100x weight), balance bonus, congestion threshold |

## Development Order (Dependency Chain)

Implement modules in this order:

1. **Ryu REST Client** (`src/client/ryu_rest.py`) - Base HTTP client for all API calls
2. **Telemetry Parser** (`src/parsers/telemetry.py`) - Parse controller responses (pluggable design)
3. **Action Translator** (`src/actions/translator.py`) - Convert action IDs to REST calls
4. **Environment Wrapper** (`src/env/sdn_env.py`) - Gym-style interface for RL
5. **DQN Agent** (`src/agent/dqn.py`) - Neural network and training logic
6. **Training Loop** (`src/training/trainer.py`) - Episode management, logging
7. **Evaluation Module** (`src/eval/evaluator.py`) - Metrics and baseline comparison

## State Space

```python
state[0] = link1_utilization  # Port 1 utilization (normalized 0-1)
state[1] = link2_utilization  # Port 2 utilization
state[2] = link3_utilization  # Port 3 utilization
state[3] = packet_loss_rate   # Dropped / total packets
state[4] = total_traffic_load  # Sum of utilizations
```

## Action Space

| ID | Action | REST Call |
|----|--------|-----------|
| 0 | `do_nothing` | None |
| 1 | `route_to_queue_0` | `POST /qos/rules/{switch_id}` queue_id=0 |
| 2 | `route_to_queue_1` | `POST /qos/rules/{switch_id}` queue_id=1 |
| 3 | `apply_rate_limit` | `POST /qos/queue/{switch_id}` max_rate=500000 |
| 4 | `remove_rate_limit` | `POST /qos/queue/{switch_id}` max_rate=1000000 |

## Reward Function

```
R = R_util + R_loss + R_balance + R_congestion
R_util = -mean(utilizations)²           # weight 1.0
R_loss = -packet_loss_rate × 100        # weight 100.0
R_balance = 1 / (1 + std(utilizations)) # weight 1.0
R_congestion = -5 if any(u) > 0.8 else 0 # threshold 0.8
```
Clip final reward to [-10, 10].

## External Dependencies

| Dependency | Version | Purpose |
|------------|---------|---------|
| PyTorch | ≥2.0 | Neural network |
| gymnasium | ≥0.26 | RL environment interface |
| requests | ≥2.28 | Ryu REST client |
| TensorBoard | ≥2.12 | Training visualization |
| numpy | — | State vectors |

## External Systems

- **Ryu SDN Controller**: Must be running at `localhost:8080`
- **Mininet**: Must generate traffic for meaningful learning
- **OpenFlow Switch**: DPID `0000000000000001`, 3 ports

## API Contracts (Provisional)

Response formats depend on Ryu controller implementation. Parser module is pluggable to accommodate changes. See `PROJECT_SPECIFICATION.md` for detailed contracts.