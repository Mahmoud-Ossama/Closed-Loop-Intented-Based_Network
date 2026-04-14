import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Run 3-seed reward-alignment experiment")
    parser.add_argument("--base-config", default="prod.json", help="Base config path")
    parser.add_argument("--episodes", type=int, default=300, help="Training episodes per seed")
    parser.add_argument("--eval-episodes", type=int, default=10, help="Evaluation episodes per policy")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44], help="Seeds to run")
    return parser.parse_args()


def run_cmd(cmd):
    result = subprocess.run(cmd, check=True)
    return result.returncode


def main():
    args = parse_args()
    root = Path(__file__).resolve().parent

    base_config_path = root / args.base_config
    with open(base_config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    # Apply experiment settings compatible with the operational reward components.
    cfg["reward_function"]["components"]["action_repeat_penalty"]["enabled"] = True
    cfg["reward_function"]["components"]["action_repeat_penalty"]["weight"] = 0.1
    cfg["reward_function"]["components"]["outcome_improvement_bonus"]["enabled"] = True
    cfg["reward_function"]["components"]["outcome_improvement_bonus"]["weight"] = 1.0
    cfg["reward_function"]["components"]["outcome_improvement_bonus"]["max_bonus"] = 0.2
    cfg["training"]["run_startup_setup"] = False
    cfg["evaluation"]["run_startup_setup"] = False
    cfg["training"]["num_episodes"] = int(args.episodes)
    cfg["evaluation"]["num_episodes"] = int(args.eval_episodes)

    exp_config_path = root / "configs" / "reward_alignment_exp.json"
    exp_config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(exp_config_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    seed_reports = []

    for seed in args.seeds:
        model_path = root / "models" / f"dqn_model_seed{seed}.pth"
        metrics_path = root / "logs" / f"evaluation_metrics_seed{seed}.json"

        train_cmd = [
            sys.executable,
            str(root / "train.py"),
            "--config",
            str(exp_config_path),
            "--seed",
            str(seed),
            "--model-path",
            str(model_path),
        ]
        run_cmd(train_cmd)

        eval_cmd = [
            sys.executable,
            str(root / "evaluate.py"),
            "--config",
            str(exp_config_path),
            "--seed",
            str(seed),
            "--model-path",
            str(model_path),
            "--metrics-path",
            str(metrics_path),
        ]
        run_cmd(eval_cmd)

        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)

        dqn = metrics["dqn"]
        baselines = {b["policy"]: b for b in metrics.get("baselines", [])}
        random_r = baselines.get("random", {}).get("average_reward", float("-inf"))
        do_nothing_r = baselines.get("do_nothing", {}).get("average_reward", float("-inf"))

        shares = dqn.get("reward_component_shares", {})
        throughput_share = float(shares.get("throughput_bonus", 0.0))
        outcome_share = float(shares.get("outcome_improvement_bonus", 0.0))
        latency_share = float(shares.get("latency_penalty", 0.0))
        loss_share = float(shares.get("packet_loss_penalty", 0.0))
        util_share = float(shares.get("utilization_penalty", 0.0))
        congestion_share = float(shares.get("congestion_penalty", 0.0))
        failover_share = float(shares.get("failover_penalty", 0.0))

        beats_baselines = dqn["average_reward"] > random_r and dqn["average_reward"] > do_nothing_r
        no_collapse = float(dqn.get("dominant_action_ratio", 1.0)) < 0.85
        no_jitter_only = float(dqn.get("action_switch_rate", 1.0)) < 0.75
        outcome_driven = (
            (latency_share + loss_share + util_share + congestion_share + failover_share + outcome_share)
            > throughput_share
        )

        seed_reports.append(
            {
                "seed": seed,
                "dqn_avg_reward": dqn["average_reward"],
                "random_avg_reward": random_r,
                "do_nothing_avg_reward": do_nothing_r,
                "action_counts": dqn.get("action_counts", {}),
                "action_switch_rate": dqn.get("action_switch_rate", 0.0),
                "dominant_action_ratio": dqn.get("dominant_action_ratio", 1.0),
                "avg_latency_proxy": dqn.get("avg_latency_proxy", 0.0),
                "avg_packet_loss_proxy": dqn.get("avg_packet_loss_proxy", 0.0),
                "congestion_hit_rate": dqn.get("congestion_hit_rate", 0.0),
                "reward_component_shares": shares,
                "checks": {
                    "beats_baselines": beats_baselines,
                    "no_action_collapse": no_collapse,
                    "no_jitter_only": no_jitter_only,
                    "outcome_driven_not_balance_dominated": outcome_driven,
                },
            }
        )

    overall = {
        "beats_baselines_all_seeds": all(r["checks"]["beats_baselines"] for r in seed_reports),
        "no_action_collapse_all_seeds": all(r["checks"]["no_action_collapse"] for r in seed_reports),
        "no_jitter_only_all_seeds": all(r["checks"]["no_jitter_only"] for r in seed_reports),
        "outcome_driven_all_seeds": all(
            r["checks"]["outcome_driven_not_balance_dominated"] for r in seed_reports
        ),
    }

    report = {
        "config": str(exp_config_path),
        "seeds": args.seeds,
        "training_episodes": args.episodes,
        "evaluation_episodes": args.eval_episodes,
        "per_seed": seed_reports,
        "overall": overall,
    }

    report_path = root / "logs" / "reward_alignment_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\nReward Alignment Report")
    print(json.dumps(overall, indent=2))
    print(f"Saved full report to {report_path}")


if __name__ == "__main__":
    main()
