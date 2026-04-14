import argparse
import json

from ai_layer.network_setup import NetworkInitializer


def parse_args():
    parser = argparse.ArgumentParser(description="Initialize routing and baseline QoS once")
    parser.add_argument("--config", default="prod.json", help="Path to config JSON")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue setup when a step fails",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned setup steps without calling APIs",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)

    if args.continue_on_error:
        config.setdefault("environment", {}).setdefault("startup_setup", {})[
            "continue_on_error"
        ] = True
    if args.dry_run:
        config.setdefault("debugging", {})["dry_run_setup"] = True

    initializer = NetworkInitializer(config)
    summary = initializer.initialize()

    print("Network setup summary:")
    print(json.dumps(summary.as_dict(), indent=2))


if __name__ == "__main__":
    main()
