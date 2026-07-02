#!/usr/bin/env python3
"""
Generate scenario config JSON entries with controllable counters.

Rules implemented from train_av.json:
- data_id increments by 1 starting from a chosen base.
- For each scenario_id, route_id iterates from a chosen start to end (inclusive).
- Each route_id is repeated `entries_per_route` times before incrementing.
- When route_id passes the end value, scenario_id increments by 1 and route_id
  resets to the start value.
- parameters follows: model.scenepilot.{scenario_id}.{per_scenario_index:04d}.torch
  where per_scenario_index resets to 1 for each scenario_id.
"""

import argparse
import json
from pathlib import Path
from typing import List, Optional

DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[1]
    / "safebench/scenario/config/scenario_type/test_av.json"
)


def build_entries(
    start_scenario: int,
    end_scenario: int,
    start_route: int,
    end_route: int,
    entries_per_route: int,
    start_data_id: int,
    scenario_folder: str,
) -> List[dict]:
    if start_scenario > end_scenario:
        raise ValueError("start_scenario must be <= end_scenario")
    if start_route > end_route:
        raise ValueError("start_route must be <= end_route")
    if entries_per_route < 1:
        raise ValueError("entries_per_route must be >= 1")

    data_id = start_data_id
    entries: List[dict] = []

    for scenario_id in range(start_scenario, end_scenario + 1):
        per_scenario_idx = 81
        for route_id in range(start_route, end_route + 1):
            for _ in range(entries_per_route):
                entries.append(
                    {
                        "data_id": data_id,
                        "scenario_folder": scenario_folder,
                        "scenario_id": scenario_id,
                        "route_id": route_id,
                        "risk_level": None,
                        "parameters": f"model.scenepilot.{scenario_id}.{per_scenario_idx:04d}.torch",
                    }
                )
                data_id += 1
                per_scenario_idx += 1
    return entries


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate scenario JSON following train_av style counters."
    )
    parser.add_argument(
        "--start-scenario",
        type=int,
        default=1,
        help="First scenario_id (e.g., 5).",
    )
    parser.add_argument(
        "--end-scenario",
        type=int,
        default=8,
        help="Last scenario_id (inclusive).",
    )
    parser.add_argument(
        "--start-route",
        type=int,
        default=12,
        help="First route_id per scenario (default: 4).",
    )
    parser.add_argument(
        "--end-route",
        type=int,
        default=13,
        help="Last route_id per scenario (inclusive, default: 13).",
    )
    parser.add_argument(
        "--entries-per-route",
        type=int,
        default=10,
        help="How many records to emit before incrementing route_id (default: 10).",
    )
    parser.add_argument(
        "--start-data-id",
        type=int,
        default=0,
        help="Starting data_id value (default: 0).",
    )
    parser.add_argument(
        "--scenario-folder",
        type=str,
        default="adv_behavior_single",
        help="Value for scenario_folder field (default: adv_behavior_single).",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        type=Path,
        help="Path to write JSON (will be overwritten).",
    )
    return parser.parse_args(args)


def main() -> None:
    args = parse_args()
    entries = build_entries(
        start_scenario=args.start_scenario,
        end_scenario=args.end_scenario,
        start_route=args.start_route,
        end_route=args.end_route,
        entries_per_route=args.entries_per_route,
        start_data_id=args.start_data_id,
        scenario_folder=args.scenario_folder,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(entries, indent=4))
    print(
        f"Wrote {len(entries)} entries to {args.output} "
        f"(scenarios {args.start_scenario}-{args.end_scenario}, "
        f"routes {args.start_route}-{args.end_route})"
    )


if __name__ == "__main__":
    main()
