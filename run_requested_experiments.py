from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from algorithms import dsi_c2ode, edapp, opmwade, rnd, tpde
from algorithms.common import (
    DEFAULT_INIT_DATA_ROOT,
    WORKSPACE_ROOT,
    RunResult,
    save_npz,
    set_trajectory_damping_mode,
    target_angle_init_data_filename,
)


ALGORITHMS = {
    "dsi-c2ode": dsi_c2ode.run,
    "edapp": edapp.run,
    "opmwade": opmwade.run,
    "rnd": rnd.run,
    "tpde": tpde.run,
}

GROUP1_ALGORITHMS = ["dsi-c2ode", "edapp", "opmwade", "rnd", "tpde"]
TARGET_ANGLES = [1.05, 1.57, 2.09]
TIP_MASSES_G = [9.78, 19.30, 40.47]
DAMPING_MODES = ["none", "fixed", "adaptive"]
ExperimentKey = tuple[str, float, float, str]


@dataclass(frozen=True)
class Experiment:
    group: str
    algorithm: str
    target_angle: float
    tip_mass_g: float
    damping_mode: str

    @property
    def tip_mass_kg(self) -> float:
        return self.tip_mass_g / 1000.0

    @property
    def name(self) -> str:
        angle = str(self.target_angle).replace(".", "_")
        mass = f"{self.tip_mass_g:.2f}".replace(".", "_")
        return f"{self.group}_{self.algorithm}_{self.damping_mode}_target_{angle}_mass_{mass}g"


def build_experiments() -> list[Experiment]:
    experiments: list[Experiment] = []

    for algorithm in GROUP1_ALGORITHMS:
        experiments.append(Experiment("group1_algorithm_comparison_adaptive_target_1_05_mass_9_78g", algorithm, 1.05, 9.78, "adaptive"))

    for target_angle in TARGET_ANGLES:
        experiments.append(Experiment("group2_adaptive_damping_three_angles_mass_9_78g", "opmwade", target_angle, 9.78, "adaptive"))

    for tip_mass_g in TIP_MASSES_G:
        experiments.append(Experiment("group3_adaptive_damping_target_1_05_three_masses", "opmwade", 1.05, tip_mass_g, "adaptive"))

    for damping_mode in DAMPING_MODES:
        experiments.append(Experiment("group4_opmwade_three_damping_modes_target_1_05_mass_9_78g", "opmwade", 1.05, 9.78, damping_mode))

    return experiments


def experiment_key(experiment: Experiment) -> ExperimentKey:
    return (
        experiment.algorithm,
        round(experiment.target_angle, 12),
        round(experiment.tip_mass_g, 12),
        experiment.damping_mode,
    )


def init_file_for_target(target_angle: float, init_data_root: Path, damping_mode: str) -> Path:
    return init_data_root / target_angle_init_data_filename(21, target_angle, damping_mode)


def runner_kwargs(experiment: Experiment, args: argparse.Namespace, init_file: Path) -> dict:
    kwargs = {
        "evals_range": [21],
        "repeat_num": args.repeat,
        "seed": args.seed,
        "max_nfes": args.max_nfes,
        "save": args.save_algorithm_mat,
        "init_file": init_file,
        "tip_mass": experiment.tip_mass_kg,
    }
    if experiment.algorithm == "edapp":
        kwargs["map_type"] = args.map_type
    if experiment.algorithm == "rnd":
        kwargs["hessian_mode"] = args.rnd_hessian_mode
        kwargs["perturbation"] = None
    return kwargs


def result_row(experiment: Experiment, result: RunResult, init_file: Path, process_file: Path) -> dict[str, str | int | float]:
    return {
        "group": experiment.group,
        "experiment": experiment.name,
        "algorithm": experiment.algorithm,
        "result_algorithm": result.algorithm,
        "evals": result.evals,
        "target_angle_rad": experiment.target_angle,
        "tip_mass_g": experiment.tip_mass_g,
        "tip_mass_kg": experiment.tip_mass_kg,
        "damping_mode": experiment.damping_mode,
        "init_file": str(init_file),
        "best": result.best,
        "median": result.median,
        "mean": result.mean,
        "worst": result.worst,
        "std": result.std,
        "fearate": result.fearate,
        "elapsed_time": result.elapsed_time,
        "process_file": str(process_file),
    }


def write_summary(rows: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "summary.csv"
    json_path = output_dir / "summary.json"

    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the requested problem 21 experiment batches.")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-nfes", type=int, default=None, help="Optional smoke-test budget override.")
    parser.add_argument("--init-data-root", type=Path, default=DEFAULT_INIT_DATA_ROOT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Folder for summary.csv, summary.json, and process arrays.",
    )
    parser.add_argument("--map-type", choices=["LD", "LS", "BD", "BS"], default="LD")
    parser.add_argument("--rnd-hessian-mode", choices=["diagonal", "full"], default="diagonal")
    parser.add_argument("--limit", type=int, default=None, help="Run only the first N scheduled experiments.")
    parser.add_argument("--dry-run", action="store_true", help="Print the scheduled experiments without running them.")
    parser.add_argument(
        "--save-algorithm-mat",
        action="store_true",
        help="Also let each algorithm write its default .mat result file. The central summary is always written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    experiments = build_experiments()
    if args.limit is not None:
        experiments = experiments[: args.limit]
    first_experiment_by_key: dict[ExperimentKey, int] = {}
    for index, experiment in enumerate(experiments, start=1):
        first_experiment_by_key.setdefault(experiment_key(experiment), index)

    missing = [
        init_file_for_target(experiment.target_angle, args.init_data_root, experiment.damping_mode)
        for experiment in experiments
        if not init_file_for_target(experiment.target_angle, args.init_data_root, experiment.damping_mode).exists()
    ]
    if missing:
        missing_list = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing required initialization files:\n{missing_list}")

    if args.dry_run:
        for index, experiment in enumerate(experiments, start=1):
            init_file = init_file_for_target(experiment.target_angle, args.init_data_root, experiment.damping_mode)
            first_index = first_experiment_by_key[experiment_key(experiment)]
            run_status = "run" if first_index == index else f"reuse #{first_index:02d}"
            print(
                f"{index:02d} [{run_status}] {experiment.group} | {experiment.algorithm} | "
                f"mode={experiment.damping_mode} | "
                f"target={experiment.target_angle:.2f} rad | "
                f"tip_mass={experiment.tip_mass_g:.2f} g ({experiment.tip_mass_kg:.5f} kg) | "
                f"init={init_file}"
            )
        print(f"Scheduled {len(experiments)} logical rows, {len(first_experiment_by_key)} unique optimization runs.")
        return

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = args.output_dir or (WORKSPACE_ROOT / "results" / "requested_experiments" / timestamp)
    process_dir = output_dir / "process"
    rows: list[dict] = []
    result_cache: dict[ExperimentKey, list[RunResult]] = {}
    first_completed_name: dict[ExperimentKey, str] = {}

    for index, experiment in enumerate(experiments, start=1):
        init_file = init_file_for_target(experiment.target_angle, args.init_data_root, experiment.damping_mode)
        key = experiment_key(experiment)
        if key in result_cache:
            results = result_cache[key]
            print(
                f"[{index}/{len(experiments)}] reuse {experiment.algorithm} "
                f"mode={experiment.damping_mode}, target={experiment.target_angle:.2f} rad, "
                f"tip_mass={experiment.tip_mass_g:.2f} g from {first_completed_name[key]}"
            )
        else:
            set_trajectory_damping_mode(experiment.damping_mode)
            print(
                f"[{index}/{len(experiments)}] run {experiment.algorithm} "
                f"mode={experiment.damping_mode}, target={experiment.target_angle:.2f} rad, "
                f"tip_mass={experiment.tip_mass_g:.2f} g"
            )
            runner = ALGORITHMS[experiment.algorithm]
            results = runner(**runner_kwargs(experiment, args, init_file))
            result_cache[key] = results
            first_completed_name[key] = experiment.name
        for result in results:
            process_file = process_dir / f"{experiment.name}.npz"
            save_npz(process_file, process=result.process)
            rows.append(result_row(experiment, result, init_file, process_file))
            print(
                f"    best={result.best:.10g}, mean={result.mean:.10g}, "
                f"fearate={result.fearate:.4g}, time={result.elapsed_time:.3f}s"
            )
        write_summary(rows, output_dir)

    print(f"Wrote summary to {output_dir}")


if __name__ == "__main__":
    main()
