from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

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
INNER_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


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


@dataclass(frozen=True)
class RunnerOptions:
    repeat: int
    seed: int | None
    max_nfes: int | None
    save_algorithm_mat: bool
    map_type: str
    rnd_hessian_mode: str
    progress_interval: int
    dsi_max_surrogate_samples: int
    dsi_w_max: int


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


def runner_options_from_args(args: argparse.Namespace) -> RunnerOptions:
    return RunnerOptions(
        repeat=args.repeat,
        seed=args.seed,
        max_nfes=args.max_nfes,
        save_algorithm_mat=args.save_algorithm_mat,
        map_type=args.map_type,
        rnd_hessian_mode=args.rnd_hessian_mode,
        progress_interval=args.progress_interval,
        dsi_max_surrogate_samples=args.dsi_max_surrogate_samples,
        dsi_w_max=args.dsi_w_max,
    )


def progress_label(experiment: Experiment) -> str:
    return (
        f"mode={experiment.damping_mode}, "
        f"target={experiment.target_angle:.2f} rad, "
        f"tip_mass={experiment.tip_mass_g:.2f} g"
    )


def runner_kwargs(experiment: Experiment, options: RunnerOptions, init_file: Path) -> dict:
    kwargs = {
        "evals_range": [21],
        "repeat_num": options.repeat,
        "seed": options.seed,
        "max_nfes": options.max_nfes,
        "save": options.save_algorithm_mat,
        "init_file": init_file,
        "tip_mass": experiment.tip_mass_kg,
        "progress_interval": options.progress_interval,
        "progress_label": progress_label(experiment),
    }
    if experiment.algorithm == "edapp":
        kwargs["map_type"] = options.map_type
    if experiment.algorithm == "rnd":
        kwargs["hessian_mode"] = options.rnd_hessian_mode
        kwargs["perturbation"] = None
    if experiment.algorithm == "dsi-c2ode":
        kwargs["max_surrogate_samples"] = options.dsi_max_surrogate_samples
        kwargs["w_max"] = options.dsi_w_max
    return kwargs


def run_unique_experiment(
    experiment: Experiment,
    options: RunnerOptions,
    init_file: Path,
) -> tuple[ExperimentKey, list[RunResult]]:
    set_trajectory_damping_mode(experiment.damping_mode)
    runner = ALGORITHMS[experiment.algorithm]
    return experiment_key(experiment), runner(**runner_kwargs(experiment, options, init_file))


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


def result_process_arrays(result: RunResult) -> dict[str, object]:
    return {"process": result.process, **result.diagnostics}


def unique_experiments(experiments: list[Experiment]) -> list[tuple[ExperimentKey, int, Experiment]]:
    seen: set[ExperimentKey] = set()
    unique: list[tuple[ExperimentKey, int, Experiment]] = []
    for index, experiment in enumerate(experiments, start=1):
        key = experiment_key(experiment)
        if key in seen:
            continue
        seen.add(key)
        unique.append((key, index, experiment))
    return unique


def resolve_worker_count(requested_workers: int, unique_count: int) -> int:
    if requested_workers < 0:
        raise ValueError("--workers must be >= 0; use 0 for auto.")
    if unique_count == 0:
        return 1
    if requested_workers == 0:
        return max(1, min(os.cpu_count() or 1, unique_count))
    return max(1, min(requested_workers, unique_count))


def configure_inner_thread_env(inner_threads: int, workers: int) -> list[str]:
    if inner_threads < 0:
        raise ValueError("--inner-threads must be >= 0.")
    if workers <= 1 or inner_threads == 0:
        return []

    configured: list[str] = []
    for name in INNER_THREAD_ENV_VARS:
        if name not in os.environ:
            os.environ[name] = str(inner_threads)
            configured.append(name)
    return configured


def save_rows_from_cache(
    experiments: list[Experiment],
    result_cache: dict[ExperimentKey, list[RunResult]],
    init_data_root: Path,
    output_dir: Path,
    process_dir: Path,
) -> list[dict]:
    rows: list[dict] = []
    for experiment in experiments:
        key = experiment_key(experiment)
        init_file = init_file_for_target(experiment.target_angle, init_data_root, experiment.damping_mode)
        for result in result_cache[key]:
            process_file = process_dir / f"{experiment.name}.npz"
            save_npz(process_file, **result_process_arrays(result))
            rows.append(result_row(experiment, result, init_file, process_file))
    write_summary(rows, output_dir)
    return rows


def run_parallel_experiments(
    unique_runs: list[tuple[ExperimentKey, int, Experiment]],
    options: RunnerOptions,
    init_data_root: Path,
    workers: int,
) -> dict[ExperimentKey, list[RunResult]]:
    result_cache: dict[ExperimentKey, list[RunResult]] = {}
    print(f"Running {len(unique_runs)} unique optimization runs with {workers} worker processes.")

    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_to_run = {}
        for key, first_index, experiment in unique_runs:
            init_file = init_file_for_target(experiment.target_angle, init_data_root, experiment.damping_mode)
            future = executor.submit(run_unique_experiment, experiment, options, init_file)
            future_to_run[future] = (key, first_index, experiment)
            print(
                f"[submit #{first_index:02d}] {experiment.algorithm} "
                f"mode={experiment.damping_mode}, target={experiment.target_angle:.2f} rad, "
                f"tip_mass={experiment.tip_mass_g:.2f} g"
            )

        for completed, future in enumerate(as_completed(future_to_run), start=1):
            key, first_index, experiment = future_to_run[future]
            try:
                returned_key, results = future.result()
            except Exception as exc:
                raise RuntimeError(f"Experiment #{first_index:02d} failed: {experiment.name}") from exc
            if returned_key != key:
                raise RuntimeError(f"Experiment #{first_index:02d} returned an unexpected cache key.")
            result_cache[key] = results
            for result in results:
                print(
                    f"[done {completed}/{len(unique_runs)} | #{first_index:02d}] {experiment.algorithm} "
                    f"mode={experiment.damping_mode}, target={experiment.target_angle:.2f} rad, "
                    f"tip_mass={experiment.tip_mass_g:.2f} g | "
                    f"best={result.best:.10g}, mean={result.mean:.10g}, "
                    f"fearate={result.fearate:.4g}, time={result.elapsed_time:.3f}s"
                )

    return result_cache


def run_serial_experiments(
    experiments: list[Experiment],
    options: RunnerOptions,
    init_data_root: Path,
    output_dir: Path,
    process_dir: Path,
) -> list[dict]:
    rows: list[dict] = []
    result_cache: dict[ExperimentKey, list[RunResult]] = {}
    first_completed_name: dict[ExperimentKey, str] = {}

    for index, experiment in enumerate(experiments, start=1):
        init_file = init_file_for_target(experiment.target_angle, init_data_root, experiment.damping_mode)
        key = experiment_key(experiment)
        if key in result_cache:
            results = result_cache[key]
            print(
                f"[{index}/{len(experiments)}] reuse {experiment.algorithm} "
                f"mode={experiment.damping_mode}, target={experiment.target_angle:.2f} rad, "
                f"tip_mass={experiment.tip_mass_g:.2f} g from {first_completed_name[key]}"
            )
        else:
            print(
                f"[{index}/{len(experiments)}] run {experiment.algorithm} "
                f"mode={experiment.damping_mode}, target={experiment.target_angle:.2f} rad, "
                f"tip_mass={experiment.tip_mass_g:.2f} g"
            )
            results = run_unique_experiment(experiment, options, init_file)[1]
            result_cache[key] = results
            first_completed_name[key] = experiment.name
        for result in results:
            process_file = process_dir / f"{experiment.name}.npz"
            save_npz(process_file, **result_process_arrays(result))
            rows.append(result_row(experiment, result, init_file, process_file))
            print(
                f"    best={result.best:.10g}, mean={result.mean:.10g}, "
                f"fearate={result.fearate:.4g}, time={result.elapsed_time:.3f}s"
            )
        write_summary(rows, output_dir)

    return rows


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
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of process workers for unique runs. Use 0 for auto, capped by the number of unique runs.",
    )
    parser.add_argument(
        "--inner-threads",
        type=int,
        default=1,
        help="Per-worker BLAS/OpenMP threads when --workers > 1. Use 0 to keep the current environment.",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=500,
        help="Print progress every N function evaluations inside each optimization run. Use 0 to disable.",
    )
    parser.add_argument(
        "--dsi-max-surrogate-samples",
        type=int,
        default=dsi_c2ode.DEFAULT_MAX_SURROGATE_SAMPLES,
        help="DSI-C2oDE only: cap retained surrogate training samples. Use 0 to keep all samples.",
    )
    parser.add_argument(
        "--dsi-w-max",
        type=int,
        default=dsi_c2ode.DEFAULT_W_MAX,
        help="DSI-C2oDE only: maximum search intensity.",
    )
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
    if args.progress_interval < 0:
        raise SystemExit("--progress-interval must be >= 0.")
    if args.dsi_max_surrogate_samples < 0:
        raise SystemExit("--dsi-max-surrogate-samples must be >= 0.")
    if args.dsi_w_max < 1:
        raise SystemExit("--dsi-w-max must be >= 1.")
    unique_runs = unique_experiments(experiments)
    workers = resolve_worker_count(args.workers, len(unique_runs))
    if args.save_algorithm_mat and workers > 1:
        raise SystemExit("--save-algorithm-mat writes shared algorithm .mat files and is only safe with --workers 1.")
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
        print(f"Resolved worker processes: {workers}")
        return

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = args.output_dir or (WORKSPACE_ROOT / "results" / "requested_experiments" / timestamp)
    process_dir = output_dir / "process"

    options = runner_options_from_args(args)
    if workers == 1:
        rows = run_serial_experiments(experiments, options, args.init_data_root, output_dir, process_dir)
    else:
        configured_env = configure_inner_thread_env(args.inner_threads, workers)
        if configured_env:
            print(f"Set per-worker inner thread env to {args.inner_threads}: {', '.join(configured_env)}")
        result_cache = run_parallel_experiments(unique_runs, options, args.init_data_root, workers)
        rows = save_rows_from_cache(experiments, result_cache, args.init_data_root, output_dir, process_dir)

    print(f"Wrote {len(rows)} logical rows to {output_dir}")


if __name__ == "__main__":
    main()
