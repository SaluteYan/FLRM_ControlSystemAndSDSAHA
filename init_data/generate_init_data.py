from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


CONVERTED_ROOT = Path(__file__).resolve().parents[1]
if str(CONVERTED_ROOT) not in sys.path:
    sys.path.insert(0, str(CONVERTED_ROOT))

from algorithms.common import (
    DEFAULT_INIT_DATA_ROOT,
    DEFAULT_PROBLEM21_TARGET_ANGLE,
    adaptive_damping_value,
    get_problem21_target_angle,
    get_trajectory_damping_mode,
    generate_population,
    normalize_trajectory_damping_mode,
    save_npz,
    set_problem21_target_angle,
    set_initial_scope,
    set_trajectory_damping_mode,
    set_use_leaky_dynamic_damping,
    target_angle_init_data_filename,
)


PROBLEM21_DAMPING_MODES = ("none", "fixed", "adaptive")


def parse_evals(raw: str) -> list[int]:
    values: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_raw, end_raw = part.split("-", 1)
            start = int(start_raw)
            end = int(end_raw)
            step = 1 if end >= start else -1
            values.extend(range(start, end + step, step))
        else:
            values.append(int(part))
    return values


def default_population_size(evals: int, pop_dim: int) -> int:
    return 10 * pop_dim if evals == 21 else 18 * pop_dim


def problem21_scope_for_mode(target_angle: float, damping_mode: str) -> tuple[np.ndarray, np.ndarray, int]:
    old_target_angle = get_problem21_target_angle()
    old_damping_mode = get_trajectory_damping_mode()
    try:
        set_problem21_target_angle(target_angle)
        set_trajectory_damping_mode(damping_mode)
        pop_max, pop_min, pop_dim = set_initial_scope(21)
        return pop_max.copy(), pop_min.copy(), pop_dim
    finally:
        set_problem21_target_angle(old_target_angle)
        set_trajectory_damping_mode(old_damping_mode)


def problem21_mode_specs(target_angle: float) -> dict[str, tuple[np.ndarray, np.ndarray, int, int]]:
    specs = {}
    for mode in PROBLEM21_DAMPING_MODES:
        pop_max, pop_min, pop_dim = problem21_scope_for_mode(target_angle, mode)
        specs[mode] = (pop_max, pop_min, pop_dim, default_population_size(21, pop_dim))
    return specs


def problem21_output_path(
    output_dir: Path,
    target_angle: float,
    damping_mode: str,
    target_angle_was_explicit: bool,
    output_file: Path | None = None,
) -> Path:
    if output_file is not None:
        return output_file
    if not target_angle_was_explicit:
        return output_dir / "PrG21InitData.npz"
    return output_dir / target_angle_init_data_filename(21, target_angle, damping_mode)


def save_problem21_init_data(
    output_path: Path,
    pop: np.ndarray,
    pop_min: np.ndarray,
    pop_max: np.ndarray,
    pop_dim: int,
    target_angle: float,
    damping_mode: str,
) -> Path:
    save_npz(
        output_path,
        pop=pop,
        pop_min=pop_min,
        pop_max=pop_max,
        pop_dim=np.array([pop_dim]),
        evals=np.array([21]),
        target_angle=np.array([target_angle]),
        damping_mode=np.array([damping_mode]),
    )
    return output_path


def problem21_population_from_adaptive_master(
    master_pop: np.ndarray,
    damping_mode: str,
    pop_min: np.ndarray,
    pop_max: np.ndarray,
    pop_dim: int,
    rows: int,
    fixed_damping_error_scale: float | None = None,
) -> np.ndarray:
    mode = normalize_trajectory_damping_mode(damping_mode)
    master = np.asarray(master_pop, dtype=float)[:rows]
    if mode == "fixed":
        lambda_f = master[:, 11]
        c_d = master[:, 12]
        g_min = master[:, 13]
        g_max = master[:, 14]
        eta_e = master[:, 15]
        error_scale = get_problem21_target_angle() if fixed_damping_error_scale is None else fixed_damping_error_scale
        d_f = adaptive_damping_value(c_d, g_min, g_max, eta_e, error_scale)
        pop = np.column_stack([master[:, :11], lambda_f, d_f])
    else:
        pop = master[:, :pop_dim]
    return np.minimum(np.maximum(pop, pop_min), pop_max)


def generate_problem21_mode_init_data(
    output_dir: Path,
    target_angle: float,
    damping_mode: str,
    rows: int | None = None,
    output_file: Path | None = None,
    target_angle_was_explicit: bool = True,
) -> Path:
    mode = normalize_trajectory_damping_mode(damping_mode)
    specs = problem21_mode_specs(target_angle)
    pop_max, pop_min, pop_dim, _ = specs[mode]
    output_rows = rows if rows is not None else specs["adaptive"][-1]
    master_rows = max(output_rows, max(spec[-1] for spec in specs.values()))
    master_max, master_min, _, _ = specs["adaptive"]
    master_pop = generate_population(master_min, master_max, master_rows, 21)
    pop = problem21_population_from_adaptive_master(
        master_pop,
        mode,
        pop_min,
        pop_max,
        pop_dim,
        output_rows,
        fixed_damping_error_scale=target_angle,
    )
    output_path = problem21_output_path(output_dir, target_angle, mode, target_angle_was_explicit, output_file)
    return save_problem21_init_data(output_path, pop, pop_min, pop_max, pop_dim, target_angle, mode)


def generate_problem21_all_damping_init_data(
    output_dir: Path,
    target_angle: float,
    rows: int | None = None,
) -> list[Path]:
    specs = problem21_mode_specs(target_angle)
    output_rows_by_mode = {
        mode: rows if rows is not None else specs["adaptive"][-1]
        for mode in PROBLEM21_DAMPING_MODES
    }
    master_rows = max(max(output_rows_by_mode.values()), max(spec[-1] for spec in specs.values()))
    master_max, master_min, _, _ = specs["adaptive"]
    master_pop = generate_population(master_min, master_max, master_rows, 21)

    paths = []
    for mode in PROBLEM21_DAMPING_MODES:
        pop_max, pop_min, pop_dim, _ = specs[mode]
        output_rows = output_rows_by_mode[mode]
        pop = problem21_population_from_adaptive_master(
            master_pop,
            mode,
            pop_min,
            pop_max,
            pop_dim,
            output_rows,
            fixed_damping_error_scale=target_angle,
        )
        output_path = output_dir / target_angle_init_data_filename(21, target_angle, mode)
        paths.append(save_problem21_init_data(output_path, pop, pop_min, pop_max, pop_dim, target_angle, mode))
    return paths


def generate_init_data(
    evals: int,
    output_dir: Path,
    rows: int | None = None,
    target_angle: float | None = None,
    output_file: Path | None = None,
    filename_damping_mode: str | None = None,
) -> Path:
    resolved_target_angle = target_angle if target_angle is not None else DEFAULT_PROBLEM21_TARGET_ANGLE
    if evals == 21:
        return generate_problem21_mode_init_data(
            output_dir,
            resolved_target_angle,
            filename_damping_mode or get_trajectory_damping_mode(),
            rows=rows,
            output_file=output_file,
            target_angle_was_explicit=target_angle is not None,
        )
    pop_max, pop_min, pop_dim = set_initial_scope(evals)
    pop_size = rows if rows is not None else default_population_size(evals, pop_dim)
    pop = generate_population(pop_min, pop_max, pop_size, evals)
    if output_file is not None:
        output_path = output_file
    elif evals == 21 and target_angle is not None:
        output_path = output_dir / target_angle_init_data_filename(evals, resolved_target_angle, filename_damping_mode)
    else:
        output_path = output_dir / f"PrG{evals}InitData.npz"
    arrays = {
        "pop": pop,
        "pop_min": pop_min,
        "pop_max": pop_max,
        "pop_dim": np.array([pop_dim]),
        "evals": np.array([evals]),
    }
    save_npz(output_path, **arrays)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate shared initial population files by test problem.")
    parser.add_argument("--evals", default="21", help="Problem numbers, for example: 21, 2,3,4, or 2-21.")
    parser.add_argument("--output-dir", default=DEFAULT_INIT_DATA_ROOT, type=Path)
    parser.add_argument("--output-file", default=None, type=Path, help="Write one explicit .npz file. Only valid for one evals value.")
    parser.add_argument("--rows", type=int, default=None, help="Override the generated population size for each problem.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--target-angle", type=float, default=None, help="Problem 21 target angle sitab in radians.")
    parser.add_argument(
        "--use-leaky-dynamic-damping",
        action="store_true",
        help="Generate problem 21 initial populations with optimized leaky dynamic damping parameters.",
    )
    parser.add_argument(
        "--damping-mode",
        choices=["none", "fixed", "adaptive"],
        default=None,
        help="Problem 21 trajectory correction mode. Use fixed for group 2 initialization data.",
    )
    parser.add_argument(
        "--all-damping-modes",
        action="store_true",
        help="For problem 21, generate matched none/fixed/adaptive initialization files from one 16-D master population.",
    )
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)
    if args.damping_mode is not None and args.use_leaky_dynamic_damping and args.damping_mode != "adaptive":
        parser.error("--use-leaky-dynamic-damping is only compatible with --damping-mode adaptive.")
    if args.damping_mode is not None:
        set_trajectory_damping_mode(args.damping_mode)
    else:
        set_use_leaky_dynamic_damping(args.use_leaky_dynamic_damping)

    output_dir = args.output_dir
    evals_values = parse_evals(args.evals)
    if args.output_file is not None and len(evals_values) != 1:
        raise ValueError("--output-file can only be used with one evals value.")
    if args.output_file is not None and args.all_damping_modes:
        raise ValueError("--output-file cannot be used with --all-damping-modes.")

    for evals in evals_values:
        if evals == 21 and args.all_damping_modes:
            target_angle = args.target_angle if args.target_angle is not None else DEFAULT_PROBLEM21_TARGET_ANGLE
            paths = generate_problem21_all_damping_init_data(output_dir, target_angle, rows=args.rows)
            for path in paths:
                print(f"P{evals}: wrote {path}")
        else:
            path = generate_init_data(
                evals,
                output_dir,
                rows=args.rows,
                target_angle=args.target_angle if evals == 21 else None,
                output_file=args.output_file,
                filename_damping_mode=get_trajectory_damping_mode() if evals == 21 else None,
            )
            print(f"P{evals}: wrote {path}")


if __name__ == "__main__":
    main()
