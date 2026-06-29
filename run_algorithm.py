from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

import numpy as np

from algorithms import dsi_c2ode, edapp, opmwade, rnd, tpde
from algorithms.common import (
    DEFAULT_INIT_DATA_ROOT,
    DEFAULT_PROBLEM21_TIP_MASS,
    set_trajectory_damping_mode,
    target_angle_init_data_filename,
)


RUNNERS: dict[str, Callable] = {
    "dsi-c2ode": dsi_c2ode.run,
    "edapp": edapp.run,
    "opmwade": opmwade.run,
    "rnd": rnd.run,
    "tpde": tpde.run,
}

def parse_evals(raw: str) -> list[int]:
    values: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        values.append(int(part))
    return values


def parse_rnd_perturbation(raw: str):
    raw = raw.strip().lower()
    if raw in ("", "none", "zero", "0"):
        return None
    if raw.startswith("constant:"):
        value = float(raw.split(":", 1)[1])
        return lambda _k, z: np.full_like(z, value)
    if raw.startswith("uniform:"):
        parts = raw.split(":")
        if len(parts) != 3:
            raise ValueError("uniform perturbation must use uniform:min:max")
        low = float(parts[1])
        high = float(parts[2])
        return lambda _k, z: np.random.uniform(low, high, size=z.shape)
    raise ValueError("Use none, constant:value, or uniform:min:max for --rnd-perturbation.")


def target_angle_init_file(
    target_angle: float,
    init_data_root: str | None = None,
    damping_mode: str | None = None,
) -> Path:
    root = Path(init_data_root) if init_data_root else DEFAULT_INIT_DATA_ROOT
    return root / target_angle_init_data_filename(21, target_angle, damping_mode)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run converted MATLAB optimization algorithms.")
    parser.add_argument("--algorithm", choices=[*RUNNERS.keys(), "all"], default="all")
    parser.add_argument("--evals", default="21", help="Comma-separated problem numbers, for example: 21 or 2,3,4")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-nfes", type=int, default=None, help="Override nfesMax for quick smoke tests.")
    parser.add_argument(
        "--use-leaky-dynamic-damping",
        action="store_true",
        help="Enable problem 21 leaky dynamic damping and append its five optimized parameters.",
    )
    parser.add_argument(
        "--damping-mode",
        choices=["none", "fixed", "adaptive"],
        default=None,
        help="Problem 21 trajectory correction mode: none, fixed, or adaptive.",
    )
    parser.add_argument("--map-type", choices=["LD", "LS", "BD", "BS"], default="LD", help="EDA++ mapping mechanism.")
    parser.add_argument("--rnd-hessian-mode", choices=["diagonal", "full"], default="diagonal")
    parser.add_argument("--rnd-perturbation", default="none", help="RND only: none, constant:value, or uniform:min:max")
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=0,
        help="Print progress every N real objective evaluations. Use 0 to disable.",
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
    parser.add_argument(
        "--init-data-root",
        default=None,
        help="Folder with shared PrG{evals}InitData files. Defaults to converted_python_algorithms/init_data.",
    )
    parser.add_argument(
        "--init-file",
        default=None,
        help="Use one explicit .npz initial population file.",
    )
    parser.add_argument(
        "--target-angle",
        type=float,
        default=None,
        help="Problem 21 target angle. Selects PrG21InitData-target_{angle}.npz from --init-data-root.",
    )
    parser.add_argument(
        "--tip-mass",
        type=float,
        default=DEFAULT_PROBLEM21_TIP_MASS,
        help="Problem 21 flexible manipulator tip mass m.",
    )
    parser.add_argument("--no-save", action="store_true", help="Do not write .mat result files.")
    args = parser.parse_args()
    if args.damping_mode is not None and args.use_leaky_dynamic_damping and args.damping_mode != "adaptive":
        parser.error("--use-leaky-dynamic-damping is only compatible with --damping-mode adaptive.")
    damping_mode = args.damping_mode or ("adaptive" if args.use_leaky_dynamic_damping else "none")
    set_trajectory_damping_mode(damping_mode)
    if args.tip_mass <= 0:
        parser.error("--tip-mass must be positive.")
    if args.progress_interval < 0:
        parser.error("--progress-interval must be >= 0.")
    if args.dsi_max_surrogate_samples < 0:
        parser.error("--dsi-max-surrogate-samples must be >= 0.")
    if args.dsi_w_max < 1:
        parser.error("--dsi-w-max must be >= 1.")

    evals_values = parse_evals(args.evals)
    if args.target_angle is not None:
        if args.init_file:
            parser.error("--target-angle cannot be used together with --init-file.")
        if evals_values != [21]:
            parser.error("--target-angle can only be used with --evals 21.")
        selected_init_file = target_angle_init_file(args.target_angle, args.init_data_root, damping_mode)
        if not selected_init_file.exists():
            parser.error(f"Initial data for target angle {args.target_angle:g} not found: {selected_init_file}")
    else:
        selected_init_file = Path(args.init_file) if args.init_file else None

    selected = list(RUNNERS.keys()) if args.algorithm == "all" else [args.algorithm]
    for name in selected:
        extra = {"map_type": args.map_type} if name == "edapp" else {}
        if args.init_data_root and selected_init_file is None:
            extra["init_data_dir"] = Path(args.init_data_root)
        if selected_init_file is not None:
            extra["init_file"] = selected_init_file
        extra["tip_mass"] = args.tip_mass
        if name == "rnd":
            extra["hessian_mode"] = args.rnd_hessian_mode
            extra["perturbation"] = parse_rnd_perturbation(args.rnd_perturbation)
        if name == "dsi-c2ode":
            extra["max_surrogate_samples"] = args.dsi_max_surrogate_samples
            extra["w_max"] = args.dsi_w_max
        results = RUNNERS[name](
            evals_range=evals_values,
            repeat_num=args.repeat,
            seed=args.seed,
            max_nfes=args.max_nfes,
            save=not args.no_save,
            progress_interval=args.progress_interval,
            **extra,
        )
        for result in results:
            print(
                f"{result.algorithm} P{result.evals}: "
                f"best={result.best:.10g}, mean={result.mean:.10g}, "
                f"fearate={result.fearate:.4g}, time={result.elapsed_time:.3f}s"
            )


if __name__ == "__main__":
    main()
