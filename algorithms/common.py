from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Callable, Iterable

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.io import savemat


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INIT_DATA_ROOT = WORKSPACE_ROOT / "init_data"


@dataclass
class EvalState:
    nfes: int = 0
    nfes_max: int = 0
    np_g: int = 0


@dataclass
class RunResult:
    algorithm: str
    evals: int
    best: float
    median: float
    mean: float
    worst: float
    std: float
    fearate: float
    elapsed_time: float
    process: np.ndarray
    diagnostics: dict[str, np.ndarray] = field(default_factory=dict)


@dataclass(frozen=True)
class ProgressInfo:
    algorithm: str
    evals: int
    repeat_index: int
    repeat_num: int
    nfes: int
    nfes_max: int
    elapsed_time: float
    best: float | None = None
    fearate: float | None = None
    label: str | None = None
    extra: dict[str, int | float | str] | None = None


ProgressCallback = Callable[[ProgressInfo], None]


def _format_progress_value(value: float | None) -> str:
    if value is None:
        return "NA"
    if np.isfinite(value):
        return f"{value:.10g}"
    return str(value)


def print_progress(info: ProgressInfo) -> None:
    progress_pct = 100.0 * min(info.nfes, info.nfes_max) / max(info.nfes_max, 1)
    head = f"[progress] {info.algorithm} P{info.evals} repeat={info.repeat_index}/{info.repeat_num}"
    if info.label:
        head = f"{head} | {info.label}"

    parts = [
        head,
        f"nfes={info.nfes}/{info.nfes_max} ({progress_pct:.1f}%)",
        f"best={_format_progress_value(info.best)}",
    ]
    if info.fearate is not None:
        parts.append(f"fearate={info.fearate:.4g}")
    if info.extra:
        parts.extend(f"{key}={value}" for key, value in info.extra.items())
    parts.append(f"elapsed={info.elapsed_time:.1f}s")
    print(" | ".join(parts), flush=True)


class ProgressReporter:
    def __init__(
        self,
        algorithm: str,
        evals: int,
        repeat_index: int,
        repeat_num: int,
        interval: int = 0,
        label: str | None = None,
        callback: ProgressCallback | None = None,
    ) -> None:
        self.algorithm = algorithm
        self.evals = evals
        self.repeat_index = repeat_index
        self.repeat_num = repeat_num
        self.interval = max(0, int(interval or 0))
        self.label = label
        self.callback = callback or print_progress
        self.start_time = perf_counter()
        self.next_nfes = self.interval
        self.last_nfes = -1

    @property
    def enabled(self) -> bool:
        return self.interval > 0

    def maybe(
        self,
        state: EvalState,
        best: float | None = None,
        fearate: float | None = None,
        extra: dict[str, int | float | str] | None = None,
        force: bool = False,
    ) -> None:
        if not self.enabled:
            return

        nfes = int(state.nfes)
        nfes_max = int(state.nfes_max)
        if not force and nfes < self.next_nfes and nfes < nfes_max:
            return
        if nfes == self.last_nfes:
            return
        while self.next_nfes <= nfes:
            self.next_nfes += self.interval
        self.last_nfes = nfes
        self.callback(
            ProgressInfo(
                algorithm=self.algorithm,
                evals=self.evals,
                repeat_index=self.repeat_index,
                repeat_num=self.repeat_num,
                nfes=nfes,
                nfes_max=nfes_max,
                elapsed_time=perf_counter() - self.start_time,
                best=best,
                fearate=fearate,
                label=self.label,
                extra=extra,
            )
        )


TRAJECTORY_DAMPING_MODES = {"none", "fixed", "adaptive"}
TRAJECTORY_DAMPING_MODE = "none"
USE_LEAKY_DYNAMIC_DAMPING = False
FIXED_DAMPING_PARAM_MIN = np.array([5.0, 2.0], dtype=float)
FIXED_DAMPING_PARAM_MAX = np.array([25.0, 60.0], dtype=float)
DEFAULT_FIXED_DAMPING_PARAMS = 0.5 * (FIXED_DAMPING_PARAM_MIN + FIXED_DAMPING_PARAM_MAX)
LEAKY_DAMPING_PARAM_MIN = np.array([5.0, 2.0, 0.05, 0.70, 0.02], dtype=float)
LEAKY_DAMPING_PARAM_MAX = np.array([25.0, 60.0, 0.35, 1.00, 0.12], dtype=float)
DEFAULT_LEAKY_DAMPING_PARAMS = 0.5 * (LEAKY_DAMPING_PARAM_MIN + LEAKY_DAMPING_PARAM_MAX)
DEFAULT_PROBLEM21_TARGET_ANGLE = 1.05
PROBLEM21_TARGET_ANGLE = DEFAULT_PROBLEM21_TARGET_ANGLE
DEFAULT_PROBLEM21_TIP_MASS = 0.3398
PROBLEM21_TIP_MASS = DEFAULT_PROBLEM21_TIP_MASS


def normalize_trajectory_damping_mode(mode: str) -> str:
    value = str(mode).strip().lower()
    aliases = {
        "off": "none",
        "no": "none",
        "leaky": "adaptive",
        "dynamic": "adaptive",
        "leaky_dynamic": "adaptive",
    }
    value = aliases.get(value, value)
    if value not in TRAJECTORY_DAMPING_MODES:
        allowed = ", ".join(sorted(TRAJECTORY_DAMPING_MODES))
        raise ValueError(f"Unsupported trajectory damping mode {mode!r}; use one of: {allowed}")
    return value


def set_trajectory_damping_mode(mode: str) -> None:
    global TRAJECTORY_DAMPING_MODE, USE_LEAKY_DYNAMIC_DAMPING
    TRAJECTORY_DAMPING_MODE = normalize_trajectory_damping_mode(mode)
    USE_LEAKY_DYNAMIC_DAMPING = TRAJECTORY_DAMPING_MODE == "adaptive"


def get_trajectory_damping_mode() -> str:
    return TRAJECTORY_DAMPING_MODE


def set_use_leaky_dynamic_damping(enabled: bool) -> None:
    set_trajectory_damping_mode("adaptive" if enabled else "none")


def set_problem21_target_angle(target_angle: float) -> None:
    global PROBLEM21_TARGET_ANGLE
    PROBLEM21_TARGET_ANGLE = float(target_angle)


def get_problem21_target_angle() -> float:
    return PROBLEM21_TARGET_ANGLE


def set_problem21_tip_mass(tip_mass: float) -> None:
    global PROBLEM21_TIP_MASS
    value = float(tip_mass)
    if value <= 0:
        raise ValueError(f"Problem 21 tip mass must be positive, got {tip_mass}")
    PROBLEM21_TIP_MASS = value


def get_problem21_tip_mass() -> float:
    return PROBLEM21_TIP_MASS


def target_angle_label(target_angle: float) -> str:
    text = f"{float(target_angle):.12g}"
    return text.replace("-", "neg_").replace(".", "_")


def target_angle_init_data_filename(evals: int, target_angle: float, damping_mode: str | None = None) -> str:
    mode_suffix = ""
    if damping_mode is not None:
        mode_suffix = f"-{normalize_trajectory_damping_mode(damping_mode)}"
    return f"PrG{evals}InitData-target_{target_angle_label(target_angle)}{mode_suffix}.npz"


def as_2d(values: np.ndarray | Iterable[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 0:
        return arr.reshape(1, 1)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    return arr.copy()


def flatten_if_single(values: np.ndarray) -> np.ndarray | float:
    arr = np.asarray(values)
    if arr.size == 1:
        return float(arr.reshape(-1)[0])
    return arr


def matlab_round(value: float) -> int:
    return int(np.floor(value + 0.5))


def enforce_problem21_coupling(pop: np.ndarray | Iterable[float]) -> np.ndarray:
    arr = np.asarray(pop, dtype=float).copy()
    if arr.shape[-1] >= 10:
        arr[..., 9] = 1.0 - arr[..., 8]
    return arr


def _match_problem_dimension(pop: np.ndarray, evals: int) -> np.ndarray:
    pop_max, pop_min, pop_dim = set_initial_scope(evals)
    arr = np.asarray(pop, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"Initial population must be a 2-D matrix, got shape {arr.shape}")

    if arr.shape[1] < pop_dim:
        missing_min = pop_min[arr.shape[1] :]
        missing_max = pop_max[arr.shape[1] :]
        supplement = np.random.uniform(missing_min, missing_max, size=(arr.shape[0], pop_dim - arr.shape[1]))
        arr = np.hstack([arr, supplement])
    elif arr.shape[1] > pop_dim:
        arr = arr[:, :pop_dim]

    arr = np.minimum(np.maximum(arr, pop_min), pop_max)
    return enforce_problem21_coupling(arr) if evals == 21 else arr


def set_initial_scope(evals: int) -> tuple[np.ndarray, np.ndarray, int]:
    if evals == 2:
        min_v = np.zeros(10)
        max_v = 10 * np.ones(10)
    elif evals == 3:
        min_v = np.zeros(10)
        max_v = np.ones(10)
    elif evals == 4:
        min_v = np.array([78, 33, 27, 27, 27], dtype=float)
        max_v = np.array([102, 45, 45, 45, 45], dtype=float)
    elif evals == 5:
        min_v = np.array([0.125, 0.1, 0.1, 0.1], dtype=float)
        max_v = np.array([10, 10, 10, 10], dtype=float)
    elif evals == 6:
        min_v = np.array([0, 0, 10, 10], dtype=float)
        max_v = np.array([99, 99, 200, 200], dtype=float)
    elif evals == 7:
        min_v = np.array([0.05, 0.25, 2], dtype=float)
        max_v = np.array([2, 1.3, 15], dtype=float)
    elif evals == 8:
        min_v = np.array([0, 0, -0.55, -0.55], dtype=float)
        max_v = np.array([1200, 1200, 0.55, 0.55], dtype=float)
    elif evals == 9:
        min_v = np.array([13, 0], dtype=float)
        max_v = np.array([100, 100], dtype=float)
    elif evals == 10:
        min_v = -10 * np.ones(10)
        max_v = 10 * np.ones(10)
    elif evals == 11:
        min_v = np.array([0, 0], dtype=float)
        max_v = np.array([10, 10], dtype=float)
    elif evals == 12:
        min_v = -10 * np.ones(7)
        max_v = 10 * np.ones(7)
    elif evals == 13:
        min_v = 10 * np.array([10, 100, 100, 1, 1, 1, 1, 1], dtype=float)
        max_v = 1000 * np.array([10, 10, 10, 1, 1, 1, 1, 1], dtype=float)
    elif evals == 14:
        min_v = np.array([-1, -1], dtype=float)
        max_v = np.array([1, 1], dtype=float)
    elif evals == 15:
        min_v = np.zeros(3)
        max_v = 10 * np.ones(3)
    elif evals == 16:
        min_v = -np.array([2.3, 2.3, 3.2, 3.2, 3.2], dtype=float)
        max_v = np.array([2.3, 2.3, 3.2, 3.2, 3.2], dtype=float)
    elif evals == 17:
        min_v = np.array([2, 0.2], dtype=float)
        max_v = np.array([14, 0.8], dtype=float)
    elif evals == 18:
        min_v = np.array([10, 10, 0.9, 0.9], dtype=float)
        max_v = np.array([80, 50, 5, 5], dtype=float)
    elif evals == 19:
        min_v = 0.01 * np.ones(5)
        max_v = 100 * np.ones(5)
    elif evals == 20:
        sitab = 0.5
        sita0 = 0.0
        te = 5.0
        t_real = np.linspace(0, te, 12)
        sita_init = (sitab - sita0) * (
            6 * (t_real / te) ** 5
            - 15 * (t_real / te) ** 4
            + 10 * (t_real / te) ** 3
        ) + sita0
        sita_limit = sita_init[1:11]
        min_v = -sita_limit
        max_v = sita_limit
    elif evals == 21:
        sitab = get_problem21_target_angle()
        sita0 = 0.0
        te = 1.0
        t_real = np.linspace(0, te, 7)
        sita_init = (sitab - sita0) * (
            6 * (t_real / te) ** 5
            - 15 * (t_real / te) ** 4
            + 10 * (t_real / te) ** 3
        ) + sita0
        curve_max = 0.3 * sita_init[1:7]
        curve_min = -curve_max
        h_max = np.array([5, 0.5], dtype=float)
        h_min = np.array([0, 0], dtype=float)
        zv_max = np.array([1, 1, 1], dtype=float)
        zv_min = np.array([0, 0, 0], dtype=float)
        min_v = np.concatenate([curve_min, h_min, zv_min])
        max_v = np.concatenate([curve_max, h_max, zv_max])
        damping_mode = get_trajectory_damping_mode()
        if damping_mode == "fixed":
            min_v = np.concatenate([min_v, FIXED_DAMPING_PARAM_MIN])
            max_v = np.concatenate([max_v, FIXED_DAMPING_PARAM_MAX])
        elif damping_mode == "adaptive":
            min_v = np.concatenate([min_v, LEAKY_DAMPING_PARAM_MIN])
            max_v = np.concatenate([max_v, LEAKY_DAMPING_PARAM_MAX])
    else:
        raise ValueError(f"Unsupported evals problem number: {evals}")

    return max_v.astype(float), min_v.astype(float), int(min_v.size)


def iteration_setting(evals: int, pop_dim: int) -> int:
    return 360 * pop_dim if evals not in (20, 21) else 3000


def population_size(evals: int, pop_dim: int, dsi: bool = False) -> tuple[int, int | None]:
    if dsi:
        return 30, None
    if evals not in (20, 21):
        return 18 * pop_dim, 18
    return 10 * pop_dim, 10


def generate_population(pop_min: np.ndarray, pop_max: np.ndarray, n: int, evals: int) -> np.ndarray:
    pop = np.random.uniform(pop_min, pop_max, size=(int(n), pop_min.size))
    return enforce_problem21_coupling(pop) if evals == 21 else pop


def _workspace_path(path: str | Path) -> Path:
    value = Path(path)
    if value.is_absolute():
        return value
    if value.exists():
        return value.resolve()
    return WORKSPACE_ROOT / value


def _load_population_file(path: Path) -> np.ndarray:
    if path.suffix.lower() != ".npz":
        raise ValueError(f"Initial population files must use .npz format: {path}")
    data = np.load(path)
    try:
        if "pop" not in data:
            raise KeyError(f"{path} does not contain a 'pop' array")
        return np.asarray(data["pop"], dtype=float)
    finally:
        data.close()


def _target_angle_from_file(path: Path) -> float | None:
    if path.suffix.lower() != ".npz":
        raise ValueError(f"Initial population files must use .npz format: {path}")
    data = np.load(path)
    try:
        for key in ("target_angle", "sitab"):
            if key in data:
                return float(np.asarray(data[key], dtype=float).reshape(-1)[0])
    finally:
        data.close()
    return None


def _init_data_path(init_dir: Path, evals: int) -> Path | None:
    candidate = init_dir / f"PrG{evals}InitData.npz"
    return candidate if candidate.exists() else None


def _resolve_initial_population_path(
    evals: int,
    init_data_dir: str | Path | None = None,
    init_file: str | Path | None = None,
) -> Path | None:
    if init_file is not None:
        init_path = _workspace_path(init_file)
        if not init_path.exists():
            raise FileNotFoundError(f"Initial population file not found: {init_path}")
        return init_path
    init_dir = _workspace_path(init_data_dir or DEFAULT_INIT_DATA_ROOT)
    return _init_data_path(init_dir, evals)


def configure_problem_from_init_data(
    evals: int,
    init_data_dir: str | Path | None = None,
    init_file: str | Path | None = None,
) -> None:
    if evals != 21:
        return
    set_problem21_target_angle(DEFAULT_PROBLEM21_TARGET_ANGLE)
    init_path = _resolve_initial_population_path(evals, init_data_dir=init_data_dir, init_file=init_file)
    if init_path is None:
        return
    target_angle = _target_angle_from_file(init_path)
    if target_angle is not None:
        set_problem21_target_angle(target_angle)


def load_initial_population(
    evals: int,
    n: int | None = None,
    init_data_dir: str | Path | None = None,
    init_file: str | Path | None = None,
) -> np.ndarray:
    init_path = _resolve_initial_population_path(evals, init_data_dir=init_data_dir, init_file=init_file)

    if init_path is not None:
        configure_problem_from_init_data(evals, init_file=init_path)
        pop = _match_problem_dimension(_load_population_file(init_path), evals)
    else:
        pop_max, pop_min, pop_dim = set_initial_scope(evals)
        default_n = 10 * pop_dim if evals == 21 else 18 * pop_dim
        pop = generate_population(pop_min, pop_max, default_n, evals)

    if n is not None and n < pop.shape[0]:
        rows = np.random.choice(pop.shape[0], size=int(n), replace=False)
        pop = pop[rows, :]
    elif n is not None and n > pop.shape[0]:
        pop_max, pop_min, _ = set_initial_scope(evals)
        extra = generate_population(pop_min, pop_max, int(n) - pop.shape[0], evals)
        pop = np.vstack([pop, extra])
    return _match_problem_dimension(pop, evals)


def save_mat(path: str | Path, **arrays: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    savemat(path, arrays)


def save_npz(path: str | Path, **arrays: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def pr_g2_f(x: np.ndarray, d: int) -> float:
    j = np.arange(1, d + 1, dtype=float)
    sum_jx = np.sum(j * x[:d] ** 2)
    return -abs((np.sum(np.cos(x) ** 4) - 2 * np.prod(np.cos(x) ** 2)) / (np.sqrt(sum_jx) + 1e-8))


def pr_g3_f(x: np.ndarray, n: int) -> float:
    return -(np.sqrt(n) ** n) * np.prod(x[:n])


def pr_g4_f(x: np.ndarray) -> float:
    return 5.3578547 * x[2] ** 2 + 0.8356891 * x[0] * x[4] + 37.293239 * x[0] - 40792.141


def pr_w_f(x: np.ndarray) -> float:
    return 1.10471 * x[0] ** 2 * x[1] + 0.04811 * x[2] * x[3] * (14.0 + x[1])


def pr_p_f(x: np.ndarray) -> float:
    return (
        0.6224 * x[0] * x[2] * x[3]
        + 1.7781 * x[1] * x[2] ** 2
        + 3.1661 * x[0] ** 2 * x[3]
        + 19.84 * x[0] ** 2 * x[2]
    )


def pr_t_f(x: np.ndarray) -> float:
    return x[0] ** 2 * x[1] * (x[2] + 2)


def pr_g5_f(x: np.ndarray) -> float:
    return 3 * x[0] + 1e-6 * x[0] ** 3 + 2 * x[1] + 2e-6 / 3 * x[1] ** 3


def pr_g6_f(x: np.ndarray) -> float:
    return (x[0] - 10) ** 3 + (x[1] - 20) ** 3


def pr_g7_f(x: np.ndarray) -> float:
    return (
        x[0] ** 2
        + x[1] ** 2
        + x[0] * x[1]
        - 14 * x[0]
        - 16 * x[1]
        + (x[2] - 10) ** 2
        + 4 * (x[3] - 5) ** 2
        + (x[4] - 3) ** 2
        + 2 * (x[5] - 1) ** 2
        + 5 * x[6] ** 2
        + 7 * (x[7] - 11) ** 2
        + 2 * (x[8] - 10) ** 2
        + (x[9] - 7) ** 2
        + 45
    )


def pr_g8_f(x: np.ndarray) -> float:
    return -(np.sin(2 * np.pi * x[0]) ** 3 * np.sin(2 * np.pi * x[1])) / (x[0] ** 3 * (x[0] + x[1]))


def pr_g9_f(x: np.ndarray) -> float:
    return (
        (x[0] - 10) ** 2
        + 5 * (x[1] - 12) ** 2
        + x[2] ** 4
        + 3 * (x[3] - 11) ** 2
        + 10 * x[4] ** 6
        + 7 * x[5] ** 2
        + x[6] ** 4
        - 4 * x[5] * x[6]
        - 10 * x[5]
        - 8 * x[6]
    )


def pr_g10_f(x: np.ndarray) -> float:
    return x[0] + x[1] + x[2]


def pr_g11_f(x: np.ndarray) -> float:
    return x[0] ** 2 + (x[1] - 1) ** 2


def pr_g12_f(x: np.ndarray) -> float:
    return -(1 - 0.01 * ((x[0] - 5) ** 2 + (x[1] - 5) ** 2 + (x[2] - 5) ** 2))


def pr_g13_f(x: np.ndarray) -> float:
    value = np.exp(np.prod(x))
    return 1e8 if np.isinf(value) else float(value)


def pr_g14_f_self(x: np.ndarray) -> float:
    return 9.82 * x[0] * x[1] + 2 * x[0]


def pr_g15_f_self(x: np.ndarray) -> float:
    denom = x[2] * (x[0] - 2 * x[3]) ** 3 / 12 + x[1] * x[2] ** 3 / 6 + 2 * x[1] * x[3] * ((x[0] - x[1]) / 2) ** 2
    return 5000 / denom


def pr_g16_f_self(x: np.ndarray) -> float:
    return 0.0624 * np.sum(x[:5])


def pr_g17_f_self(_: np.ndarray) -> float:
    raise NotImplementedError("Problem 20 needs the MATLAB GetJParas helper, which is not present in the source tree.")


def test_problems(pop: np.ndarray | Iterable[float], evals: int) -> tuple[np.ndarray, np.ndarray]:
    arr = as_2d(pop)
    fitness = np.zeros(arr.shape[0])
    constraint_vio = np.zeros(arr.shape[0])

    for i, x in enumerate(arr):
        with np.errstate(all="ignore"):
            if evals == 1:
                value = 0.0
            elif evals == 2:
                value = pr_g2_f(x, arr.shape[1])
            elif evals == 3:
                value = pr_g3_f(x, arr.shape[1])
            elif evals == 4:
                value = pr_g4_f(x)
            elif evals == 5:
                value = pr_w_f(x)
            elif evals == 6:
                value = pr_p_f(x)
            elif evals == 7:
                value = pr_t_f(x)
            elif evals == 8:
                value = pr_g5_f(x)
            elif evals == 9:
                value = pr_g6_f(x)
            elif evals == 10:
                value = pr_g7_f(x)
            elif evals == 11:
                value = pr_g8_f(x)
            elif evals == 12:
                value = pr_g9_f(x)
            elif evals == 13:
                value = pr_g10_f(x)
            elif evals == 14:
                value = pr_g11_f(x)
            elif evals == 15:
                value = pr_g12_f(x)
            elif evals == 16:
                value = pr_g13_f(x)
            elif evals == 17:
                value = pr_g14_f_self(x)
            elif evals == 18:
                value = pr_g15_f_self(x)
            elif evals == 19:
                value = pr_g16_f_self(x)
            elif evals == 20:
                value = pr_g17_f_self(x)
            elif evals == 21:
                value, vio = pr_g18_f_self(x, damping_mode=get_trajectory_damping_mode())
                constraint_vio[i] = vio
            else:
                raise ValueError(f"Unsupported evals problem number: {evals}")
        fitness[i] = value
    return fitness, constraint_vio


def raw_constraints(x: np.ndarray, evals: int) -> np.ndarray:
    n = x.size
    if evals == 2:
        return np.array([-np.prod(x) + 0.75, np.sum(x) - 7.5 * n])
    if evals == 3:
        return np.array([abs(np.sum(x**2) - 1) - 1e-4])
    if evals == 4:
        u = 85.334407 + 0.0056858 * x[1] * x[4] + 0.0006262 * x[0] * x[3] - 0.0022053 * x[2] * x[4]
        v = 80.51249 + 0.0071317 * x[1] * x[4] + 0.0029955 * x[0] * x[1] + 0.0021813 * x[2] ** 2
        w = 9.300961 + 0.0047026 * x[2] * x[4] + 0.0012547 * x[0] * x[2] + 0.0019085 * x[2] * x[3]
        return np.array([-u, u - 92, -v + 90, v - 110, -w + 20, w - 25])
    if evals == 5:
        p = 6000
        ll = 14
        e = 30e6
        g = 12e6
        t_max = 13600
        s_max = 30000
        d_max = 0.25
        m = p * (ll + x[1] / 2)
        r = np.sqrt(0.25 * (x[1] ** 2 + (x[0] + x[2]) ** 2))
        j = 2 / np.sqrt(2) * x[0] * x[1] * (x[1] ** 2 / 12 + 0.25 * (x[0] + x[2]) ** 2)
        p_c = (4.013 * e / (6 * ll**2)) * x[2] * x[3] ** 3 * (1 - 0.25 * x[2] * np.sqrt(e / g) / ll)
        t1 = p / (np.sqrt(2) * x[0] * x[1])
        t2 = m * r / j
        t = np.sqrt(t1**2 + t1 * t2 * x[1] / r + t2**2)
        s = 6 * p * ll / (x[3] * x[2] ** 2)
        d = 4 * p * ll**3 / (e * x[3] * x[2] ** 3)
        return np.array([
            t - t_max,
            s - s_max,
            x[0] - x[3],
            0.10471 * x[0] ** 2 + 0.04811 * x[2] * x[3] * (14.0 + x[1]) - 5.0,
            d - d_max,
            p - p_c,
        ])
    if evals == 6:
        return np.array([
            -x[0] + 0.0193 * x[2],
            -x[1] + 0.00954 * x[2],
            -np.pi * x[2] ** 2 * x[3] - (4 / 3) * np.pi * x[2] ** 3 + 1296000,
            x[3] - 240,
        ])
    if evals == 7:
        return np.array([
            1 - (x[1] ** 3 * x[2]) / (71785 * x[0] ** 4),
            (4 * x[1] ** 2 - x[0] * x[1]) / (12566 * x[0] ** 3 * (x[1] - x[0])) + 1 / (5108 * x[0] ** 2) - 1,
            1 - 140.45 * x[0] / (x[2] * x[1] ** 2),
            (x[0] + x[1]) / 1.5 - 1,
        ])
    if evals == 8:
        return np.array([
            x[2] - x[3] - 0.55,
            x[3] - x[2] - 0.55,
            abs(1000 * (np.sin(-x[2] - 0.25) + np.sin(-x[3] - 0.25)) + 894.8 - x[0]) - 1e-4,
            abs(1000 * (np.sin(x[2] - 0.25) + np.sin(x[2] - x[3] - 0.25)) + 894.8 - x[1]) - 1e-4,
            abs(1000 * (np.sin(x[3] - 0.25) + np.sin(x[3] - x[2] - 0.25)) + 1294.8) - 1e-4,
        ])
    if evals == 9:
        return np.array([-(x[0] - 5) ** 2 - (x[1] - 5) ** 2 + 100, (x[0] - 6) ** 2 + (x[1] - 5) ** 2 - 82.81])
    if evals == 10:
        return np.array([
            4 * x[0] + 5 * x[1] - 3 * x[6] + 9 * x[7] - 105,
            10 * x[0] - 8 * x[1] - 17 * x[6] + 2 * x[7],
            -8 * x[0] + 2 * x[1] + 5 * x[8] - 2 * x[9] - 12,
            3 * (x[0] - 2) ** 2 + 4 * (x[1] - 3) ** 2 + 2 * x[2] ** 2 - 7 * x[3] - 120,
            5 * x[0] ** 2 + 8 * x[1] + (x[2] - 6) ** 2 - 2 * x[3] - 40,
            0.5 * (x[0] - 8) ** 2 + 2 * (x[1] - 4) ** 2 + 3 * x[4] ** 2 - x[5] - 30,
            x[0] ** 2 + 2 * (x[1] - 2) ** 2 - 2 * x[0] * x[1] + 14 * x[4] - 6 * x[5],
            -3 * x[0] + 6 * x[1] + 12 * (x[8] - 8) ** 2 - 7 * x[9],
        ])
    if evals == 11:
        return np.array([x[0] ** 2 - x[1] + 1, 1 - x[0] + (x[1] - 4) ** 2])
    if evals == 12:
        v1 = 2 * x[0] ** 2
        v2 = x[1] ** 2
        return np.array([
            v1 + 3 * v2**2 + x[2] + 4 * x[3] ** 2 + 5 * x[4] - 127,
            7 * x[0] + 3 * x[1] + 10 * x[2] ** 2 + x[3] - x[4] - 282,
            23 * x[0] + v2 + 6 * x[5] ** 2 - 8 * x[6] - 196,
            2 * v1 + v2 - 3 * x[0] * x[1] + 2 * x[2] ** 2 + 5 * x[5] - 11 * x[6],
        ])
    if evals == 13:
        return np.array([
            -1 + 0.0025 * (x[3] + x[5]),
            -1 + 0.0025 * (-x[3] + x[4] + x[6]),
            -1 + 0.01 * (-x[4] + x[7]),
            100 * x[0] - x[0] * x[5] + 833.33252 * x[3] - 83333.333,
            x[1] * x[3] - x[1] * x[6] - 1250 * x[3] + 1250 * x[4],
            x[2] * x[4] - x[2] * x[7] - 2500 * x[4] + 1250000,
        ])
    if evals == 14:
        return np.array([abs(x[1] - x[0] ** 2) - 1e-4])
    if evals == 15:
        points = np.arange(1, 10)
        best = np.inf
        for p in points:
            for q in points:
                for r in points:
                    best = min(best, (x[0] - p) ** 2 + (x[1] - q) ** 2 + (x[2] - r) ** 2 - 0.0625)
        return np.array([best])
    if evals == 16:
        return np.array([
            abs(np.sum(x**2) - 10) - 1e-4,
            abs(x[1] * x[2] - 5 * x[3] * x[4]) - 1e-4,
            abs(x[0] ** 3 + x[1] ** 3 + 1) - 1e-4,
        ])
    if evals == 17:
        p = 2500
        sigma = 500
        e = 0.85 * 10**6
        ll = 250
        return np.array([
            p / (np.pi * x[0] * x[1] * sigma) - 1,
            8 * p * ll**2 / (np.pi**3 * e * x[0] * x[1] * (x[0] ** 2 + x[1] ** 2)) - 1,
            2 / x[0] - 1,
            x[0] / 14 - 1,
            0.2 / x[1] - 1,
            x[1] / 0.8 - 1,
        ])
    if evals == 18:
        return np.array([
            2 * x[1] * x[2] + x[2] * (x[0] - 2 * x[3]) - 300,
            18 * x[0] * 10**4 / (x[2] * (x[0] - 2 * x[3]) ** 3 + 2 * x[1] * x[2] * (4 * x[3] ** 2 + 3 * x[0] * (x[0] - 2 * x[3]))),
        ])
    if evals == 19:
        return np.array([61 / x[0] ** 3 + 37 / x[1] ** 3 + 19 / x[2] ** 3 + 7 / x[3] ** 3 + 1 / x[4] ** 3 - 1])
    if evals in (20, 21):
        return np.array([0.0])
    raise ValueError(f"Unsupported evals problem number: {evals}")


def test_constraints(pop: np.ndarray | Iterable[float], evals: int, constraint_value: np.ndarray | Iterable[float] | float, real: bool = False) -> np.ndarray:
    arr = as_2d(pop)
    cons = np.asarray(constraint_value, dtype=float).reshape(-1)
    if cons.size == 1 and arr.shape[0] > 1:
        cons = np.repeat(cons, arr.shape[0])
    penalty = np.zeros(arr.shape[0])
    for i, x in enumerate(arr):
        if evals == 21:
            penalty[i] = cons[i]
        elif evals == 20:
            penalty[i] = 0.0
        else:
            vals = raw_constraints(x, evals)
            if evals == 15:
                penalty[i] = vals[0]
            elif real:
                penalty[i] = np.sum(vals)
            else:
                penalty[i] = np.sum(np.maximum(vals, 0))
    return penalty


def fearate_calculate(
    pop: np.ndarray | Iterable[float],
    evals: int,
    constraint_value: np.ndarray | Iterable[float] | float,
    variant: str = "standard",
) -> np.ndarray:
    arr = as_2d(pop)
    cons = np.asarray(constraint_value, dtype=float).reshape(-1)
    if cons.size == 1 and arr.shape[0] > 1:
        cons = np.repeat(cons, arr.shape[0])
    feasibility = np.zeros(arr.shape[0])

    for i, x in enumerate(arr):
        if evals == 21:
            feasibility[i] = 1.0 - cons[i]
        elif evals == 20:
            feasibility[i] = 1.0
        else:
            is_feasible = bool(np.all(raw_constraints(x, evals) <= 0))
            if variant == "opmwade":
                pop_max, pop_min, _ = set_initial_scope(evals)
                is_feasible = is_feasible and bool(np.all(x >= pop_min - 1e-12) and np.all(x <= pop_max + 1e-12))
            feasibility[i] = float(is_feasible)
    return feasibility


def get_fitness_and_penalty(
    pop: np.ndarray | Iterable[float],
    evals: int,
    state: EvalState | None = None,
    count: bool = False,
    opmwade_repair_inf: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = as_2d(pop)
    local_pop = arr.copy()
    fitness, constraint_value = test_problems(local_pop, evals)
    bad = ~np.isfinite(fitness)

    if opmwade_repair_inf and np.any(bad):
        good_pop = local_pop[~bad]
        for row in np.where(bad)[0]:
            r1 = np.random.rand()
            r2 = np.random.rand()
            if good_pop.shape[0] == 0:
                local_pop[row] = r1 * local_pop[row]
            else:
                local_pop[row] = r1 * local_pop[row] + r2 * good_pop[np.random.randint(good_pop.shape[0])]
            if evals == 21:
                local_pop[row] = enforce_problem21_coupling(local_pop[row])
            repaired_fitness, repaired_constraint = test_problems(local_pop[row], evals)
            fitness[row] = repaired_fitness[0]
            constraint_value[row] = repaired_constraint[0]
            if constraint_value[row] == 1:
                fitness[row] = 1e8
    else:
        fitness[bad] = 1e8

    penalty = test_constraints(local_pop, evals, constraint_value)
    if count and state is not None:
        state.nfes += local_pop.shape[0]
    return fitness, penalty, bad.astype(float)


def process_best_record(
    process: np.ndarray,
    pop: np.ndarray,
    fitness: np.ndarray,
    penalty: np.ndarray,
    nfes: int,
    evals: int,
    variant: str = "standard",
) -> np.ndarray:
    feasible = fearate_calculate(pop, evals, penalty, variant=variant) > 0
    values = np.asarray(fitness).reshape(-1)[feasible]
    if values.size == 0:
        return process
    row = np.array([[float(np.min(values)), float(nfes)]])
    return row if process.size == 0 else np.vstack([process, row])


def best_and_fearate(
    pop: np.ndarray,
    fitness: np.ndarray,
    penalty: np.ndarray,
    evals: int,
    variant: str = "standard",
) -> tuple[float, float]:
    feasible = fearate_calculate(pop, evals, penalty, variant=variant) > 0
    fearate = float(np.sum(feasible) / as_2d(pop).shape[0])
    values = np.asarray(fitness).reshape(-1)[feasible]
    return (float(np.min(values)) if values.size else float("inf")), fearate


def summarize(values: list[float], fearates: list[float], times: list[float]) -> tuple[float, float, float, float, float, float, float]:
    arr = np.asarray(values, dtype=float)
    return (
        float(np.min(arr)),
        float(np.median(arr)),
        float(np.mean(arr)),
        float(np.max(arr)),
        float(np.std(arr, ddof=0)),
        float(np.mean(fearates)),
        float(np.mean(times)),
    )


def timed() -> float:
    return perf_counter()


def optimizing_zv(sita_real: np.ndarray, a1: float, a2: float, t2: float, fhz: int = 10000) -> np.ndarray:
    y = np.zeros_like(sita_real)
    nn = sita_real.size
    u1 = a1 * sita_real
    u2 = a2 * sita_real
    idx_mat = matlab_round(t2 * fhz + 1)
    idx0 = max(0, min(nn, idx_mat - 1))
    y[:idx0] = u1[:idx0]
    tail_len = nn - idx0
    if tail_len > 0:
        y[idx0:] = u1[idx0:] + u2[:tail_len]
    return y


def trajectory_derivatives(trajectory: np.ndarray, step: float) -> tuple[np.ndarray, np.ndarray]:
    velocity = np.zeros_like(trajectory)
    velocity[1:trajectory.size - 1] = (trajectory[2:trajectory.size] - trajectory[1:trajectory.size - 1]) / step
    acceleration = np.zeros_like(trajectory)
    acceleration[1:trajectory.size - 1] = (velocity[2:trajectory.size] - velocity[1:trajectory.size - 1]) / step
    return velocity, acceleration


def fixed_damping_params_from_vector(x: np.ndarray) -> tuple[float, float]:
    if x.size >= 13:
        params = np.asarray(x[11:13], dtype=float)
    else:
        params = DEFAULT_FIXED_DAMPING_PARAMS.copy()
    return tuple(float(value) for value in params)


def valid_fixed_damping_params(params: tuple[float, float]) -> bool:
    lambda_f, d_f = params
    return bool(lambda_f > 0 and d_f > 0)


def leaky_damping_params_from_vector(x: np.ndarray) -> tuple[float, float, float, float, float]:
    if x.size >= 16:
        params = np.asarray(x[11:16], dtype=float)
    else:
        params = DEFAULT_LEAKY_DAMPING_PARAMS.copy()
    return tuple(float(value) for value in params)


def valid_leaky_damping_params(params: tuple[float, float, float, float, float]) -> bool:
    lambda_d, c_d, g_min, g_max, eta_e = params
    return bool(
        lambda_d > 0
        and c_d > 0
        and 0 < g_min < g_max <= 1
        and eta_e > 0
    )


def leaky_damping_gain(error: float, c_d: float, g_min: float, g_max: float, eta_e: float) -> tuple[float, float]:
    error2 = error * error
    eta2 = eta_e * eta_e
    denom = error2 + eta2
    g = g_min + (g_max - g_min) * error2 / denom
    dg_de = (g_max - g_min) * 2.0 * error * eta2 / (denom * denom)
    return c_d * g, c_d * dg_de


def integrate_model(
    flexible_robot_gbest: np.ndarray,
    h1: float,
    h2: float,
    a1: float,
    a2: float,
    t2: float,
    use_leaky_dynamic_damping: bool = False,
    damping_mode: str | None = None,
    fixed_damping_params: tuple[float, float] | None = None,
    leaky_damping_params: tuple[float, float, float, float, float] | None = None,
    target_angle: float | None = None,
    tip_mass: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Integrate the flexible manipulator model.

    The reference trajectory is selected by ``damping_mode``:
    ``none`` uses theta_s = phi_a; ``fixed`` uses a constant damping gain;
    ``adaptive`` uses the leaky dynamic damping gain optimized in problem 21.
    """
    if damping_mode is None:
        damping_mode = "adaptive" if use_leaky_dynamic_damping else get_trajectory_damping_mode()
    damping_mode = normalize_trajectory_damping_mode(damping_mode)

    ei = 0.09230
    rho = 0.2572
    m = get_problem21_tip_mass() if tip_mass is None else float(tip_mass)
    if m <= 0:
        raise ValueError(f"Problem 21 tip mass must be positive, got {m}")
    ih = 0.03399
    length = 0.2662
    t_curve = 1.0
    t_total = 3.0
    nx = 10
    dx = length / nx
    step = 1e-4
    nt = int(round(t_total / step))

    sitab = get_problem21_target_angle() if target_angle is None else float(target_angle)
    sita0 = 0.0
    te = t_curve
    sita_dist_total = np.concatenate([[0.0], np.asarray(flexible_robot_gbest, dtype=float)])
    sita_dist_total_last = np.zeros(6)
    t = np.linspace(0, te, 7)
    t_total_grid = np.linspace(te, t_total, 6)
    sita_init = (sitab - sita0) * (6 * (t / te) ** 5 - 15 * (t / te) ** 4 + 10 * (t / te) ** 3) + sita0
    sita_plus_dist = sita_init + sita_dist_total
    sita_plus_dist_last = sitab + sita_dist_total_last
    sita_actual = np.concatenate([sita_plus_dist, sita_plus_dist_last[1:6]])
    t_actual = np.concatenate([t, t_total_grid[1:6]])
    tt = np.linspace(t_actual[0], t_actual[-1], nt)
    sita_real = CubicSpline(t_actual, sita_actual)(tt)
    phi_a = optimizing_zv(sita_real, a1, a2, t2)
    dphi_a, ddphi_a = trajectory_derivatives(phi_a, step)
    if fixed_damping_params is None:
        fixed_damping_params = tuple(float(value) for value in DEFAULT_FIXED_DAMPING_PARAMS)
    lambda_f, d_f = fixed_damping_params
    if leaky_damping_params is None:
        leaky_damping_params = tuple(float(value) for value in DEFAULT_LEAKY_DAMPING_PARAMS)
    lambda_d, c_d, g_min, g_max, eta_e = leaky_damping_params

    phi_d = np.zeros(nt)
    thd = phi_a.copy()
    dthd = dphi_a.copy()
    ddthd = ddphi_a.copy()

    y = np.zeros((nx, nt))
    z = np.zeros((nx, nt))
    yxxxx = np.zeros((nx, nt))
    zxxxx = np.zeros((nx, nt))
    dy = np.zeros((nx, nt))
    yxxx_l = np.zeros(nt)
    th = np.zeros(nt)
    dth = np.zeros(nt)
    ddth = np.zeros(nt)
    tol = np.zeros(nt)
    f_1 = 0.0

    for j in range(2, nt):
        prev = j - 1
        prev2 = j - 2
        correction_dot_prev = 0.0
        if damping_mode == "fixed":
            y_l_prev = y[nx - 1, prev]
            correction_gain_prev = d_f
            correction_dot_prev = -lambda_f * phi_d[prev] + correction_gain_prev * y_l_prev
            dthd[prev] = dphi_a[prev] + correction_dot_prev
            ddthd[prev] = ddphi_a[prev] + lambda_f * lambda_f * phi_d[prev] - lambda_f * d_f * y_l_prev + d_f * dy[nx - 1, prev]
        elif damping_mode == "adaptive":
            y_l_prev = y[nx - 1, prev]
            error_prev = th[prev] - thd[prev]
            damping_prev, damping_derivative_prev = leaky_damping_gain(error_prev, c_d, g_min, g_max, eta_e)
            correction_dot_prev = -lambda_d * phi_d[prev] + damping_prev * y_l_prev
            dthd[prev] = dphi_a[prev] + correction_dot_prev
            error_dot_prev = dth[prev] - dthd[prev]
            damping_dot_prev = damping_derivative_prev * error_dot_prev
            ddthd[prev] = (
                ddphi_a[prev]
                + lambda_d * lambda_d * phi_d[prev]
                + (damping_dot_prev - lambda_d * damping_prev) * y_l_prev
                + damping_prev * dy[nx - 1, prev]
            )

        phi1 = th[prev] - thd[prev]
        phi1_t = dth[prev] - dthd[prev]
        phi2 = phi1_t + h1 * phi1
        yxx0 = (y[2, prev] - 2 * y[1, prev] + y[0, prev]) / dx**2
        tol[prev] = -h2 * phi2 - ei * yxx0 - ih * h1 * phi1_t - phi1 + ih * ddthd[prev]

        th[j] = 2 * th[prev] - th[prev2] + step**2 / ih * (tol[prev] + ei * yxx0)
        dth[j] = (th[j] - th[prev]) / step
        ddth[j] = (th[j] - 2 * th[prev] + th[prev2]) / step**2

        y[0, :] = 0
        y[1, :] = 0
        z[0, :] = 0
        z[1, :] = 0

        for ii in range(2, nx - 2):
            i_mat = ii + 1
            yxxxx_val = (y[ii + 2, prev] - 4 * y[ii + 1, prev] + 6 * y[ii, prev] - 4 * y[ii - 1, prev] + y[ii - 2, prev]) / dx**4
            y[ii, j] = step**2 * (-i_mat * dx * ddth[j] - (ei * yxxxx_val) / rho) + 2 * y[ii, prev] - y[ii, prev2]
            zxxxx[ii, prev] = yxxxx_val
            dy[ii, prev] = (y[ii, prev] - y[ii, prev2]) / step

        idx = nx - 2
        yxxxx[idx, prev] = (-2 * y[nx - 1, prev] + 5 * y[nx - 2, prev] - 4 * y[nx - 3, prev] + y[nx - 4, prev]) / dx**4
        y[idx, j] = step**2 * (-(nx - 1) * dx * ddth[j] - ei * yxxxx[idx, prev] / rho) + 2 * y[idx, prev] - y[idx, prev2]
        zxxxx[idx, prev] = yxxxx[idx, prev]
        dy[idx, j] = (y[idx, j] - y[idx, prev]) / step

        yxxx_l[prev] = (-y[nx - 1, prev] + 2 * y[nx - 2, prev] - y[nx - 3, prev]) / dx**3
        y[nx - 1, j] = step**2 * (-length * ddth[prev] + (ei * yxxx_l[prev] + f_1) / m) + 2 * y[nx - 1, prev] - y[nx - 1, prev2]
        dy[nx - 1, j] = (y[nx - 1, j] - y[nx - 1, prev]) / step

        if damping_mode in ("fixed", "adaptive"):
            phi_d[j] = phi_d[prev] + step * correction_dot_prev
            thd[j] = phi_a[j] + phi_d[j]

    return tol, dth, y[nx - 1, :], thd, th, step


def pr_g18_f_self(
    x: np.ndarray,
    use_leaky_dynamic_damping: bool | None = None,
    damping_mode: str | None = None,
) -> tuple[float, float]:
    flexible_robot_gbest = x[:6]
    h1 = x[6]
    h2 = x[7]
    a1 = x[8]
    a2 = x[9]
    t2 = x[10]
    if damping_mode is None:
        if use_leaky_dynamic_damping is None:
            damping_mode = get_trajectory_damping_mode()
        else:
            damping_mode = "adaptive" if use_leaky_dynamic_damping else "none"
    damping_mode = normalize_trajectory_damping_mode(damping_mode)

    fixed_damping_params = fixed_damping_params_from_vector(x) if damping_mode == "fixed" else None
    leaky_damping_params = leaky_damping_params_from_vector(x) if damping_mode == "adaptive" else None
    if fixed_damping_params is not None and not valid_fixed_damping_params(fixed_damping_params):
        return 1e8, 1.0
    if leaky_damping_params is not None and not valid_leaky_damping_params(leaky_damping_params):
        return 1e8, 1.0
    tol, dth, y_l, thd, th, dt = integrate_model(
        flexible_robot_gbest,
        h1,
        h2,
        a1,
        a2,
        t2,
        damping_mode=damping_mode,
        fixed_damping_params=fixed_damping_params,
        leaky_damping_params=leaky_damping_params,
        target_angle=get_problem21_target_angle(),
        tip_mass=get_problem21_tip_mass(),
    )
    track_error_max = np.max(np.abs(th - thd))
    value = 0.25 * np.sum(np.abs(tol * dth) * dt) + 0.5 * np.sum(y_l * y_l * dt) + 0.25 * track_error_max
    constraint_vio = 1.0 if (abs(th[-1] - thd[-1]) >= thd[-1] * 0.02) or np.isnan(th[-1]) else 0.0
    return float(value), constraint_vio
