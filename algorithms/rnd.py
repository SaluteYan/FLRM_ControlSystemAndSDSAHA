from __future__ import annotations

from typing import Callable, Iterable, Literal

import numpy as np

from .common import (
    DEFAULT_PROBLEM21_TIP_MASS,
    EvalState,
    ProgressReporter,
    RunResult,
    WORKSPACE_ROOT,
    best_individual_diagnostics,
    configure_problem_from_init_data,
    enforce_problem21_coupling,
    fearate_calculate,
    get_fitness_and_penalty,
    iteration_setting,
    load_initial_population,
    population_size,
    save_mat,
    set_initial_scope,
    set_problem21_tip_mass,
    summarize,
    test_problems,
    timed,
)


PYTHON_INIT_DIR = WORKSPACE_ROOT / "init_data"
RND_INITIAL_STEP_FRACTION = 0.22
RND_FINAL_STEP_FRACTION = 0.035
RND_MEMORY_DECAY = 0.88
RND_DIVERSITY_THRESHOLD = 0.08


def limit_range(z: np.ndarray, pop_max: np.ndarray, pop_min: np.ndarray, evals: int) -> np.ndarray:
    arr = np.asarray(z, dtype=float)
    clipped = np.minimum(np.maximum(arr, pop_min), pop_max)
    return enforce_problem21_coupling(clipped) if evals == 21 else clipped


def problem_project(
    z: np.ndarray,
    evals: int,
    pop_max: np.ndarray | None = None,
    pop_min: np.ndarray | None = None,
) -> np.ndarray:
    arr = np.asarray(z, dtype=float)
    if pop_max is not None and pop_min is not None:
        return limit_range(arr, pop_max, pop_min, evals)
    return enforce_problem21_coupling(arr) if evals == 21 else arr.copy()


Perturbation = None | float | np.ndarray | Callable[[int, np.ndarray], np.ndarray]
HessianMode = Literal["diagonal", "full"]


class EvaluationBudgetExhausted(RuntimeError):
    pass


def _evaluation_count(pop: np.ndarray) -> int:
    arr = np.asarray(pop)
    return 1 if arr.ndim <= 1 else int(arr.shape[0])


def _reserve_evaluations(state: EvalState | None, count: int) -> None:
    if state is None:
        return
    if state.nfes + count > state.nfes_max:
        raise EvaluationBudgetExhausted
    state.nfes += count


def evaluate_with_penalty(
    pop: np.ndarray,
    evals: int,
    penalty_factor: float,
    state: EvalState | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    _reserve_evaluations(state, _evaluation_count(pop))
    fitness, penalty, _ = get_fitness_and_penalty(pop, evals)
    return fitness, penalty, fitness + penalty * penalty_factor


def objective_for_evals(
    evals: int,
    penalty_factor: float = 1e8,
    use_penalty: bool = True,
    state: EvalState | None = None,
    pop_max: np.ndarray | None = None,
    pop_min: np.ndarray | None = None,
) -> Callable[[np.ndarray], float]:
    def objective(x: np.ndarray) -> float:
        xx = problem_project(x, evals, pop_max, pop_min)
        if use_penalty:
            return float(evaluate_with_penalty(xx, evals, penalty_factor, state=state)[2][0])
        _reserve_evaluations(state, 1)
        return float(test_problems(xx, evals)[0][0])

    return objective


def numerical_gradient(objective: Callable[[np.ndarray], float], z: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    grad = np.zeros(z.size)
    for i in range(z.size):
        z_forward = z.copy()
        z_backward = z.copy()
        z_forward[i] += epsilon
        z_backward[i] -= epsilon
        forward_value = np.nan_to_num(objective(z_forward), nan=1e100, posinf=1e100, neginf=-1e100)
        backward_value = np.nan_to_num(objective(z_backward), nan=1e100, posinf=1e100, neginf=-1e100)
        grad[i] = (forward_value - backward_value) / (2 * epsilon)
    return grad


def compute_hessian(
    z: np.ndarray,
    objective: Callable[[np.ndarray], float],
    epsilon: float = 1e-5,
    mode: HessianMode = "diagonal",
) -> np.ndarray:
    """Numerically approximate H(z)=d(grad psi(z))/dz from Eq. (2).

    The paper defines H as the Hessian. Full finite differences are expensive for
    high-dimensional expensive objectives, so the default keeps the diagonal
    Hessian terms. Use ``mode="full"`` for low-dimensional paper examples.
    """
    m = z.size
    hessian = np.zeros((m, m))
    if mode == "full":
        for j in range(m):
            z_forward = z.copy()
            z_backward = z.copy()
            z_forward[j] += epsilon
            z_backward[j] -= epsilon
            hessian[:, j] = (
                numerical_gradient(objective, z_forward, epsilon)
                - numerical_gradient(objective, z_backward, epsilon)
            ) / (2 * epsilon)
        return 0.5 * (hessian + hessian.T)

    center = np.nan_to_num(objective(z), nan=1e100, posinf=1e100, neginf=-1e100)
    for i in range(m):
        z_forward = z.copy()
        z_backward = z.copy()
        z_forward[i] += epsilon
        z_backward[i] -= epsilon
        forward_value = np.nan_to_num(objective(z_forward), nan=1e100, posinf=1e100, neginf=-1e100)
        backward_value = np.nan_to_num(objective(z_backward), nan=1e100, posinf=1e100, neginf=-1e100)
        hessian[i, i] = (forward_value - 2 * center + backward_value) / epsilon**2
    return hessian


def perturbation_vector(perturbation: Perturbation, iteration_index: int, z: np.ndarray) -> np.ndarray:
    if perturbation is None:
        return np.zeros_like(z)
    if callable(perturbation):
        return np.asarray(perturbation(iteration_index, z), dtype=float)
    if np.isscalar(perturbation):
        return np.full_like(z, float(perturbation))
    return np.asarray(perturbation, dtype=float)


def solve_rnd_velocity(hessian: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Solve Eq. (2), H(z) z_dot = rhs, with damping for singular Hessians."""
    matrix = hessian.copy()
    if not np.all(np.isfinite(matrix)) or np.linalg.norm(matrix, ord=np.inf) < 1e-14:
        matrix = np.eye(rhs.size)
    matrix = matrix + 1e-8 * np.eye(rhs.size)
    try:
        return np.linalg.solve(matrix, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(matrix, rhs, rcond=None)[0]


def compute_integral(z_trajectory: np.ndarray, k: int, dt: float, objective: Callable[[np.ndarray], float]) -> np.ndarray:
    integral = np.zeros(z_trajectory.shape[1])
    for i in range(max(k - 1, 0)):
        integral += numerical_gradient(objective, z_trajectory[i]) * dt
    return integral


def neural_dynamics_solver(
    objective: Callable[[np.ndarray], float],
    a: float,
    b: float,
    z_trajectory: np.ndarray,
    pop_index: np.ndarray,
    iteration_index: int,
    dt: float,
    pop_max: np.ndarray,
    pop_min: np.ndarray,
    evals: int,
    perturbation: Perturbation = None,
    hessian_mode: HessianMode = "diagonal",
) -> tuple[np.ndarray, np.ndarray, float]:
    z = pop_index.copy()
    grad_prev = numerical_gradient(objective, z)
    grad = numerical_gradient(objective, z)
    h_z = compute_hessian(z, objective, mode=hessian_mode)
    integral_term = compute_integral(z_trajectory, iteration_index, dt, objective)
    grad_t = (grad - grad_prev) / dt
    rhs = -a * grad - grad_t - b * integral_term + perturbation_vector(perturbation, iteration_index, z)
    z_dot = solve_rnd_velocity(h_z, rhs)
    z_new = limit_range(z + dt * z_dot, pop_max, pop_min, evals)
    if np.any(np.abs(z_new) > 1e6):
        dt = dt / 2
    if iteration_index >= z_trajectory.shape[0]:
        extra = np.zeros((iteration_index - z_trajectory.shape[0] + 1, z_trajectory.shape[1]))
        z_trajectory = np.vstack([z_trajectory, extra])
    z_trajectory[iteration_index] = z_new
    return z_new, z_trajectory, dt


def rnd_step(
    a: float,
    b: float,
    evals: int,
    z_trajectory: np.ndarray,
    pop_index: np.ndarray,
    iteration_index: int,
    dt: float,
    pop_max: np.ndarray,
    pop_min: np.ndarray,
    penalty_factor: float = 1e8,
    perturbation: Perturbation = None,
    hessian_mode: HessianMode = "diagonal",
    state: EvalState | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    objective = objective_for_evals(
        evals,
        penalty_factor=penalty_factor,
        use_penalty=True,
        state=state,
        pop_max=pop_max,
        pop_min=pop_min,
    )
    z_line, z_trajectory, dt = neural_dynamics_solver(
        objective,
        a,
        b,
        z_trajectory,
        pop_index,
        iteration_index,
        dt,
        pop_max,
        pop_min,
        evals,
        perturbation=perturbation,
        hessian_mode=hessian_mode,
    )
    return limit_range(z_line, pop_max, pop_min, evals), limit_range(z_trajectory, pop_max, pop_min, evals), dt


def update_pbest(
    z_line: np.ndarray,
    pop_pbest_index: np.ndarray,
    pbest_value: float,
    pbest_fitness: float,
    pbest_penalty: float,
    evals: int,
    penalty_factor: float,
    state: EvalState,
) -> tuple[np.ndarray, float, float, float]:
    z_fitness, z_penalty, z_values = evaluate_with_penalty(z_line, evals, penalty_factor, state=state)
    z_value = float(z_values[0])
    if solution_better(
        z_line,
        float(z_fitness[0]),
        float(z_penalty[0]),
        z_value,
        pop_pbest_index,
        pbest_fitness,
        pbest_penalty,
        pbest_value,
        evals,
    ):
        return z_line.copy(), z_value, float(z_fitness[0]), float(z_penalty[0])
    return pop_pbest_index, pbest_value, pbest_fitness, pbest_penalty


def solution_better(
    candidate: np.ndarray,
    candidate_fitness: float,
    candidate_penalty: float,
    candidate_value: float,
    incumbent: np.ndarray,
    incumbent_fitness: float,
    incumbent_penalty: float,
    incumbent_value: float,
    evals: int,
) -> bool:
    candidate_feasible = bool(fearate_calculate(candidate, evals, np.array([candidate_penalty]))[0] > 0)
    incumbent_feasible = bool(fearate_calculate(incumbent, evals, np.array([incumbent_penalty]))[0] > 0)
    if candidate_feasible and incumbent_feasible:
        return candidate_fitness < incumbent_fitness
    if candidate_feasible != incumbent_feasible:
        return candidate_feasible
    if not np.isclose(candidate_penalty, incumbent_penalty):
        return candidate_penalty < incumbent_penalty
    return candidate_value < incumbent_value


def candidate_better_mask(
    candidate_pop: np.ndarray,
    candidate_fitness: np.ndarray,
    candidate_penalty: np.ndarray,
    candidate_values: np.ndarray,
    incumbent_pop: np.ndarray,
    incumbent_fitness: np.ndarray,
    incumbent_penalty: np.ndarray,
    incumbent_values: np.ndarray,
    evals: int,
) -> np.ndarray:
    candidate_feasible = fearate_calculate(candidate_pop, evals, candidate_penalty) > 0
    incumbent_feasible = fearate_calculate(incumbent_pop, evals, incumbent_penalty) > 0
    better = np.zeros(candidate_fitness.size, dtype=bool)

    both_feasible = candidate_feasible & incumbent_feasible
    better[both_feasible] = candidate_fitness[both_feasible] < incumbent_fitness[both_feasible]

    candidate_only_feasible = candidate_feasible & ~incumbent_feasible
    better[candidate_only_feasible] = True

    both_infeasible = ~candidate_feasible & ~incumbent_feasible
    if np.any(both_infeasible):
        lower_penalty = candidate_penalty[both_infeasible] < incumbent_penalty[both_infeasible]
        tied_penalty = np.isclose(candidate_penalty[both_infeasible], incumbent_penalty[both_infeasible])
        lower_value = candidate_values[both_infeasible] < incumbent_values[both_infeasible]
        better[both_infeasible] = lower_penalty | (tied_penalty & lower_value)

    return better


def apply_candidate_update(
    candidate_pop: np.ndarray,
    candidate_fitness: np.ndarray,
    candidate_penalty: np.ndarray,
    candidate_values: np.ndarray,
    incumbent_pop: np.ndarray,
    incumbent_fitness: np.ndarray,
    incumbent_penalty: np.ndarray,
    incumbent_values: np.ndarray,
    evals: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    better = candidate_better_mask(
        candidate_pop,
        candidate_fitness,
        candidate_penalty,
        candidate_values,
        incumbent_pop,
        incumbent_fitness,
        incumbent_penalty,
        incumbent_values,
        evals,
    )
    if np.any(better):
        incumbent_pop[better] = candidate_pop[better]
        incumbent_fitness[better] = candidate_fitness[better]
        incumbent_penalty[better] = candidate_penalty[better]
        incumbent_values[better] = candidate_values[better]
    return incumbent_pop, incumbent_fitness, incumbent_penalty, incumbent_values, int(np.sum(better))


def best_index_by_feasibility(
    pop: np.ndarray,
    fitness: np.ndarray,
    penalty: np.ndarray,
    values: np.ndarray,
    evals: int,
) -> int:
    feasible = fearate_calculate(pop, evals, penalty) > 0
    if np.any(feasible):
        feasible_idx = np.flatnonzero(feasible)
        return int(feasible_idx[np.argmin(fitness[feasible_idx])])
    penalty_min = np.min(penalty)
    tied = np.flatnonzero(np.isclose(penalty, penalty_min))
    return int(tied[np.argmin(values[tied])])


def update_gbest(
    pop_pbest: np.ndarray,
    pbest_values: np.ndarray,
    pbest_fitness: np.ndarray,
    pbest_penalty: np.ndarray,
    pop_gbest: np.ndarray,
    gbest_value: float,
    gbest_fitness: float,
    gbest_penalty: float,
    evals: int,
) -> tuple[np.ndarray, float, float, float]:
    idx = best_index_by_feasibility(pop_pbest, pbest_fitness, pbest_penalty, pbest_values, evals)
    if solution_better(
        pop_pbest[idx],
        float(pbest_fitness[idx]),
        float(pbest_penalty[idx]),
        float(pbest_values[idx]),
        pop_gbest,
        gbest_fitness,
        gbest_penalty,
        gbest_value,
        evals,
    ):
        return pop_pbest[idx].copy(), float(pbest_values[idx]), float(pbest_fitness[idx]), float(pbest_penalty[idx])
    return pop_gbest, gbest_value, gbest_fitness, gbest_penalty


def normalized_diversity(
    pop: np.ndarray,
    reference: np.ndarray,
    pop_max: np.ndarray,
    pop_min: np.ndarray,
) -> float:
    span = np.maximum(pop_max - pop_min, 1e-12)
    distances = np.linalg.norm((pop - reference.reshape(1, -1)) / span, axis=1)
    return float(np.mean(distances) / np.sqrt(pop.shape[1]))


def cap_normalized_step(direction: np.ndarray, max_norm: float) -> np.ndarray:
    norms = np.linalg.norm(direction, axis=1, keepdims=True)
    scale = np.minimum(1.0, max_norm / np.maximum(norms, 1e-12))
    return direction * scale


def budget_rnd_population_step(
    pop: np.ndarray,
    pop_pbest: np.ndarray,
    pop_gbest: np.ndarray,
    rnd_integral: np.ndarray,
    iteration_index: int,
    pop_max: np.ndarray,
    pop_min: np.ndarray,
    evals: int,
    state: EvalState,
    perturbation: Perturbation = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    span = np.maximum(pop_max - pop_min, 1e-12)
    progress = min(max(state.nfes / max(state.nfes_max, 1), 0.0), 1.0)
    pbest_error = (pop_pbest - pop) / span
    gbest_error = (pop_gbest.reshape(1, -1) - pop) / span
    target_error = 0.65 * pbest_error + 0.35 * gbest_error
    rnd_integral = RND_MEMORY_DECAY * rnd_integral + target_error

    noise = np.random.normal(size=pop.shape)
    noise_norm = np.linalg.norm(noise, axis=1, keepdims=True)
    noise = noise / np.maximum(noise_norm, 1e-12)
    diversity = normalized_diversity(pop, pop_gbest, pop_max, pop_min)
    noise_weight = 0.16 * (1.0 - progress) + 0.035
    if diversity < RND_DIVERSITY_THRESHOLD:
        noise_weight *= 1.8

    perturb = np.zeros_like(pop)
    if perturbation is not None:
        for i, row in enumerate(pop):
            perturb[i] = perturbation_vector(perturbation, iteration_index, row)
        perturb = perturb / span

    direction = 0.70 * target_error + 0.18 * rnd_integral + noise_weight * noise + 0.05 * perturb
    step_fraction = RND_FINAL_STEP_FRACTION + (RND_INITIAL_STEP_FRACTION - RND_FINAL_STEP_FRACTION) * (1.0 - progress)
    direction = cap_normalized_step(direction, step_fraction * np.sqrt(pop.shape[1]))
    z_line = limit_range(pop + direction * span, pop_max, pop_min, evals)
    return z_line, rnd_integral, step_fraction


def generate_next_generation(
    pop: np.ndarray,
    z_line: np.ndarray,
    ita: float,
    ita_line: float,
    c1: float,
    c2: float,
    tar1: float,
    tar2: float,
    w: float,
    pop_pbest: np.ndarray,
    pop_gbest: np.ndarray,
    pop_max: np.ndarray,
    pop_min: np.ndarray,
    evals: int,
    state: EvalState,
    kappa: float = 5.0,
) -> np.ndarray:
    if ita > ita_line:
        r1 = np.random.rand(*pop.shape)
        r2 = np.random.rand(*pop.shape)
        pop_new = pop + w * (z_line - pop) + c1 * r1 * tar1 * (pop_pbest - pop) + c2 * r2 * tar2 * (pop_gbest - pop)
    else:
        g = 10000
        gamma = np.exp((-np.log(g)) * (1 - state.nfes / state.nfes_max) ** 2) + np.log(g)
        f = gamma * (-2.5 + 5 * np.random.rand(*pop.shape))
        c3 = 1 / np.sqrt(gamma) * np.exp(-(f / gamma) ** 2 / 2) * np.cos(kappa * f / gamma)
        pop_new = np.where(c3 > 0, pop + c3 * (pop_max - pop), pop + c3 * (pop - pop_min))
    return limit_range(pop_new, pop_max, pop_min, evals)


def process_record(
    process: np.ndarray,
    pop_gbest: np.ndarray,
    gbest_fitness: float,
    gbest_penalty: float,
    nfes: int,
    evals: int,
) -> np.ndarray:
    feasible = fearate_calculate(pop_gbest, evals, np.array([gbest_penalty])) > 0
    if feasible[0]:
        return np.vstack([process, [gbest_fitness, nfes]])
    return process


def evaluated_best_and_fearate(
    pop_pbest: np.ndarray,
    pbest_fitness: np.ndarray,
    pbest_penalty: np.ndarray,
    evals: int,
) -> tuple[float, float]:
    feasible = fearate_calculate(pop_pbest, evals, pbest_penalty) > 0
    fearate = float(np.sum(feasible) / pop_pbest.shape[0])
    feasible_fitness = pbest_fitness[feasible]
    return (float(np.min(feasible_fitness)) if feasible_fitness.size else float("inf")), fearate


def evaluated_best_individual(
    pop_pbest: np.ndarray,
    pbest_fitness: np.ndarray,
    pbest_penalty: np.ndarray,
    evals: int,
) -> tuple[np.ndarray, float, float]:
    feasible = fearate_calculate(pop_pbest, evals, pbest_penalty) > 0
    if np.any(feasible):
        feasible_idx = np.flatnonzero(feasible)
        idx = int(feasible_idx[np.argmin(pbest_fitness[feasible_idx])])
    else:
        idx = int(np.argmin(pbest_penalty))
    return pop_pbest[idx].copy(), float(pbest_fitness[idx]), float(pbest_penalty[idx])


def run(
    evals_range: Iterable[int] = (21,),
    repeat_num: int = 1,
    seed: int | None = None,
    max_nfes: int | None = None,
    save: bool = True,
    perturbation: Perturbation = None,
    hessian_mode: HessianMode = "diagonal",
    init_data_dir: str | None = None,
    init_file: str | None = None,
    tip_mass: float | None = None,
    progress_interval: int = 0,
    progress_label: str | None = None,
) -> list[RunResult]:
    if seed is not None:
        np.random.seed(seed)
    results: list[RunResult] = []

    for evals in evals_range:
        best_values: list[float] = []
        fearates: list[float] = []
        times: list[float] = []
        process = np.empty((0, 2))
        best_individuals: list[np.ndarray] = []
        best_individual_fitness: list[float] = []
        best_individual_penalty: list[float] = []

        for repeat_index in range(1, repeat_num + 1):
            start = timed()
            set_problem21_tip_mass(DEFAULT_PROBLEM21_TIP_MASS if tip_mass is None else tip_mass)
            configure_problem_from_init_data(evals, init_data_dir=init_data_dir or PYTHON_INIT_DIR, init_file=init_file)
            pop_max, pop_min, pop_dim = set_initial_scope(evals)
            state = EvalState(nfes=0, nfes_max=max_nfes or iteration_setting(evals, pop_dim))
            reporter = ProgressReporter("RND", evals, repeat_index, repeat_num, progress_interval, progress_label)
            np_g, _ = population_size(evals, pop_dim)
            np_g = max(1, min(np_g, state.nfes_max))
            state.np_g = np_g

            pop = load_initial_population(
                evals,
                np_g,
                init_data_dir=init_data_dir or PYTHON_INIT_DIR,
                init_file=init_file,
            )
            penalty_factor = 1e8
            pop_fitness, pop_penalty, pop_values = evaluate_with_penalty(pop, evals, penalty_factor, state=state)
            pop_pbest = pop.copy()
            pbest_fitness = pop_fitness.copy()
            pbest_penalty = pop_penalty.copy()
            pbest_values = pop_values.copy()
            idx = best_index_by_feasibility(pop_pbest, pbest_fitness, pbest_penalty, pbest_values, evals)
            pop_gbest = pop[idx].copy()
            gbest_value = float(pbest_values[idx])
            gbest_fitness = float(pbest_fitness[idx])
            gbest_penalty = float(pbest_penalty[idx])
            process = process_record(process, pop_gbest, gbest_fitness, gbest_penalty, state.nfes, evals)
            current_best = float(process[-1, 0]) if process.size else None
            _, current_fearate = evaluated_best_and_fearate(pop_pbest, pbest_fitness, pbest_penalty, evals)
            reporter.maybe(state, best=current_best, fearate=current_fearate, extra={"iter": 0, "np": state.np_g})

            c1 = 1.6
            c2 = 1.6
            tar1 = 1
            tar2 = 1
            w_initial = 0.72
            w_final = 0.35
            ita_line = RND_DIVERSITY_THRESHOLD
            iteration_index = 0
            rnd_integral = np.zeros_like(pop)

            while state.nfes < state.nfes_max:
                if state.nfes_max - state.nfes < np_g:
                    break
                iteration_index += 1
                z_line, rnd_integral, step_fraction = budget_rnd_population_step(
                    pop,
                    pop_pbest,
                    pop_gbest,
                    rnd_integral,
                    iteration_index,
                    pop_max,
                    pop_min,
                    evals,
                    state,
                    perturbation=perturbation,
                )
                try:
                    z_fitness, z_penalty, z_values = evaluate_with_penalty(z_line, evals, penalty_factor, state=state)
                except EvaluationBudgetExhausted:
                    break
                pop_pbest, pbest_fitness, pbest_penalty, pbest_values, local_pbest_updates = apply_candidate_update(
                    z_line,
                    z_fitness,
                    z_penalty,
                    z_values,
                    pop_pbest,
                    pbest_fitness,
                    pbest_penalty,
                    pbest_values,
                    evals,
                )
                pop, pop_fitness, pop_penalty, pop_values, local_pop_updates = apply_candidate_update(
                    z_line,
                    z_fitness,
                    z_penalty,
                    z_values,
                    pop,
                    pop_fitness,
                    pop_penalty,
                    pop_values,
                    evals,
                )
                pop_gbest, gbest_value, gbest_fitness, gbest_penalty = update_gbest(
                    pop_pbest,
                    pbest_values,
                    pbest_fitness,
                    pbest_penalty,
                    pop_gbest,
                    gbest_value,
                    gbest_fitness,
                    gbest_penalty,
                    evals,
                )
                process = process_record(process, pop_gbest, gbest_fitness, gbest_penalty, state.nfes, evals)
                current_best = float(process[-1, 0]) if process.size else None
                _, current_fearate = evaluated_best_and_fearate(pop_pbest, pbest_fitness, pbest_penalty, evals)
                reporter.maybe(
                    state,
                    best=current_best,
                    fearate=current_fearate,
                    extra={
                        "iter": iteration_index,
                        "np": state.np_g,
                        "phase": "rnd",
                        "step": f"{step_fraction:.3g}",
                        "pbest+": local_pbest_updates,
                        "pop+": local_pop_updates,
                    },
                )

                if state.nfes_max - state.nfes < np_g:
                    break

                ita = normalized_diversity(pop, pop_gbest, pop_max, pop_min)
                progress = min(max(state.nfes / max(state.nfes_max, 1), 0.0), 1.0)
                w = w_final + (w_initial - w_final) * (1.0 - progress)
                pop_candidate = generate_next_generation(
                    pop,
                    z_line,
                    ita,
                    ita_line,
                    c1,
                    c2,
                    tar1,
                    tar2,
                    w,
                    pop_pbest,
                    pop_gbest,
                    pop_max,
                    pop_min,
                    evals,
                    state,
                )
                try:
                    cand_fitness, cand_penalty, cand_values = evaluate_with_penalty(pop_candidate, evals, penalty_factor, state=state)
                except EvaluationBudgetExhausted:
                    break
                pop, pop_fitness, pop_penalty, pop_values, gen_pop_updates = apply_candidate_update(
                    pop_candidate,
                    cand_fitness,
                    cand_penalty,
                    cand_values,
                    pop,
                    pop_fitness,
                    pop_penalty,
                    pop_values,
                    evals,
                )
                pop_pbest, pbest_fitness, pbest_penalty, pbest_values, gen_pbest_updates = apply_candidate_update(
                    pop_candidate,
                    cand_fitness,
                    cand_penalty,
                    cand_values,
                    pop_pbest,
                    pbest_fitness,
                    pbest_penalty,
                    pbest_values,
                    evals,
                )
                pop_gbest, gbest_value, gbest_fitness, gbest_penalty = update_gbest(
                    pop_pbest,
                    pbest_values,
                    pbest_fitness,
                    pbest_penalty,
                    pop_gbest,
                    gbest_value,
                    gbest_fitness,
                    gbest_penalty,
                    evals,
                )
                process = process_record(process, pop_gbest, gbest_fitness, gbest_penalty, state.nfes, evals)
                current_best = float(process[-1, 0]) if process.size else None
                _, current_fearate = evaluated_best_and_fearate(pop_pbest, pbest_fitness, pbest_penalty, evals)
                reporter.maybe(
                    state,
                    best=current_best,
                    fearate=current_fearate,
                    extra={
                        "iter": iteration_index,
                        "np": state.np_g,
                        "phase": "pso" if ita > ita_line else "wavelet",
                        "div": f"{ita:.3g}",
                        "w": f"{w:.3g}",
                        "pbest+": gen_pbest_updates,
                        "pop+": gen_pop_updates,
                    },
                )

            best, fearate = evaluated_best_and_fearate(pop_pbest, pbest_fitness, pbest_penalty, evals)
            best_x, best_x_fit, best_x_pen = evaluated_best_individual(pop_pbest, pbest_fitness, pbest_penalty, evals)
            reporter.maybe(state, best=best, fearate=fearate, extra={"iter": iteration_index, "np": state.np_g}, force=True)
            best_individuals.append(best_x)
            best_individual_fitness.append(best_x_fit)
            best_individual_penalty.append(best_x_pen)
            best_values.append(best)
            fearates.append(fearate)
            times.append(timed() - start)

        summary = summarize(best_values, fearates, times)
        result = RunResult("RND", evals, *summary, process=process, diagnostics=best_individual_diagnostics(
            best_individuals,
            best_individual_fitness,
            best_individual_penalty,
        ))
        results.append(result)
        if save:
            row = np.zeros((21, 8))
            row[evals - 1, :] = np.array([evals, *summary])
            best_diag = result.diagnostics
            save_mat(
                WORKSPACE_ROOT / "results" / "rnd" / f"RND-P{evals}.mat",
                everyevalBestMediMeanWorstStdFearateTime=row,
                testProcessBestFitAndNfes=process,
                testBestIndividuals=np.vstack(best_individuals) if best_individuals else np.empty((0, 0)),
                testBestIndividualFitness=np.asarray(best_individual_fitness, dtype=float),
                testBestIndividualPenalty=np.asarray(best_individual_penalty, dtype=float),
                testSummaryBestIndividual=best_diag["summary_best_individual"].reshape(1, -1)
                if best_diag["summary_best_individual"].size
                else np.empty((0, 0)),
                testSummaryBestIndividualFitness=best_diag["summary_best_individual_fitness"],
                testSummaryBestIndividualPenalty=best_diag["summary_best_individual_penalty"],
                testFinalBestIndividual=best_diag["final_best_individual"].reshape(1, -1)
                if best_diag["final_best_individual"].size
                else np.empty((0, 0)),
            )
    return results


if __name__ == "__main__":
    run()
