from __future__ import annotations

from typing import Callable, Iterable, Literal

import numpy as np

from .common import (
    DEFAULT_PROBLEM21_TIP_MASS,
    EvalState,
    ProgressReporter,
    RunResult,
    WORKSPACE_ROOT,
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


def limit_range(z: np.ndarray, pop_max: np.ndarray, pop_min: np.ndarray, evals: int) -> np.ndarray:
    arr = np.asarray(z, dtype=float)
    clipped = np.minimum(np.maximum(arr, pop_min), pop_max)
    return enforce_problem21_coupling(clipped) if evals == 21 else clipped


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
) -> Callable[[np.ndarray], float]:
    def objective(x: np.ndarray) -> float:
        xx = np.asarray(x, dtype=float)
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
    objective = objective_for_evals(evals, penalty_factor=penalty_factor, use_penalty=True, state=state)
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
    if z_value < pbest_value:
        return z_line.copy(), z_value, float(z_fitness[0]), float(z_penalty[0])
    return pop_pbest_index, pbest_value, pbest_fitness, pbest_penalty


def update_gbest(
    pop_pbest: np.ndarray,
    pbest_values: np.ndarray,
    pbest_fitness: np.ndarray,
    pbest_penalty: np.ndarray,
    pop_gbest: np.ndarray,
    gbest_value: float,
    gbest_fitness: float,
    gbest_penalty: float,
) -> tuple[np.ndarray, float, float, float]:
    idx = int(np.argmin(pbest_values))
    if pbest_values[idx] < gbest_value:
        return pop_pbest[idx].copy(), float(pbest_values[idx]), float(pbest_fitness[idx]), float(pbest_penalty[idx])
    return pop_gbest, gbest_value, gbest_fitness, gbest_penalty


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
        f = gamma * (-2.5 + 5 * np.random.rand())
        c3 = 1 / np.sqrt(gamma) * np.exp(-(f / gamma) ** 2 / 2) * np.cos(kappa * f / gamma)
        if c3 > 0:
            pop_new = pop + c3 * (pop_max - pop)
        else:
            pop_new = pop + c3 * (pop - pop_min)
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
            pop_next = np.zeros_like(pop)
            penalty_factor = 1e8
            pbest_fitness, pbest_penalty, pbest_values = evaluate_with_penalty(pop, evals, penalty_factor, state=state)
            pop_pbest = pop.copy()
            idx = int(np.argmin(pbest_values))
            pop_gbest = pop[idx].copy()
            gbest_value = float(pbest_values[idx])
            gbest_fitness = float(pbest_fitness[idx])
            gbest_penalty = float(pbest_penalty[idx])
            process = process_record(process, pop_gbest, gbest_fitness, gbest_penalty, state.nfes, evals)
            current_best = float(process[-1, 0]) if process.size else None
            reporter.maybe(state, best=current_best, extra={"iter": 0, "np": state.np_g})

            c1 = 2
            c2 = 2
            tar1 = 1
            tar2 = 1
            w = 0.6
            a = 100
            b = 100
            dt = 0.01
            ita_line = 0.5
            iteration_index = 1
            trajectory_len = int(np.ceil(state.nfes_max / np_g)) + 2
            z_trajectory = np.zeros((np_g, trajectory_len, pop_dim))
            z_trajectory[:, 0, :] = pop
            z_line = np.zeros_like(pop)
            budget_exhausted = False

            while state.nfes < state.nfes_max:
                iteration_index += 1
                dt_generated = np.full(np_g, dt)
                processed_count = 0
                for i in range(np_g):
                    if state.nfes >= state.nfes_max:
                        budget_exhausted = True
                        break
                    try:
                        z_line[i], z_trajectory[i], dt_generated[i] = rnd_step(
                            a,
                            b,
                            evals,
                            z_trajectory[i],
                            pop[i],
                            iteration_index,
                            dt,
                            pop_max,
                            pop_min,
                            penalty_factor=penalty_factor,
                            perturbation=perturbation,
                            hessian_mode=hessian_mode,
                            state=state,
                        )
                        (
                            pop_pbest[i],
                            pbest_values[i],
                            pbest_fitness[i],
                            pbest_penalty[i],
                        ) = update_pbest(
                            z_line[i],
                            pop_pbest[i],
                            pbest_values[i],
                            pbest_fitness[i],
                            pbest_penalty[i],
                            evals,
                            penalty_factor,
                            state,
                        )
                    except EvaluationBudgetExhausted:
                        budget_exhausted = True
                        break
                    processed_count += 1
                if processed_count == 0:
                    break
                dt = float(np.min(dt_generated[:processed_count]))
                pop_gbest, gbest_value, gbest_fitness, gbest_penalty = update_gbest(
                    pop_pbest,
                    pbest_values,
                    pbest_fitness,
                    pbest_penalty,
                    pop_gbest,
                    gbest_value,
                    gbest_fitness,
                    gbest_penalty,
                )
                process = process_record(process, pop_gbest, gbest_fitness, gbest_penalty, state.nfes, evals)
                current_best = float(process[-1, 0]) if process.size else None
                reporter.maybe(
                    state,
                    best=current_best,
                    extra={"iter": iteration_index, "np": state.np_g, "dt": f"{dt:.3g}"},
                )
                ita = float(np.linalg.norm(z_line - pop_gbest) / np_g)
                pop = generate_next_generation(pop, z_line, ita, ita_line, c1, c2, tar1, tar2, w, pop_pbest, pop_gbest, pop_max, pop_min, evals, state)
                if budget_exhausted:
                    break

            best, fearate = evaluated_best_and_fearate(pop_pbest, pbest_fitness, pbest_penalty, evals)
            reporter.maybe(state, best=best, fearate=fearate, extra={"iter": iteration_index, "np": state.np_g}, force=True)
            best_values.append(best)
            fearates.append(fearate)
            times.append(timed() - start)

        summary = summarize(best_values, fearates, times)
        result = RunResult("RND", evals, *summary, process=process)
        results.append(result)
        if save:
            row = np.zeros((21, 8))
            row[evals - 1, :] = np.array([evals, *summary])
            save_mat(WORKSPACE_ROOT / "results" / "rnd" / f"RND-P{evals}.mat", everyevalBestMediMeanWorstStdFearateTime=row, testProcessBestFitAndNfes=process)
    return results


if __name__ == "__main__":
    run()
