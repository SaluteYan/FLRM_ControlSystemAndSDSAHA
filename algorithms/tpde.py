from __future__ import annotations

from typing import Iterable

import numpy as np

from .common import (
    DEFAULT_PROBLEM21_TIP_MASS,
    EvalState,
    RunResult,
    WORKSPACE_ROOT,
    best_and_fearate,
    configure_problem_from_init_data,
    enforce_problem21_coupling,
    fearate_calculate,
    get_fitness_and_penalty,
    iteration_setting,
    load_initial_population,
    population_size,
    raw_constraints,
    save_mat,
    set_initial_scope,
    set_problem21_tip_mass,
    summarize,
    timed,
)


PYTHON_INIT_DIR = WORKSPACE_ROOT / "init_data"


def get_pop_fitness_plus_penalty(pop: np.ndarray, evals: int, np_g: int, state: EvalState) -> tuple[np.ndarray, np.ndarray, float]:
    fitness, penalty, _ = get_fitness_and_penalty(pop, evals, state=state, count=True)
    rf = float(np.sum(penalty <= 0) / np_g)
    return fitness, penalty, rf


def update_gbest(
    pop: np.ndarray,
    fitness: np.ndarray,
    penalty: np.ndarray,
    pop_gbest: np.ndarray | None,
    pop_gbest_fitness: float | None,
    evals: int,
) -> tuple[np.ndarray | None, float | None]:
    feasible = fearate_calculate(pop, evals, penalty) > 0
    if not np.any(feasible):
        return pop_gbest, pop_gbest_fitness
    feasible_fitness = fitness[feasible]
    feasible_pop = pop[feasible]
    idx = int(np.argmin(feasible_fitness))
    if pop_gbest is None or pop_gbest_fitness is None or feasible_fitness[idx] < pop_gbest_fitness:
        return feasible_pop[idx].copy(), float(feasible_fitness[idx])
    return pop_gbest, pop_gbest_fitness


def process_record(process: np.ndarray, pop_gbest_fitness: float | None, nfes: int) -> np.ndarray:
    if pop_gbest_fitness is None:
        return process
    return np.vstack([process, [pop_gbest_fitness, nfes]])


def rand_choose(pop_index: np.ndarray, v: np.ndarray, cr: float) -> np.ndarray:
    pop_dim = pop_index.size
    out = np.zeros(pop_dim)
    for i in range(pop_dim):
        rand_dim = np.random.randint(pop_dim)
        if np.random.rand() < cr or rand_dim == i:
            out[i] = v[i]
        else:
            out[i] = pop_index[i]
    return out


def random_de_indices(np_g: int, current: int | None = None, count: int = 3) -> np.ndarray:
    excluded = {current} if current is not None else set()
    candidates = np.array([idx for idx in range(np_g) if idx not in excluded], dtype=int)
    replace = candidates.size < count
    return np.random.choice(candidates, size=count, replace=replace)


def de_search(
    pop: np.ndarray,
    fitness: np.ndarray,
    penalty: np.ndarray,
    np_g: int,
    pop_max: np.ndarray,
    pop_min: np.ndarray,
    evals: int,
    penalty_coeff: float | np.ndarray = 1.0,
) -> np.ndarray:
    pop_pro = np.zeros_like(pop)
    f_values = np.array([1.0, 0.8, 0.6])
    cr_values = np.array([1.0, 0.2, 0.1])
    coeff = np.asarray(penalty_coeff, dtype=float)
    if coeff.size == 1:
        coeff = np.full(np_g, float(coeff.reshape(-1)[0]))
    penalty_fitness = fitness + coeff[:np_g] * penalty
    pop_best = pop[int(np.argmin(penalty_fitness))]

    for i in range(np_g):
        f = float(np.random.choice(f_values))
        cr = float(np.random.choice(cr_values))
        if np.random.rand() < 0.5:
            r1, r2, r3 = random_de_indices(np_g, current=i)
            v = pop[r1] + f * (pop_best - pop[r1]) + f * (pop[r2] - pop[r3])
            candidate = rand_choose(pop[i], v, cr)
        else:
            r1, r2, r3 = random_de_indices(np_g, current=i)
            candidate = pop[i] + np.random.rand() * (pop[r1] - pop[i]) + f * (pop[r2] - pop[r3])
        candidate = np.minimum(np.maximum(candidate, pop_min), pop_max)
        pop_pro[i] = enforce_problem21_coupling(candidate) if evals == 21 else candidate
    return pop_pro


def penalty_coefficient(nfes_max: int, nfes: int, rf: float, pop_dim: int) -> float:
    ratio = nfes / nfes_max
    if ratio < 0.4:
        return 1.0
    if (ratio > 0.5 and rf == 0) or (ratio > 0.75 and rf == 1):
        return float(ratio * pop_dim**2)
    return max(0.001, 0.01 / pop_dim)


def feasible_ratio(penalty: np.ndarray) -> float:
    return float(np.sum(np.asarray(penalty) <= 0) / np.asarray(penalty).size)


def split_pop_by_base(
    pop: np.ndarray,
    pop_max: np.ndarray,
    pop_min: np.ndarray,
    np_base: int,
    np_g: int,
    pop_dim: int,
    fitness: np.ndarray,
    penalty: np.ndarray,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    group_size = int(np_base / 2)
    remaining_pop = pop.copy()
    remaining_fitness = fitness.copy()
    remaining_penalty = penalty.copy()
    groups: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    while remaining_pop.shape[0] >= group_size:
        reference = np.random.uniform(pop_min, pop_max)
        distances = np.sum((remaining_pop - reference) ** 2, axis=1)
        x_tilde = remaining_pop[int(np.argmin(distances))]
        distances = np.sum((remaining_pop - x_tilde) ** 2, axis=1)
        chosen = np.argsort(distances)[:group_size]
        groups.append((remaining_pop[chosen], remaining_fitness[chosen], remaining_penalty[chosen]))
        remaining_pop = np.delete(remaining_pop, chosen, axis=0)
        remaining_fitness = np.delete(remaining_fitness, chosen)
        remaining_penalty = np.delete(remaining_penalty, chosen)
    return groups


def sigma_penalty_coefficient(nfes_max: int, nfes: int, rf: float, pop_dim: int, split_index: int, split_count: int) -> float:
    epsilon = penalty_coefficient(nfes_max, nfes, rf, pop_dim)
    return float((split_index + 1) * epsilon / split_count)


def epm(
    pop: np.ndarray,
    fitness: np.ndarray,
    penalty: np.ndarray,
    np_g: int,
    iteration_index: int,
    rf: float,
    pop_max: np.ndarray,
    pop_min: np.ndarray,
    pop_dim: int,
    evals: int,
    np_base: int,
    state: EvalState,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if iteration_index % 2 == 0:
        epsilon = penalty_coefficient(state.nfes_max, state.nfes, rf, pop_dim)
        pop_pro = de_search(pop, fitness, penalty, np_g, pop_max, pop_min, evals, epsilon)
        pro_fitness, pro_penalty, _ = get_pop_fitness_plus_penalty(pop_pro, evals, np_g, state)
        better = pro_fitness + epsilon * pro_penalty < fitness + epsilon * penalty
        pop[better] = pop_pro[better]
        fitness[better] = pro_fitness[better]
        penalty[better] = pro_penalty[better]
    else:
        groups = split_pop_by_base(pop, pop_max, pop_min, np_base, np_g, pop_dim, fitness, penalty)
        group_size = int(np_base / 2)
        for split_index, (group_pop, group_fitness, group_penalty) in enumerate(groups):
            sigma = sigma_penalty_coefficient(state.nfes_max, state.nfes, rf, pop_dim, split_index, len(groups))
            group_pro = de_search(group_pop, group_fitness, group_penalty, group_size, pop_max, pop_min, evals, sigma)
            pro_fitness, pro_penalty, _ = get_pop_fitness_plus_penalty(group_pro, evals, np_g, state)
            better = pro_fitness + sigma * pro_penalty < group_fitness + sigma * group_penalty
            group_pop[better] = group_pro[better]
            group_fitness[better] = pro_fitness[better]
            group_penalty[better] = pro_penalty[better]
            start = split_index * group_size
            stop = start + group_size
            pop[start:stop] = group_pop
            fitness[start:stop] = group_fitness
            penalty[start:stop] = group_penalty
    return pop, fitness, penalty


def constraint_matrix(pop: np.ndarray, evals: int, fallback_penalty: np.ndarray | None = None) -> np.ndarray:
    rows: list[np.ndarray] = []
    for i, individual in enumerate(pop):
        if evals == 21 and fallback_penalty is not None:
            # The flexible-manipulator source exposes only a scalar endpoint
            # violation, so use it as a <= 0 surrogate for the IPM branch.
            rows.append(np.array([float(fallback_penalty[i])]))
        else:
            rows.append(np.asarray(raw_constraints(individual, evals), dtype=float).reshape(-1))
    max_len = max(row.size for row in rows)
    matrix = np.zeros((len(rows), max_len))
    for i, row in enumerate(rows):
        matrix[i, : row.size] = row
    return matrix


def barrier_values(parent_constraints: np.ndarray, offspring_constraints: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    all_constraints = np.vstack([parent_constraints, offspring_constraints])
    denom = np.maximum(np.abs(np.min(all_constraints, axis=0)), 1e-12)

    def barrier(values: np.ndarray) -> np.ndarray:
        relative = values / denom
        out = np.full(values.shape[0], np.inf)
        feasible = np.all(relative < 0, axis=1)
        out[feasible] = -np.sum(np.log(-relative[feasible]), axis=1)
        return out

    return barrier(parent_constraints), barrier(offspring_constraints)


def ipm(
    pop: np.ndarray,
    fitness: np.ndarray,
    penalty: np.ndarray,
    rw: float,
    np_g: int,
    pop_max: np.ndarray,
    pop_min: np.ndarray,
    pop_dim: int,
    evals: int,
    state: EvalState,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pop_pro = de_search(pop, fitness, penalty, np_g, pop_max, pop_min, evals, 1.0)
    pro_fitness, pro_penalty, _ = get_pop_fitness_plus_penalty(pop_pro, evals, np_g, state)

    parent_constraints = constraint_matrix(pop, evals, penalty)
    offspring_constraints = constraint_matrix(pop_pro, evals, pro_penalty)
    pop_bf, pro_bf = barrier_values(parent_constraints, offspring_constraints)

    both_feasible = (penalty <= 0) & (pro_penalty <= 0) & np.isfinite(pop_bf) & np.isfinite(pro_bf)
    better = np.zeros(np_g, dtype=bool)
    better[both_feasible] = pro_fitness[both_feasible] + rw * pro_bf[both_feasible] < fitness[both_feasible] + rw * pop_bf[both_feasible]

    mixed = ~both_feasible
    if np.any(mixed):
        coeff = penalty_coefficient(state.nfes_max, state.nfes, feasible_ratio(pro_penalty), pop_dim)
        better[mixed] = pro_fitness[mixed] + coeff * pro_penalty[mixed] < fitness[mixed] + coeff * penalty[mixed]

    pop[better] = pop_pro[better]
    fitness[better] = pro_fitness[better]
    penalty[better] = pro_penalty[better]
    return pop, fitness, penalty


def run(
    evals_range: Iterable[int] = (21,),
    repeat_num: int = 1,
    seed: int | None = None,
    max_nfes: int | None = None,
    save: bool = True,
    init_data_dir: str | None = None,
    init_file: str | None = None,
    tip_mass: float | None = None,
) -> list[RunResult]:
    if seed is not None:
        np.random.seed(seed)
    results: list[RunResult] = []

    for evals in evals_range:
        best_values: list[float] = []
        fearates: list[float] = []
        times: list[float] = []
        process = np.empty((0, 2))

        for _ in range(repeat_num):
            start = timed()
            set_problem21_tip_mass(DEFAULT_PROBLEM21_TIP_MASS if tip_mass is None else tip_mass)
            configure_problem_from_init_data(evals, init_data_dir=init_data_dir or PYTHON_INIT_DIR, init_file=init_file)
            pop_max, pop_min, pop_dim = set_initial_scope(evals)
            state = EvalState(nfes=0, nfes_max=max_nfes or iteration_setting(evals, pop_dim))
            np_g, np_base = population_size(evals, pop_dim)
            state.np_g = np_g

            pop = load_initial_population(
                evals,
                np_g,
                init_data_dir=init_data_dir or PYTHON_INIT_DIR,
                init_file=init_file,
            )
            fitness, penalty, rf = get_pop_fitness_plus_penalty(pop, evals, np_g, state)
            pop_gbest: np.ndarray | None = None
            pop_gbest_fitness: float | None = None
            pop_gbest, pop_gbest_fitness = update_gbest(pop, fitness, penalty, pop_gbest, pop_gbest_fitness, evals)
            process = process_record(process, pop_gbest_fitness, state.nfes)

            iteration_index = 0
            rw = 1e-6
            while state.nfes <= state.nfes_max:
                iteration_index += 1
                if rf != 1:
                    pop, fitness, penalty = epm(pop, fitness, penalty, np_g, iteration_index, rf, pop_max, pop_min, pop_dim, evals, int(np_base), state)
                else:
                    pop, fitness, penalty = ipm(pop, fitness, penalty, rw, np_g, pop_max, pop_min, pop_dim, evals, state)
                    rw = rw / pop_dim
                rf = feasible_ratio(penalty)
                pop_gbest, pop_gbest_fitness = update_gbest(pop, fitness, penalty, pop_gbest, pop_gbest_fitness, evals)
                process = process_record(process, pop_gbest_fitness, state.nfes)

            best, fearate = best_and_fearate(pop, fitness, penalty, evals)
            if pop_gbest_fitness is not None:
                best = pop_gbest_fitness
            best_values.append(best)
            fearates.append(fearate)
            times.append(timed() - start)

        summary = summarize(best_values, fearates, times)
        result = RunResult("TPDE", evals, *summary, process=process)
        results.append(result)
        if save:
            row = np.zeros((21, 8))
            row[evals - 1, :] = np.array([evals, *summary])
            save_mat(WORKSPACE_ROOT / "results" / "tpde" / f"TPDE-P{evals}.mat", everyevalBestMediMeanWorstStdFearateTime=row, testProcessBestFitAndNfes=process)
    return results


if __name__ == "__main__":
    run()
