from __future__ import annotations

import numpy as np

from .common import (
    DEFAULT_PROBLEM21_TIP_MASS,
    EvalState,
    ProgressReporter,
    RunResult,
    WORKSPACE_ROOT,
    best_and_fearate,
    best_individual_diagnostics,
    best_individual_by_feasibility,
    configure_problem_from_init_data,
    enforce_problem21_coupling,
    generate_population,
    get_fitness_and_penalty,
    iteration_setting,
    load_initial_population,
    process_best_record,
    save_mat,
    set_initial_scope,
    set_problem21_tip_mass,
    summarize,
    timed,
)


PYTHON_INIT_DIR = WORKSPACE_ROOT / "init_data"
EF_ADAPTIVE_NP_METHOD = 6
ADAPTIVE_NP_METHOD = 7
CONSTANT_NP_METHOD = 8
SUPPORTED_NP_METHODS = (EF_ADAPTIVE_NP_METHOD, ADAPTIVE_NP_METHOD, CONSTANT_NP_METHOD)
ADAPTIVE_NP_LOOKBACK = 5
ADAPTIVE_NP_IMPROVE_TARGET = 1e-3
ADAPTIVE_NP_STAGNATION = 1e-4
ADAPTIVE_NP_DIVERSITY_TARGET = 0.1
ADAPTIVE_NP_SIGMOID_GAIN = 3.0
ADAPTIVE_NP_EXPLORE_BOOST = 0.3
ADAPTIVE_NP_COLLAPSE_BOOST = 0.1
ADAPTIVE_NP_SLOW_SHRINK_RATIO = 0.95
ADAPTIVE_NP_MIN_SLOW_SHRINK_RATIO = 0.6
INITIAL_NP_DIM_FACTOR = 18.0
MIN_NP_DIM_FACTOR = 5.0
CONSTRAINT_AWARE_FTAR_METHOD = 4
FTAR_MIN = 0.25
FTAR_MAX = 0.75
FTAR_INFEASIBLE_BOOST = 0.25
FTAR_STAGNATION_MAX = 0.9
FTAR_STAGNATION_BOOST = 0.1
STANDARD_FTAR_METHOD = 3
SUPPORTED_FTAR_METHODS = (STANDARD_FTAR_METHOD, CONSTRAINT_AWARE_FTAR_METHOD)
SUCCESS_OBJECTIVE_WEIGHT = 1.0
SUCCESS_PENALTY_WEIGHT = 0.25
SUCCESS_OBJECTIVE_BASE = 0.5
SUCCESS_OBJECTIVE_FEASIBLE_GAIN = 1.0
SUCCESS_PENALTY_INFEASIBLE_GAIN = 1.0
PBEST_FRACTION = 0.15
PBEST_MIN_COUNT = 4
ELITE_ARCHIVE_SIZE_FACTOR = 2.0
STATE_HISTORY_MIN_GENERATIONS = 3
STATE_ENHANCEMENT_STAGNATION = 0.75
STATE_ENHANCEMENT_STALL_GENERATIONS = 4
STATE_ENHANCEMENT_LOW_ACCEPT_RATE = 0.12
STATE_ENHANCEMENT_FITNESS_DIVERSITY = 0.16
STATE_ENHANCEMENT_DECISION_DIVERSITY = 0.20
STAGNATION_RESAMPLE_FRACTION = 0.3
STAGNATION_RESAMPLE_THRESHOLD = 0.9
STAGNATION_RESAMPLE_DIVERSITY = 0.06
STAGNATION_RESAMPLE_STALL_GENERATIONS = 5
STAGNATION_RESAMPLE_LOW_ACCEPT_RATE = 0.08
STAGNATION_RESAMPLE_COOLDOWN_GENERATIONS = 4
STAGNATION_RESAMPLE_SIGMA_MIN = 0.015
STAGNATION_RESAMPLE_SIGMA_MAX = 0.04
LOCAL_REFINE_STAGNATION_THRESHOLD = 0.80
LOCAL_REFINE_FITNESS_DIVERSITY_THRESHOLD = 0.08
LOCAL_REFINE_DECISION_DIVERSITY_THRESHOLD = 0.18
LOCAL_REFINE_STALL_GENERATIONS = 4
LOCAL_REFINE_LOW_ACCEPT_RATE = 0.08
LOCAL_REFINE_COOLDOWN_GENERATIONS = 4
LOCAL_REFINE_MAX_TRIALS = 4
LOCAL_REFINE_SIGMA_MIN = 0.0025
LOCAL_REFINE_SIGMA_MAX = 0.012
LOCAL_REFINE_DIFF_MIN = 0.03
LOCAL_REFINE_DIFF_MAX = 0.12
ROLE_EXPLOIT = 1
ROLE_BALANCE = 2
ROLE_EXPLORE = 3
ROLE_EXPLOIT_FRACTION = 0.30
ROLE_EXPLORE_FRACTION = 0.30
ROLE_LOW_DIVERSITY_THRESHOLD = 0.02
ROLE_STAGNATION_THRESHOLD = 0.80
DIAGNOSTIC_COLUMNS = np.array(
    [
        "repeat",
        "iter",
        "nfes",
        "np",
        "best",
        "feasible_rate",
        "accept_rate",
        "best_improved",
        "mean_delta_f",
        "fitness_diversity",
        "decision_diversity",
        "mean_f",
        "mean_cr",
        "mean_ftar",
        "did_resample",
        "local_refine_attempted",
        "local_refine_accepted",
    ]
)


def _cauchy(mu: float, gamma: float) -> float:
    return float(mu + gamma * np.tan(np.pi * (np.random.rand() - 0.5)))


def _weighted_arithmetic(values: np.ndarray, delta_f: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    positive_delta = np.maximum(np.asarray(delta_f, dtype=float), 0.0)
    valid = np.isfinite(values) & np.isfinite(positive_delta)
    values = values[valid]
    positive_delta = positive_delta[valid]
    if values.size == 0:
        return 0.0

    denom = float(np.sum(positive_delta))
    if denom > 0:
        weights = positive_delta / denom
    else:
        weights = np.ones_like(positive_delta) / positive_delta.size
    return float(np.sum(weights * values))


def _sigmoid(value: float) -> float:
    value = float(np.clip(value, -60.0, 60.0))
    return float(1.0 / (1.0 + np.exp(-value)))


def _feasible_fitness_values(fitness: np.ndarray, penalty: np.ndarray) -> np.ndarray:
    feasible_fitness = np.asarray(fitness, dtype=float)[np.asarray(penalty, dtype=float) <= 0]
    return feasible_fitness[np.isfinite(feasible_fitness)]


def _decision_space_diversity(pop: np.ndarray | None, pop_min: np.ndarray | None, pop_max: np.ndarray | None) -> float:
    if pop is None or pop_min is None or pop_max is None:
        return 0.0
    arr = np.asarray(pop, dtype=float)
    if arr.ndim != 2 or arr.shape[0] <= 1:
        return 0.0
    span = np.maximum(np.asarray(pop_max, dtype=float) - np.asarray(pop_min, dtype=float), 1e-12)
    normalized = (arr - arr[0]) / span
    distances = np.linalg.norm(normalized, axis=1) / np.sqrt(arr.shape[1])
    distances = distances[np.isfinite(distances)]
    if distances.size == 0:
        return 0.0
    return float(np.mean(distances))


def _best_history_improve_rate(best_history: list[float] | None, best_now: float) -> float:
    if best_history:
        lookback_index = max(0, len(best_history) - ADAPTIVE_NP_LOOKBACK)
        best_prev = float(best_history[lookback_index])
    else:
        best_prev = best_now

    if np.isfinite(best_prev) and np.isfinite(best_now):
        return max(best_prev - best_now, 0.0) / (abs(best_prev) + 1e-8)
    return 0.0


def _best_history_stall_generations(best_history: list[float] | None) -> int:
    if not best_history or len(best_history) <= 1:
        return 0

    values = np.asarray(best_history, dtype=float)
    values = values[np.isfinite(values)]
    if values.size <= 1:
        return 0

    best_so_far = float(values[0])
    stall_generations = 0
    for value in values[1:]:
        if value < best_so_far - 1e-12:
            best_so_far = float(value)
            stall_generations = 0
        else:
            stall_generations += 1
    return int(stall_generations)


def _recent_accept_rate(accept_rate_history: list[float] | None) -> float:
    if not accept_rate_history:
        return 1.0
    values = np.asarray(accept_rate_history[-ADAPTIVE_NP_LOOKBACK:], dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 1.0
    return float(np.mean(values))


def _search_metrics(
    state: EvalState,
    fitness: np.ndarray,
    penalty: np.ndarray,
    best_history: list[float] | None,
    evals: int | None = None,
    pop: np.ndarray | None = None,
    pop_min: np.ndarray | None = None,
    pop_max: np.ndarray | None = None,
    accept_rate_history: list[float] | None = None,
) -> dict[str, float]:
    progress = float(np.clip(state.nfes / max(state.nfes_max, 1), 0.0, 1.0))
    feasible_fitness = _feasible_fitness_values(fitness, penalty)
    best_now = float(np.min(feasible_fitness)) if feasible_fitness.size else float("inf")
    improve_rate = _best_history_improve_rate(best_history, best_now)
    feasible_rate = float(np.mean(np.asarray(penalty, dtype=float) <= 0)) if np.asarray(penalty).size else 0.0

    if feasible_fitness.size > 1:
        fitness_diversity = float(np.std(feasible_fitness) / (abs(np.mean(feasible_fitness)) + 1e-8))
    else:
        fitness_diversity = 0.0

    diversity_score = 1.0 if feasible_fitness.size == 0 else float(
        np.clip(fitness_diversity / ADAPTIVE_NP_DIVERSITY_TARGET, 0.0, 1.0)
    )
    stagnation_arg = ADAPTIVE_NP_SIGMOID_GAIN * (
        (ADAPTIVE_NP_STAGNATION - improve_rate) / (ADAPTIVE_NP_STAGNATION + 1e-12)
    )
    stagnation_score = _sigmoid(stagnation_arg)
    return {
        "progress": progress,
        "best_now": best_now,
        "improve_rate": improve_rate,
        "feasible_rate": feasible_rate,
        "fitness_diversity": fitness_diversity,
        "decision_diversity": _decision_space_diversity(pop, pop_min, pop_max),
        "diversity_score": diversity_score,
        "stagnation_score": stagnation_score,
        "stagnation_generations": float(_best_history_stall_generations(best_history)),
        "recent_accept_rate": _recent_accept_rate(accept_rate_history),
        "history_size": float(len(best_history or [])),
        "nfes_max": float(state.nfes_max),
        "evals": float(evals or 0),
    }


def _state_enhancements_active(metrics: dict[str, float] | None) -> bool:
    if metrics is None:
        return False
    if float(metrics.get("history_size", 0.0)) < STATE_HISTORY_MIN_GENERATIONS:
        return False

    stagnated = (
        float(metrics.get("stagnation_score", 0.0)) >= STATE_ENHANCEMENT_STAGNATION
        or float(metrics.get("stagnation_generations", 0.0)) >= STATE_ENHANCEMENT_STALL_GENERATIONS
    )
    low_acceptance = float(metrics.get("recent_accept_rate", 1.0)) <= STATE_ENHANCEMENT_LOW_ACCEPT_RATE
    low_diversity = (
        float(metrics.get("fitness_diversity", 1.0)) <= STATE_ENHANCEMENT_FITNESS_DIVERSITY
        or float(metrics.get("decision_diversity", 1.0)) <= STATE_ENHANCEMENT_DECISION_DIVERSITY
    )
    poor_feasibility = float(metrics.get("feasible_rate", 1.0)) < 1.0
    return stagnated and (low_acceptance or low_diversity or poor_feasibility)


def _state_enhancement_weight(metrics: dict[str, float] | None) -> float:
    if metrics is None:
        return 0.0
    stagnation = float(np.clip(metrics.get("stagnation_score", 0.0), 0.0, 1.0))
    stall = float(np.clip(metrics.get("stagnation_generations", 0.0) / max(STATE_ENHANCEMENT_STALL_GENERATIONS, 1), 0.0, 1.0))
    recent_accept = float(metrics.get("recent_accept_rate", 1.0))
    low_acceptance = float(np.clip((STATE_ENHANCEMENT_LOW_ACCEPT_RATE - recent_accept) / STATE_ENHANCEMENT_LOW_ACCEPT_RATE, 0.0, 1.0))
    fitness_collapse = 1.0 - float(np.clip(metrics.get("fitness_diversity", 1.0) / STATE_ENHANCEMENT_FITNESS_DIVERSITY, 0.0, 1.0))
    decision_collapse = 1.0 - float(np.clip(metrics.get("decision_diversity", 1.0) / STATE_ENHANCEMENT_DECISION_DIVERSITY, 0.0, 1.0))
    diversity_pressure = max(fitness_collapse, decision_collapse, 0.0)
    return float(np.clip(0.35 * stagnation + 0.25 * stall + 0.25 * low_acceptance + 0.15 * diversity_pressure, 0.0, 1.0))


def _robust_merit_ita(merit: np.ndarray) -> float:
    merit = np.asarray(merit, dtype=float)
    merit = merit[np.isfinite(merit)]
    if merit.size <= 1:
        return 0.0

    low, high = np.percentile(merit, [10, 90])
    spread = float(high - low)
    if abs(spread) <= 1e-12:
        return 0.0
    normalized = np.clip((merit - low) / (spread + 1e-12), 0.0, 1.0)
    return float(np.mean(normalized))


def constraint_aware_ftar(
    pop_fitness: np.ndarray,
    pop_penalty: np.ndarray,
    eps: float,
    metrics: dict[str, float] | None = None,
    enable_late_enhancements: bool = True,
) -> float:
    fitness = np.asarray(pop_fitness, dtype=float).reshape(-1)
    penalty = np.asarray(pop_penalty, dtype=float).reshape(-1)
    finite = np.isfinite(fitness) & np.isfinite(penalty)
    fitness = fitness[finite]
    penalty = penalty[finite]
    if fitness.size == 0:
        return FTAR_MAX

    penalty_excess = np.maximum(penalty - eps, 0.0)
    fit_q25, fit_q75 = np.percentile(fitness, [25, 75])
    pen_q25, pen_q75 = np.percentile(penalty_excess, [25, 75])
    fit_scale = abs(float(fit_q75 - fit_q25))
    pen_scale = abs(float(pen_q75 - pen_q25))
    penalty_coeff = fit_scale / (pen_scale + 1e-8) if pen_scale > 0 else 1.0

    merit = fitness + penalty_coeff * penalty_excess
    ita = _robust_merit_ita(merit)
    feasible_rate = float(np.mean(penalty <= eps))
    ftar = 1.0 - np.sin(ita * np.pi / 2)
    ftar += FTAR_INFEASIBLE_BOOST * (1.0 - feasible_rate)
    ftar_max = FTAR_MAX
    if enable_late_enhancements and _state_enhancements_active(metrics):
        dynamic_weight = _state_enhancement_weight(metrics)
        ftar += FTAR_STAGNATION_BOOST * dynamic_weight
        ftar_max = FTAR_MAX + (FTAR_STAGNATION_MAX - FTAR_MAX) * dynamic_weight
    return float(np.clip(ftar, FTAR_MIN, ftar_max))


def mutation_and_crossover_params(
    memory_len: int,
    mcr: np.ndarray,
    mf: np.ndarray,
    pop_fitness: np.ndarray,
    pop_penalty: np.ndarray,
    eps: float,
    ftar_method: int,
    metrics: dict[str, float] | None = None,
    enable_late_enhancements: bool = True,
) -> tuple[float, float, float]:
    r = np.random.randint(memory_len)
    cr = float(np.random.normal(mcr[r], 0.1))
    cr = min(max(cr, 0.0), 1.0)

    f = _cauchy(float(mf[r]), 0.1)
    while f < 0:
        f = _cauchy(float(mf[r]), 0.1)
    f = min(f, 1.0)

    if ftar_method == STANDARD_FTAR_METHOD:
        best = np.min(pop_fitness)
        worst = np.max(pop_fitness)
        ita = np.sum((pop_fitness - best) / (worst - best + 1e-8)) / pop_fitness.size
        ftar = float(1 - np.sin(ita * np.pi / 2))
    elif ftar_method == CONSTRAINT_AWARE_FTAR_METHOD:
        ftar = constraint_aware_ftar(pop_fitness, pop_penalty, eps, metrics, enable_late_enhancements)
    else:
        raise ValueError(f"Unsupported Ftar method: {ftar_method}; supported methods are {SUPPORTED_FTAR_METHODS}")
    return f, ftar, cr


def _unique_indices(count: int, upper: int, exclude: int | None = None) -> np.ndarray:
    if exclude is None:
        candidates = np.arange(upper)
    else:
        candidates = np.array([idx for idx in range(upper) if idx != exclude], dtype=int)
    if candidates.size == 0:
        candidates = np.arange(upper)
    replace = candidates.size < count
    return np.random.choice(candidates, size=count, replace=replace)


def _top_feasible(pop: np.ndarray, fitness: np.ndarray, penalty: np.ndarray, count: int, eps: float = 0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if pop.size == 0 or count <= 0:
        return pop[:0], fitness[:0], penalty[:0]
    candidate_idx = np.flatnonzero(np.asarray(penalty, dtype=float) <= eps)
    if candidate_idx.size == 0:
        return pop[:0], fitness[:0], penalty[:0]
    order = candidate_idx[np.argsort(np.asarray(fitness, dtype=float)[candidate_idx], kind="stable")]
    keep = order[:count]
    return pop[keep], fitness[keep], penalty[keep]


def update_elite_archive(
    elite_pop: np.ndarray,
    elite_fitness: np.ndarray,
    elite_penalty: np.ndarray,
    pop: np.ndarray,
    fitness: np.ndarray,
    penalty: np.ndarray,
    pop_pbest: np.ndarray,
    pbest_fitness: np.ndarray,
    pbest_penalty: np.ndarray,
    max_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    current_pop, current_fitness, current_penalty = _top_feasible(pop, fitness, penalty, max_size)
    pbest_pop, pbest_fit, pbest_pen = _top_feasible(pop_pbest, pbest_fitness, pbest_penalty, max_size)

    pools = [elite_pop, current_pop, pbest_pop]
    fit_pool = [elite_fitness, current_fitness, pbest_fit]
    pen_pool = [elite_penalty, current_penalty, pbest_pen]
    non_empty = [idx for idx, pool in enumerate(pools) if pool.size]
    if not non_empty:
        return elite_pop, elite_fitness, elite_penalty

    merged_pop = np.vstack([pools[idx] for idx in non_empty])
    merged_fitness = np.concatenate([fit_pool[idx] for idx in non_empty])
    merged_penalty = np.concatenate([pen_pool[idx] for idx in non_empty])
    finite = np.isfinite(merged_fitness) & np.isfinite(merged_penalty) & (merged_penalty <= 0)
    merged_pop = merged_pop[finite]
    merged_fitness = merged_fitness[finite]
    merged_penalty = merged_penalty[finite]
    if merged_pop.size == 0:
        return elite_pop, elite_fitness, elite_penalty

    _, unique_idx = np.unique(np.round(merged_pop, decimals=10), axis=0, return_index=True)
    unique_idx = np.sort(unique_idx)
    merged_pop = merged_pop[unique_idx]
    merged_fitness = merged_fitness[unique_idx]
    merged_penalty = merged_penalty[unique_idx]
    order = np.argsort(merged_fitness, kind="stable")[:max_size]
    return merged_pop[order], merged_fitness[order], merged_penalty[order]


def build_pbest_pool(
    pop: np.ndarray,
    fitness: np.ndarray,
    penalty: np.ndarray,
    pop_pbest: np.ndarray,
    pbest_fitness: np.ndarray,
    pbest_penalty: np.ndarray,
    elite_pop: np.ndarray,
    eps: float,
) -> np.ndarray:
    count = max(PBEST_MIN_COUNT, int(round(PBEST_FRACTION * pop.shape[0])))
    current_pop, _, _ = _top_feasible(pop, fitness, penalty, count)
    pbest_pop, _, _ = _top_feasible(pop_pbest, pbest_fitness, pbest_penalty, count)
    relaxed_pop, _, _ = _top_feasible(pop, fitness, penalty, count, eps)
    pools = [current_pop, pbest_pop, elite_pop, relaxed_pop]
    non_empty = [pool for pool in pools if pool.size]
    if not non_empty:
        return pop[:1].copy()
    return np.vstack(non_empty)


def choose_pbest_target(fallback: np.ndarray, pbest_pool: np.ndarray) -> np.ndarray:
    if pbest_pool.size == 0:
        return fallback
    return pbest_pool[np.random.randint(pbest_pool.shape[0])]


def search_role_labels(fitness: np.ndarray, penalty: np.ndarray, metrics: dict[str, float] | None = None) -> np.ndarray:
    fitness = np.asarray(fitness, dtype=float).reshape(-1)
    penalty = np.asarray(penalty, dtype=float).reshape(-1)
    role_labels = np.full(fitness.shape, ROLE_BALANCE, dtype=int)
    finite = np.isfinite(fitness) & np.isfinite(penalty)
    if fitness.size == 0:
        return role_labels

    feasible_idx = np.flatnonzero(finite & (penalty <= 0))
    if feasible_idx.size == 0:
        role_labels[finite] = ROLE_EXPLORE
        return role_labels

    order = feasible_idx[np.argsort(fitness[feasible_idx], kind="stable")]
    n_feasible = order.size
    exploit_fraction = ROLE_EXPLOIT_FRACTION
    explore_fraction = ROLE_EXPLORE_FRACTION
    if metrics is not None:
        decision_diversity = float(metrics.get("decision_diversity", 1.0))
        stagnation_score = float(metrics.get("stagnation_score", 0.0))
        state_weight = _state_enhancement_weight(metrics) if _state_enhancements_active(metrics) else 0.0
        if state_weight > 0:
            exploit_fraction = max(0.15, exploit_fraction - 0.10 * state_weight)
            explore_fraction = min(0.55, explore_fraction + 0.15 * state_weight)
        if decision_diversity <= ROLE_LOW_DIVERSITY_THRESHOLD:
            exploit_fraction = max(0.15, exploit_fraction - 0.10)
            explore_fraction = min(0.55, explore_fraction + 0.15)
        if stagnation_score >= ROLE_STAGNATION_THRESHOLD:
            explore_fraction = min(0.60, explore_fraction + 0.10)

    n_exploit = max(1, int(round(exploit_fraction * n_feasible)))
    n_explore = max(1, int(round(explore_fraction * n_feasible))) if n_feasible > 1 else 0
    n_exploit = min(n_exploit, n_feasible)
    n_explore = min(n_explore, max(0, n_feasible - n_exploit))

    role_labels[order[:n_exploit]] = ROLE_EXPLOIT
    if n_explore > 0:
        role_labels[order[-n_explore:]] = ROLE_EXPLORE
    return role_labels


def mutation_results(
    f: float,
    ftar: float,
    pop_best: np.ndarray,
    pop_pbest: np.ndarray,
    pbest_pool: np.ndarray,
    pop: np.ndarray,
    i: int,
    class_labels: np.ndarray,
    role_labels: np.ndarray,
    evals: int,
    pop_max: np.ndarray,
    pop_min: np.ndarray,
    np_g: int,
) -> np.ndarray:
    p1_indices = np.flatnonzero(class_labels == 1)
    p2_indices = np.flatnonzero(class_labels == 2)
    role = int(role_labels[i]) if role_labels.size else ROLE_BALANCE

    if class_labels[i] == 1:
        if role == ROLE_EXPLOIT:
            r2 = _unique_indices(1, np_g, exclude=i)[0]
            pop_p2 = pop[np.random.choice(p2_indices)] if p2_indices.size else pop[i]
            pbest_target = choose_pbest_target(pop_best, pbest_pool)
            v = pop[i] + f * (pbest_target - pop[i]) + ftar * (pop[r2] - pop_p2)
        elif role == ROLE_EXPLORE:
            r1, r2, r3, r4, r5 = _unique_indices(5, np_g, exclude=i)
            v = pop[r1] + f * (pop[r2] - pop[r3]) + ftar * (pop[r4] - pop[r5])
        else:
            r1, r2 = _unique_indices(2, np_g, exclude=i)
            pbest_target = choose_pbest_target(pop_pbest[i], pbest_pool)
            v = pop[i] + f * (pbest_target - pop[i]) + ftar * (pop[r1] - pop[r2])
    elif class_labels[i] == 2:
        if role == ROLE_EXPLORE:
            r1, r2, r3, r4 = _unique_indices(4, np_g, exclude=i)
            feasible_anchor = pop[np.random.choice(p1_indices)] if p1_indices.size else pop[r1]
            v = pop[i] + 0.5 * f * (feasible_anchor - pop[i]) + ftar * (pop[r2] - pop[r3]) + 0.5 * f * (pop[r1] - pop[r4])
        elif role == ROLE_EXPLOIT:
            r1 = _unique_indices(1, np_g, exclude=i)[0]
            pop_p1 = pop[np.random.choice(p1_indices)] if p1_indices.size else pop[i]
            pbest_target = choose_pbest_target(pop_pbest[i], pbest_pool)
            v = pop[i] + f * (pbest_target - pop[i]) + ftar * (pop[r1] - pop_p1)
        else:
            r1, r2 = _unique_indices(2, np_g, exclude=i)
            pbest_target = choose_pbest_target(pop_pbest[i], pbest_pool)
            v = pop[i] + f * (pbest_target - pop[i]) + ftar * (pop[r1] - pop[r2])
    else:
        r1, r2, r3, r4 = _unique_indices(4, np_g, exclude=i)
        v = pop[i] + f * (pop[r1] - pop[r2]) + ftar * (pop[r3] - pop[r4])

    v = np.minimum(np.maximum(v, pop_min), pop_max)
    return enforce_problem21_coupling(v) if evals == 21 else v


def crossover(individual: np.ndarray, v: np.ndarray, cr: float, pop_max: np.ndarray, pop_min: np.ndarray, evals: int) -> np.ndarray:
    pop_dim = individual.size
    u = np.zeros_like(v)
    j_rand = np.random.randint(pop_dim)
    for j in range(pop_dim):
        if np.random.rand() < cr or j == j_rand:
            u[j] = v[j]
        else:
            u[j] = individual[j]
    u = np.minimum(np.maximum(u, pop_min), pop_max)
    return enforce_problem21_coupling(u) if evals == 21 else u


def greedy_choose(
    u: np.ndarray,
    u_fitness: float,
    u_penalty: float,
    person_penalty: float,
    person_fitness: float,
    f: float,
    cr: float,
    person: np.ndarray,
    eps: float,
    metrics: dict[str, float] | None = None,
    enable_late_enhancements: bool = True,
) -> tuple[np.ndarray, float, float, float, float, float]:
    accepted = epsilon_better(u_fitness, u_penalty, person_fitness, person_penalty, eps)

    if accepted:
        objective_gain = max(float(person_fitness - u_fitness), 0.0) / (abs(float(person_fitness)) + abs(float(u_fitness)) + 1e-8)
        penalty_gain = max(float(person_penalty - u_penalty), 0.0) / (abs(float(person_penalty)) + abs(float(u_penalty)) + 1e-8)
        if enable_late_enhancements and _state_enhancements_active(metrics):
            feasible_rate = float(metrics.get("feasible_rate", 0.0))
            objective_weight = SUCCESS_OBJECTIVE_BASE + SUCCESS_OBJECTIVE_FEASIBLE_GAIN * feasible_rate
            penalty_weight = SUCCESS_PENALTY_INFEASIBLE_GAIN * (1.0 - feasible_rate)
        else:
            objective_weight = SUCCESS_OBJECTIVE_WEIGHT
            penalty_weight = SUCCESS_PENALTY_WEIGHT
        delta_f = objective_weight * objective_gain + penalty_weight * penalty_gain
        return u, float(u_fitness), float(u_penalty), cr, f, delta_f
    return person, float(person_fitness), float(person_penalty), -1.0, -1.0, -1.0


def generate_epsilon(eps0: float, penalty: np.ndarray, metrics: dict[str, float] | None = None) -> float:
    positive_penalty = np.asarray(penalty, dtype=float).reshape(-1)
    positive_penalty = positive_penalty[np.isfinite(positive_penalty) & (positive_penalty > 0)]
    if positive_penalty.size == 0:
        return 0.0

    feasible_rate = (
        float(metrics.get("feasible_rate", 0.0))
        if metrics is not None
        else 1.0 - positive_penalty.size / max(np.asarray(penalty).size, 1)
    )
    quantile = 75.0 - 50.0 * float(np.clip(feasible_rate, 0.0, 1.0))
    if metrics is not None and _state_enhancements_active(metrics):
        quantile += 10.0 * _state_enhancement_weight(metrics)
    eps = float(np.percentile(positive_penalty, np.clip(quantile, 10.0, 90.0)))
    return float(np.clip(eps, 0.0, max(float(eps0), 0.0)))


def epsilon_better(
    candidate_fitness: float,
    candidate_penalty: float,
    incumbent_fitness: float,
    incumbent_penalty: float,
    eps: float,
) -> bool:
    if candidate_penalty <= eps and incumbent_penalty <= eps:
        return candidate_fitness < incumbent_fitness
    if np.isclose(candidate_penalty, incumbent_penalty):
        return candidate_fitness < incumbent_fitness
    return candidate_penalty < incumbent_penalty


def subpopulation_labels(penalty: np.ndarray, eps: float) -> np.ndarray:
    labels = np.full(penalty.shape, 3, dtype=int)
    labels[(penalty > 0) & (penalty <= eps)] = 2
    labels[penalty <= 0] = 1
    return labels


def sort_by_constraint(
    pop: np.ndarray,
    fitness: np.ndarray,
    penalty: np.ndarray,
    eps0: float,
    pop_pbest: np.ndarray,
    pbest_fitness: np.ndarray,
    pbest_penalty: np.ndarray,
    metrics: dict[str, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    eps = generate_epsilon(eps0, penalty, metrics)
    labels = subpopulation_labels(penalty, eps)
    relaxed = penalty <= eps
    strict = ~relaxed
    p1 = labels == 1
    p2 = labels == 2
    p3 = labels == 3

    num_p1 = int(np.sum(p1))
    num_p2 = int(np.sum(p2))
    num_p3 = int(np.sum(p3))

    relaxed_idx = np.where(relaxed)[0]
    relaxed_idx = relaxed_idx[np.argsort(fitness[relaxed_idx], kind="stable")]
    strict_idx = np.where(strict)[0]
    strict_idx = strict_idx[np.lexsort((fitness[strict_idx], penalty[strict_idx]))] if strict_idx.size else strict_idx
    rank = np.concatenate([relaxed_idx, strict_idx])

    return (
        pop[rank],
        fitness[rank],
        penalty[rank],
        num_p1,
        num_p2,
        num_p3,
        pop_pbest[rank],
        pbest_fitness[rank],
        pbest_penalty[rank],
        labels[rank],
    )


def update_pop_pbest(
    pop: np.ndarray,
    fitness: np.ndarray,
    penalty: np.ndarray,
    pop_pbest: np.ndarray,
    pbest_fitness: np.ndarray,
    pbest_penalty: np.ndarray,
    eps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    for i in range(pop.shape[0]):
        better = epsilon_better(fitness[i], penalty[i], pbest_fitness[i], pbest_penalty[i], eps)
        if better:
            pop_pbest[i] = pop[i]
            pbest_fitness[i] = fitness[i]
            pbest_penalty[i] = penalty[i]
    return pop_pbest, pbest_fitness, pbest_penalty


def _weighted_lehmer(values: np.ndarray, delta_f: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    positive_delta = np.maximum(np.asarray(delta_f, dtype=float), 0.0)
    valid = np.isfinite(values) & np.isfinite(positive_delta)
    values = values[valid]
    positive_delta = positive_delta[valid]
    if values.size == 0:
        return 0.0

    denom = float(np.sum(positive_delta))
    if denom > 0:
        weights = positive_delta / denom
    else:
        weights = np.ones_like(positive_delta) / positive_delta.size
    return float(np.sum(weights * values**2) / (np.sum(weights * values) + 1e-8))


def update_mcr_and_mf(
    k: int,
    memory_len: int,
    mcr: np.ndarray,
    mf: np.ndarray,
    scr: np.ndarray,
    sf: np.ndarray,
    delta_f: np.ndarray,
) -> tuple[int, np.ndarray, np.ndarray]:
    if scr.size == 0 and sf.size == 0:
        return k, mcr, mf
    mcr_new = mcr.copy()
    mf_new = mf.copy()
    if mcr[k] != 0 and (scr.size == 0 or np.max(scr) != 0):
        mcr_new[k] = _weighted_arithmetic(scr, delta_f)
    if sf.size:
        mf_new[k] = _weighted_lehmer(sf, delta_f)
    return (k + 1) % memory_len, mcr_new, mf_new


def should_resample_stagnated_tail(
    metrics: dict[str, float],
    state: EvalState,
    last_resample_nfes: int,
    elite_pop: np.ndarray,
) -> bool:
    if not _state_enhancements_active(metrics):
        return False
    if elite_pop.size == 0:
        return False
    cooldown_nfes = STAGNATION_RESAMPLE_COOLDOWN_GENERATIONS * max(state.np_g, 1)
    if state.nfes - last_resample_nfes < max(cooldown_nfes, state.np_g):
        return False
    low_acceptance = float(metrics.get("recent_accept_rate", 1.0)) <= STAGNATION_RESAMPLE_LOW_ACCEPT_RATE
    long_stall = float(metrics.get("stagnation_generations", 0.0)) >= STAGNATION_RESAMPLE_STALL_GENERATIONS
    low_fitness_diversity = metrics["fitness_diversity"] <= STAGNATION_RESAMPLE_DIVERSITY
    return (
        metrics["stagnation_score"] >= STAGNATION_RESAMPLE_THRESHOLD
        and (low_fitness_diversity or low_acceptance or long_stall)
    )


def resample_stagnated_tail(
    pop: np.ndarray,
    fitness: np.ndarray,
    penalty: np.ndarray,
    pop_next: np.ndarray,
    next_fitness: np.ndarray,
    next_penalty: np.ndarray,
    pop_pbest: np.ndarray,
    pbest_fitness: np.ndarray,
    pbest_penalty: np.ndarray,
    elite_pop: np.ndarray,
    pop_max: np.ndarray,
    pop_min: np.ndarray,
    evals: int,
    state: EvalState,
    metrics: dict[str, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:
    remaining_budget = max(state.nfes_max - state.nfes, 0)
    if remaining_budget <= 0 or elite_pop.size == 0:
        return pop, fitness, penalty, pop_next, next_fitness, next_penalty, pop_pbest, pbest_fitness, pbest_penalty, False

    n_resample = int(round(STAGNATION_RESAMPLE_FRACTION * state.np_g))
    n_resample = max(1, min(n_resample, state.np_g - 1, remaining_budget))
    replace_idx = np.arange(state.np_g - n_resample, state.np_g)
    elite_idx = np.random.choice(elite_pop.shape[0], size=n_resample, replace=elite_pop.shape[0] < n_resample)
    state_weight = _state_enhancement_weight(metrics)
    diversity_gap = 1.0 - float(np.clip(metrics.get("decision_diversity", 1.0) / STATE_ENHANCEMENT_DECISION_DIVERSITY, 0.0, 1.0))
    sigma_ratio = STAGNATION_RESAMPLE_SIGMA_MIN + (STAGNATION_RESAMPLE_SIGMA_MAX - STAGNATION_RESAMPLE_SIGMA_MIN) * max(state_weight, diversity_gap)
    sigma = sigma_ratio * (pop_max - pop_min)
    candidates = elite_pop[elite_idx] + np.random.normal(0.0, sigma, size=(n_resample, pop.shape[1]))
    candidates = np.minimum(np.maximum(candidates, pop_min), pop_max)
    if evals == 21:
        candidates = enforce_problem21_coupling(candidates)

    cand_fitness, cand_penalty, inf_flags = get_fitness_and_penalty(candidates, evals, opmwade_repair_inf=True)
    state.nfes += n_resample + int(np.sum(inf_flags != 0))
    pop[replace_idx] = candidates
    fitness[replace_idx] = cand_fitness
    penalty[replace_idx] = cand_penalty
    pop_next[replace_idx] = candidates
    next_fitness[replace_idx] = cand_fitness
    next_penalty[replace_idx] = cand_penalty
    pop_pbest[replace_idx] = candidates
    pbest_fitness[replace_idx] = cand_fitness
    pbest_penalty[replace_idx] = cand_penalty
    return pop, fitness, penalty, pop_next, next_fitness, next_penalty, pop_pbest, pbest_fitness, pbest_penalty, True


def should_local_refine(
    metrics: dict[str, float],
    state: EvalState,
    last_refine_nfes: int,
    elite_pop: np.ndarray,
) -> bool:
    if elite_pop.size == 0:
        return False
    cooldown_nfes = LOCAL_REFINE_COOLDOWN_GENERATIONS * max(state.np_g, 1)
    if state.nfes - last_refine_nfes < max(cooldown_nfes, state.np_g):
        return False
    collapsed = (
        metrics["fitness_diversity"] <= LOCAL_REFINE_FITNESS_DIVERSITY_THRESHOLD
        and metrics["decision_diversity"] <= LOCAL_REFINE_DECISION_DIVERSITY_THRESHOLD
    )
    ineffective = (
        float(metrics.get("recent_accept_rate", 1.0)) <= LOCAL_REFINE_LOW_ACCEPT_RATE
        and float(metrics.get("stagnation_generations", 0.0)) >= LOCAL_REFINE_STALL_GENERATIONS
    )
    return _state_enhancements_active(metrics) and (
        metrics["stagnation_score"] >= LOCAL_REFINE_STAGNATION_THRESHOLD
        and (collapsed or ineffective)
    )


def local_refine_elite(
    pop: np.ndarray,
    fitness: np.ndarray,
    penalty: np.ndarray,
    pop_next: np.ndarray,
    next_fitness: np.ndarray,
    next_penalty: np.ndarray,
    pop_pbest: np.ndarray,
    pbest_fitness: np.ndarray,
    pbest_penalty: np.ndarray,
    elite_pop: np.ndarray,
    pop_max: np.ndarray,
    pop_min: np.ndarray,
    evals: int,
    eps: float,
    state: EvalState,
    metrics: dict[str, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:
    remaining_budget = max(state.nfes_max - state.nfes, 0)
    if remaining_budget <= 0 or elite_pop.size == 0:
        return pop, fitness, penalty, pop_next, next_fitness, next_penalty, pop_pbest, pbest_fitness, pbest_penalty, False

    n_trials = max(1, min(LOCAL_REFINE_MAX_TRIALS, state.np_g // 4, remaining_budget))
    state_weight = _state_enhancement_weight(metrics)
    sigma_ratio = LOCAL_REFINE_SIGMA_MIN + (LOCAL_REFINE_SIGMA_MAX - LOCAL_REFINE_SIGMA_MIN) * state_weight
    diff_scale = LOCAL_REFINE_DIFF_MIN + (LOCAL_REFINE_DIFF_MAX - LOCAL_REFINE_DIFF_MIN) * state_weight
    sigma = sigma_ratio * (pop_max - pop_min)
    target = elite_pop[0]
    candidates = np.empty((n_trials, pop.shape[1]))

    for row in range(n_trials):
        if elite_pop.shape[0] >= 3:
            r1, r2 = np.random.choice(elite_pop.shape[0], size=2, replace=False)
            diff = diff_scale * (elite_pop[r1] - elite_pop[r2])
        else:
            diff = 0.0
        candidates[row] = target + diff + np.random.normal(0.0, sigma, size=pop.shape[1])

    candidates = np.minimum(np.maximum(candidates, pop_min), pop_max)
    if evals == 21:
        candidates = enforce_problem21_coupling(candidates)

    cand_fitness, cand_penalty, inf_flags = get_fitness_and_penalty(candidates, evals, opmwade_repair_inf=True)
    state.nfes += n_trials + int(np.sum(inf_flags != 0))

    accepted = False
    replace_idx = np.arange(state.np_g - n_trials, state.np_g)
    candidate_order = np.lexsort((cand_fitness, cand_penalty))
    for dst, src in zip(replace_idx, candidate_order):
        if epsilon_better(cand_fitness[src], cand_penalty[src], fitness[dst], penalty[dst], eps):
            candidate = candidates[src]
            pop[dst] = candidate
            fitness[dst] = cand_fitness[src]
            penalty[dst] = cand_penalty[src]
            pop_next[dst] = candidate
            next_fitness[dst] = cand_fitness[src]
            next_penalty[dst] = cand_penalty[src]
            pop_pbest[dst] = candidate
            pbest_fitness[dst] = cand_fitness[src]
            pbest_penalty[dst] = cand_penalty[src]
            accepted = True

    return pop, fitness, penalty, pop_next, next_fitness, next_penalty, pop_pbest, pbest_fitness, pbest_penalty, accepted


def _feasible_best(fitness: np.ndarray, penalty: np.ndarray) -> float:
    feasible_fitness = _feasible_fitness_values(fitness, penalty)
    if feasible_fitness.size == 0:
        return float("inf")
    return float(np.min(feasible_fitness))


def _protected_np_min(np_min: int, pop_dim: int) -> int:
    return max(int(np_min), int(round(MIN_NP_DIM_FACTOR * pop_dim)))


def _effective_np_min(
    np_min: int,
    pop_dim: int,
    enable_late_enhancements: bool,
    metrics: dict[str, float] | None = None,
) -> int:
    if not enable_late_enhancements:
        return int(np_min)
    protected_min = _protected_np_min(np_min, pop_dim)
    if metrics is not None and _state_enhancements_active(metrics):
        stagnated = metrics["stagnation_score"] >= LOCAL_REFINE_STAGNATION_THRESHOLD
        collapsed = (
            metrics["fitness_diversity"] <= LOCAL_REFINE_FITNESS_DIVERSITY_THRESHOLD
            and metrics["decision_diversity"] <= LOCAL_REFINE_DECISION_DIVERSITY_THRESHOLD
        )
        if stagnated and collapsed:
            return protected_min
    return protected_min


def _adaptive_population_size(
    state: EvalState,
    fitness: np.ndarray,
    penalty: np.ndarray,
    np_max: int,
    np_min: int,
    best_history: list[float] | None,
    accept_rate_history: list[float] | None,
    evals: int,
    pop_dim: int,
    enable_late_enhancements: bool = True,
    metrics: dict[str, float] | None = None,
) -> int:
    if metrics is None:
        metrics = _search_metrics(state, fitness, penalty, best_history, evals, accept_rate_history=accept_rate_history)
    improve_rate = metrics["improve_rate"]
    stagnation_score = metrics["stagnation_score"]
    diversity_score = metrics["diversity_score"]
    effective_min = _effective_np_min(np_min, pop_dim, enable_late_enhancements, metrics)

    improvement_score = float(np.clip(improve_rate / ADAPTIVE_NP_IMPROVE_TARGET, 0.0, 1.0))
    recent_accept = float(metrics.get("recent_accept_rate", 1.0))
    current_np = max(int(state.np_g), 1)
    need_explore = float(np.clip(stagnation_score * diversity_score, 0.0, 1.0))
    need_injection = float(np.clip(stagnation_score * (1.0 - diversity_score), 0.0, 1.0))

    if _state_enhancements_active(metrics):
        growth = 1.0 + ADAPTIVE_NP_EXPLORE_BOOST * need_explore + ADAPTIVE_NP_COLLAPSE_BOOST * need_injection
        if recent_accept <= STATE_ENHANCEMENT_LOW_ACCEPT_RATE:
            growth += 0.08
        new_np = max(current_np, int(round(current_np * growth)))
    elif improvement_score > 0.5 and recent_accept > STATE_ENHANCEMENT_LOW_ACCEPT_RATE:
        shrink_ratio = ADAPTIVE_NP_SLOW_SHRINK_RATIO if diversity_score > 0.5 else 0.98
        new_np = int(round(current_np * shrink_ratio))
    else:
        new_np = current_np

    if stagnation_score > 0.8:
        slow_shrink_ratio = max(
            ADAPTIVE_NP_MIN_SLOW_SHRINK_RATIO,
            ADAPTIVE_NP_SLOW_SHRINK_RATIO - 0.2 * (1.0 - diversity_score),
        )
        slow_shrink_np = int(round(slow_shrink_ratio * state.np_g))
        if diversity_score > 0.5:
            new_np = max(new_np, slow_shrink_np)
        elif need_injection > 0.8:
            injection = max(1, int(round(0.02 * np_max)))
            new_np = max(new_np, slow_shrink_np + injection)

    return int(np.clip(new_np, effective_min, np_max))


def num_pop_update(
    num_method: int,
    state: EvalState,
    pop: np.ndarray,
    fitness: np.ndarray,
    pop_best_fitness: float,
    pop_worst_fitness: float,
    np_max: int,
    np_min: int,
    pop_max: np.ndarray,
    pop_min: np.ndarray,
    pop_best: np.ndarray,
    pop_next: np.ndarray,
    evals: int,
    penalty: np.ndarray,
    next_fitness: np.ndarray,
    next_penalty: np.ndarray,
    pop_pbest: np.ndarray,
    pbest_fitness: np.ndarray,
    pbest_penalty: np.ndarray,
    best_history: list[float] | None = None,
    accept_rate_history: list[float] | None = None,
    enable_late_enhancements: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    np_metrics = _search_metrics(state, fitness, penalty, best_history, evals, pop, pop_min, pop_max, accept_rate_history)
    best_now = float(np_metrics.get("best_now", float("inf")))
    if best_history is not None and np.isfinite(best_now):
        best_history.append(best_now)

    if num_method == CONSTANT_NP_METHOD:
        return pop, fitness, penalty, pop_next, next_fitness, next_penalty, pop_pbest, pbest_fitness, pbest_penalty

    if num_method == ADAPTIVE_NP_METHOD:
        new_np = _adaptive_population_size(
            state,
            fitness,
            penalty,
            np_max,
            np_min,
            best_history,
            accept_rate_history,
            evals,
            pop.shape[1],
            enable_late_enhancements,
            np_metrics,
        )
    elif num_method == EF_ADAPTIVE_NP_METHOD:
        old_np = state.np_g
        feasible_count = int(np.sum(penalty <= 0))
        if feasible_count == 0:
            ef_total = abs(np.sum((penalty - penalty[0]) / (penalty[-1] - penalty[0] - 1e-8)))
        elif feasible_count == old_np:
            ef_total = abs(np.sum((fitness - pop_best_fitness) / (pop_worst_fitness - pop_best_fitness - 1e-8)))
        else:
            pop_feasi_ef = abs(np.sum((fitness[:feasible_count] - pop_best_fitness) / (fitness[feasible_count - 1] - pop_best_fitness - 1e-8)))
            tail = penalty[feasible_count:]
            pop_ufeasi_ef = abs(np.sum((tail - tail[0]) / (tail[-1] - tail[0] - 1e-8))) if tail.size else 0
            ef_total = pop_feasi_ef + pop_ufeasi_ef

        ef_mean = ef_total / max(old_np - 1, 1)
        ef_score = float(np.clip(ef_mean, 0.0, 1.0))
        recent_accept = float(np_metrics.get("recent_accept_rate", 1.0))
        if _state_enhancements_active(np_metrics):
            growth = 1.05 + 0.10 * (1.0 - ef_score)
            if recent_accept <= STATE_ENHANCEMENT_LOW_ACCEPT_RATE:
                growth += 0.05
            new_np = max(old_np, int(round(old_np * growth)))
        elif ef_score > 0.5 and recent_accept > STATE_ENHANCEMENT_LOW_ACCEPT_RATE:
            new_np = int(round(old_np * ADAPTIVE_NP_SLOW_SHRINK_RATIO))
        else:
            new_np = old_np
    else:
        raise ValueError(f"Unsupported population update method: {num_method}; supported methods are {SUPPORTED_NP_METHODS}")

    effective_min = _effective_np_min(np_min, pop.shape[1], enable_late_enhancements, np_metrics)
    state.np_g = int(np.clip(new_np, effective_min, np_max))
    row = fitness.size
    if state.np_g > row:
        n_supplement = state.np_g - row
        supplement = generate_population(pop_min, pop_max, n_supplement, evals)
        fit_supp, pen_supp, inf_flags = get_fitness_and_penalty(supplement, evals, opmwade_repair_inf=True)
        state.nfes += n_supplement + int(np.sum(inf_flags != 0))
        pop = np.vstack([pop, supplement])
        fitness = np.concatenate([fitness, fit_supp])
        penalty = np.concatenate([penalty, pen_supp])
        pop_next = np.vstack([pop_next, supplement])
        next_fitness = np.concatenate([next_fitness, fit_supp])
        next_penalty = np.concatenate([next_penalty, pen_supp])
        pop_pbest = np.vstack([pop_pbest, supplement])
        pbest_fitness = np.concatenate([pbest_fitness, fit_supp])
        pbest_penalty = np.concatenate([pbest_penalty, pen_supp])
    else:
        keep = slice(0, state.np_g)
        pop = pop[keep]
        fitness = fitness[keep]
        penalty = penalty[keep]
        pop_next = pop_next[keep]
        next_fitness = next_fitness[keep]
        next_penalty = next_penalty[keep]
        pop_pbest = pop_pbest[keep]
        pbest_fitness = pbest_fitness[keep]
        pbest_penalty = pbest_penalty[keep]

    return pop, fitness, penalty, pop_next, next_fitness, next_penalty, pop_pbest, pbest_fitness, pbest_penalty


def append_population_size_record(process: np.ndarray, state: EvalState) -> np.ndarray:
    row = np.array([[float(state.np_g), float(state.nfes)]])
    return row if process.size == 0 else np.vstack([process, row])


def append_diagnostic_record(
    diagnostics: np.ndarray,
    repeat_index: int,
    iteration_index: int,
    state: EvalState,
    best: float,
    metrics: dict[str, float],
    accept_rate: float,
    best_improved: bool,
    mean_delta_f: float,
    mean_f: float,
    mean_cr: float,
    mean_ftar: float,
    did_resample: bool,
    local_refine_attempted: bool,
    local_refine_accepted: bool,
) -> np.ndarray:
    row = np.array(
        [
            float(repeat_index),
            float(iteration_index),
            float(state.nfes),
            float(state.np_g),
            float(best),
            float(metrics.get("feasible_rate", 0.0)),
            float(accept_rate),
            float(best_improved),
            float(mean_delta_f),
            float(metrics.get("fitness_diversity", 0.0)),
            float(metrics.get("decision_diversity", 0.0)),
            float(mean_f),
            float(mean_cr),
            float(mean_ftar),
            float(did_resample),
            float(local_refine_attempted),
            float(local_refine_accepted),
        ]
    ).reshape(1, -1)
    return row if diagnostics.size == 0 else np.vstack([diagnostics, row])


def final_best_individual(pop: np.ndarray, fitness: np.ndarray, penalty: np.ndarray, evals: int) -> tuple[np.ndarray, float, float]:
    return best_individual_by_feasibility(pop, fitness, penalty, evals, variant="opmwade")


def run(
    evals_range: Iterable[int] = (21,),
    repeat_num: int = 1,
    seed: int | None = None,
    max_nfes: int | None = None,
    save: bool = True,
    num_method: int = ADAPTIVE_NP_METHOD,
    ftar_method: int = CONSTRAINT_AWARE_FTAR_METHOD,
    init_data_dir: str | None = None,
    init_file: str | None = None,
    tip_mass: float | None = None,
    progress_interval: int = 0,
    progress_label: str | None = None,
    enable_late_enhancements: bool = True,
) -> list[RunResult]:
    if seed is not None:
        np.random.seed(seed)
    results: list[RunResult] = []

    for evals in evals_range:
        best_values: list[float] = []
        times: list[float] = []
        fearates: list[float] = []
        process = np.empty((0, 2))
        population_size_process = np.empty((0, 2))
        diagnostic_process = np.empty((0, DIAGNOSTIC_COLUMNS.size))
        best_individuals: list[np.ndarray] = []
        best_individual_fitness: list[float] = []
        best_individual_penalty: list[float] = []

        for repeat_index in range(1, repeat_num + 1):
            start = timed()
            set_problem21_tip_mass(DEFAULT_PROBLEM21_TIP_MASS if tip_mass is None else tip_mass)
            configure_problem_from_init_data(evals, init_data_dir=init_data_dir or PYTHON_INIT_DIR, init_file=init_file)
            pop_max, pop_min, pop_dim = set_initial_scope(evals)
            state = EvalState(nfes=0, nfes_max=max_nfes or iteration_setting(evals, pop_dim))
            reporter = ProgressReporter("OPMWADE", evals, repeat_index, repeat_num, progress_interval, progress_label)
            np_init = max(1, int(round(INITIAL_NP_DIM_FACTOR * pop_dim)))
            np_min = max(1, int(round(MIN_NP_DIM_FACTOR * pop_dim)))
            np_max = np_init
            state.np_g = np_init

            pop = load_initial_population(
                evals,
                state.np_g,
                init_data_dir=init_data_dir or PYTHON_INIT_DIR,
                init_file=init_file,
            )
            pop_next = np.zeros_like(pop)
            fitness, penalty, inf_flags = get_fitness_and_penalty(pop, evals, opmwade_repair_inf=True)
            state.nfes += pop.shape[0] + int(np.sum(inf_flags != 0))
            pop_next_fitness = fitness.copy()
            pop_next_penalty = penalty.copy()
            pop_pbest = pop.copy()
            pbest_fitness = fitness.copy()
            pbest_penalty = penalty.copy()

            eps0 = float(np.max(penalty)) or 1.0
            pop, fitness, penalty, num_p1, num_p2, _, pop_pbest, pbest_fitness, pbest_penalty, class_labels = sort_by_constraint(
                pop, fitness, penalty, eps0, pop_pbest, pbest_fitness, pbest_penalty
            )
            pop_best = pop[0].copy()
            pop_best_fitness = float(fitness[0])
            pop_worst_fitness = float(fitness[-1])
            process = process_best_record(process, pop, fitness, penalty, state.nfes, evals, variant="opmwade")
            population_size_process = append_population_size_record(population_size_process, state)
            current_best, current_fearate = best_and_fearate(pop, fitness, penalty, evals, variant="opmwade")
            initial_best = _feasible_best(fitness, penalty)
            best_fitness_history = [initial_best] if np.isfinite(initial_best) else []
            elite_archive_pop = np.empty((0, pop_dim))
            elite_archive_fitness = np.empty(0)
            elite_archive_penalty = np.empty(0)
            elite_archive_size = max(PBEST_MIN_COUNT, int(round(ELITE_ARCHIVE_SIZE_FACTOR * np_min)))
            elite_archive_pop, elite_archive_fitness, elite_archive_penalty = update_elite_archive(
                elite_archive_pop,
                elite_archive_fitness,
                elite_archive_penalty,
                pop,
                fitness,
                penalty,
                pop_pbest,
                pbest_fitness,
                pbest_penalty,
                elite_archive_size,
            )
            last_resample_nfes = -state.nfes_max
            last_refine_nfes = -state.nfes_max
            accept_rate_history: list[float] = []
            reporter.maybe(state, best=current_best, fearate=current_fearate, extra={"iter": 0, "np": state.np_g})

            memory_len = 5
            mcr = 0.5 * np.ones(memory_len)
            mf = 0.5 * np.ones(memory_len)
            memory_index = 0
            iteration_index = 0

            while state.nfes <= state.nfes_max:
                iteration_index += 1
                scr = np.zeros(state.np_g)
                sf = np.zeros(state.np_g)
                delta_f = np.zeros(state.np_g)
                inf_u = np.zeros(state.np_g)
                f_values = np.zeros(state.np_g)
                cr_values = np.zeros(state.np_g)
                ftar_values = np.zeros(state.np_g)
                metrics = _search_metrics(state, fitness, penalty, best_fitness_history, evals, pop, pop_min, pop_max, accept_rate_history)
                eps = generate_epsilon(eps0, penalty, metrics)
                best_before_iteration = metrics["best_now"]
                role_labels = search_role_labels(fitness, penalty, metrics)
                if enable_late_enhancements and _state_enhancements_active(metrics):
                    pbest_pool = build_pbest_pool(
                        pop,
                        fitness,
                        penalty,
                        pop_pbest,
                        pbest_fitness,
                        pbest_penalty,
                        elite_archive_pop,
                        eps,
                    )
                else:
                    pbest_pool = np.empty((0, pop_dim))

                for i in range(state.np_g):
                    f, ftar, cr = mutation_and_crossover_params(
                        memory_len,
                        mcr,
                        mf,
                        fitness,
                        penalty,
                        eps,
                        ftar_method,
                        metrics,
                        enable_late_enhancements,
                    )
                    f_values[i] = f
                    cr_values[i] = cr
                    ftar_values[i] = ftar
                    v = mutation_results(f, ftar, pop_best, pop_pbest, pbest_pool, pop, i, class_labels, role_labels, evals, pop_max, pop_min, state.np_g)
                    u = crossover(pop[i], v, cr, pop_max, pop_min, evals)
                    u_fitness, u_penalty, u_inf = get_fitness_and_penalty(u, evals, opmwade_repair_inf=True)
                    inf_u[i] = u_inf[0]
                    (
                        pop_next[i],
                        pop_next_fitness[i],
                        pop_next_penalty[i],
                        scr[i],
                        sf[i],
                        delta_f[i],
                    ) = greedy_choose(
                        u,
                        u_fitness[0],
                        u_penalty[0],
                        penalty[i],
                        fitness[i],
                        f,
                        cr,
                        pop[i],
                        eps,
                        metrics,
                        enable_late_enhancements,
                    )

                state.nfes += state.np_g + int(np.sum(inf_u != 0))
                accepted_mask = scr != -1
                accept_rate = float(np.mean(accepted_mask)) if accepted_mask.size else 0.0
                accept_rate_history.append(accept_rate)
                accepted_delta = delta_f[accepted_mask]
                mean_delta_f = float(np.mean(accepted_delta)) if accepted_delta.size else 0.0
                mean_f = float(np.mean(f_values)) if f_values.size else 0.0
                mean_cr = float(np.mean(cr_values)) if cr_values.size else 0.0
                mean_ftar = float(np.mean(ftar_values)) if ftar_values.size else 0.0
                scr = scr[accepted_mask]
                sf = sf[accepted_mask]
                delta_f = delta_f[accepted_mask]

                pop = pop_next.copy()
                fitness = pop_next_fitness.copy()
                penalty = pop_next_penalty.copy()
                sort_metrics = _search_metrics(state, fitness, penalty, best_fitness_history, evals, pop, pop_min, pop_max, accept_rate_history)
                pop, fitness, penalty, num_p1, num_p2, _, pop_pbest, pbest_fitness, pbest_penalty, class_labels = sort_by_constraint(
                    pop, fitness, penalty, eps0, pop_pbest, pbest_fitness, pbest_penalty, sort_metrics
                )
                pop_best = pop[0].copy()
                pop_best_fitness = float(fitness[0])
                pop_worst_fitness = float(fitness[-1])
                eps = generate_epsilon(eps0, penalty, sort_metrics)
                pop_pbest, pbest_fitness, pbest_penalty = update_pop_pbest(pop, fitness, penalty, pop_pbest, pbest_fitness, pbest_penalty, eps)
                elite_archive_pop, elite_archive_fitness, elite_archive_penalty = update_elite_archive(
                    elite_archive_pop,
                    elite_archive_fitness,
                    elite_archive_penalty,
                    pop,
                    fitness,
                    penalty,
                    pop_pbest,
                    pbest_fitness,
                    pbest_penalty,
                    elite_archive_size,
                )
                process = process_best_record(process, pop, fitness, penalty, state.nfes, evals, variant="opmwade")
                memory_index, mcr, mf = update_mcr_and_mf(memory_index, memory_len, mcr, mf, scr, sf, delta_f)

                pop, fitness, penalty, pop_next, pop_next_fitness, pop_next_penalty, pop_pbest, pbest_fitness, pbest_penalty = num_pop_update(
                    num_method,
                    state,
                    pop,
                    fitness,
                    pop_best_fitness,
                    pop_worst_fitness,
                    np_max,
                    np_min,
                    pop_max,
                    pop_min,
                    pop_best,
                    pop_next,
                    evals,
                    penalty,
                    pop_next_fitness,
                    pop_next_penalty,
                    pop_pbest,
                    pbest_fitness,
                    pbest_penalty,
                    best_fitness_history,
                    accept_rate_history,
                    enable_late_enhancements,
                )
                did_resample = False
                resample_metrics = _search_metrics(state, fitness, penalty, best_fitness_history, evals, pop, pop_min, pop_max, accept_rate_history)
                if enable_late_enhancements and should_resample_stagnated_tail(resample_metrics, state, last_resample_nfes, elite_archive_pop):
                    (
                        pop,
                        fitness,
                        penalty,
                        pop_next,
                        pop_next_fitness,
                        pop_next_penalty,
                        pop_pbest,
                        pbest_fitness,
                        pbest_penalty,
                        did_resample,
                    ) = resample_stagnated_tail(
                        pop,
                        fitness,
                        penalty,
                        pop_next,
                        pop_next_fitness,
                        pop_next_penalty,
                        pop_pbest,
                        pbest_fitness,
                        pbest_penalty,
                        elite_archive_pop,
                        pop_max,
                        pop_min,
                        evals,
                        state,
                        resample_metrics,
                    )
                    if did_resample:
                        last_resample_nfes = state.nfes
                did_refine = False
                attempted_refine = False
                refine_metrics = _search_metrics(state, fitness, penalty, best_fitness_history, evals, pop, pop_min, pop_max, accept_rate_history)
                eps = generate_epsilon(eps0, penalty, refine_metrics)
                if enable_late_enhancements and should_local_refine(refine_metrics, state, last_refine_nfes, elite_archive_pop):
                    attempted_refine = True
                    (
                        pop,
                        fitness,
                        penalty,
                        pop_next,
                        pop_next_fitness,
                        pop_next_penalty,
                        pop_pbest,
                        pbest_fitness,
                        pbest_penalty,
                        did_refine,
                    ) = local_refine_elite(
                        pop,
                        fitness,
                        penalty,
                        pop_next,
                        pop_next_fitness,
                        pop_next_penalty,
                        pop_pbest,
                        pbest_fitness,
                        pbest_penalty,
                        elite_archive_pop,
                        pop_max,
                        pop_min,
                        evals,
                        eps,
                        state,
                        refine_metrics,
                    )
                    if did_refine:
                        last_refine_nfes = state.nfes
                population_size_process = append_population_size_record(population_size_process, state)
                sort_metrics = _search_metrics(state, fitness, penalty, best_fitness_history, evals, pop, pop_min, pop_max, accept_rate_history)
                pop, fitness, penalty, num_p1, num_p2, _, pop_pbest, pbest_fitness, pbest_penalty, class_labels = sort_by_constraint(
                    pop, fitness, penalty, eps0, pop_pbest, pbest_fitness, pbest_penalty, sort_metrics
                )
                if did_resample or did_refine:
                    pop_pbest, pbest_fitness, pbest_penalty = update_pop_pbest(pop, fitness, penalty, pop_pbest, pbest_fitness, pbest_penalty, eps)
                    elite_archive_pop, elite_archive_fitness, elite_archive_penalty = update_elite_archive(
                        elite_archive_pop,
                        elite_archive_fitness,
                        elite_archive_penalty,
                        pop,
                        fitness,
                        penalty,
                        pop_pbest,
                        pbest_fitness,
                        pbest_penalty,
                        elite_archive_size,
                    )
                    process = process_best_record(process, pop, fitness, penalty, state.nfes, evals, variant="opmwade")
                current_best, current_fearate = best_and_fearate(pop, fitness, penalty, evals, variant="opmwade")
                end_metrics = _search_metrics(state, fitness, penalty, best_fitness_history, evals, pop, pop_min, pop_max, accept_rate_history)
                diagnostic_process = append_diagnostic_record(
                    diagnostic_process,
                    repeat_index,
                    iteration_index,
                    state,
                    current_best,
                    end_metrics,
                    accept_rate,
                    np.isfinite(best_before_iteration) and current_best < best_before_iteration - 1e-12,
                    mean_delta_f,
                    mean_f,
                    mean_cr,
                    mean_ftar,
                    did_resample,
                    attempted_refine,
                    did_refine,
                )
                reporter.maybe(
                    state,
                    best=current_best,
                    fearate=current_fearate,
                    extra={"iter": iteration_index, "np": state.np_g},
                )

            best, fearate = best_and_fearate(pop, fitness, penalty, evals, variant="opmwade")
            reporter.maybe(state, best=best, fearate=fearate, extra={"iter": iteration_index, "np": state.np_g}, force=True)
            best_x, best_x_fitness, best_x_penalty = final_best_individual(pop, fitness, penalty, evals)
            best_individuals.append(best_x)
            best_individual_fitness.append(best_x_fitness)
            best_individual_penalty.append(best_x_penalty)
            best_values.append(best)
            fearates.append(fearate)
            times.append(timed() - start)

        summary = summarize(best_values, fearates, times)
        result = RunResult(
            "OPMWADE",
            evals,
            *summary,
            process=process,
            diagnostics={
                "population_size_and_nfes": population_size_process,
                "diagnostic_columns": DIAGNOSTIC_COLUMNS,
                "diagnostic_process": diagnostic_process,
                **best_individual_diagnostics(
                    best_individuals,
                    best_individual_fitness,
                    best_individual_penalty,
                ),
            },
        )
        results.append(result)
        if save:
            row = np.zeros((21, 8))
            row[evals - 1, :] = np.array([evals, *summary])
            best_diag = result.diagnostics
            save_mat(
                WORKSPACE_ROOT / "results" / "opmwade" / f"OPMWADE-P{evals}.mat",
                everyevalBestMediMeanWorstStdFearateTime=row,
                testProcessBestFitAndNfes=process,
                testProcessPopulationSizeAndNfes=population_size_process,
                testProcessDiagnostics=diagnostic_process,
                testDiagnosticColumns=DIAGNOSTIC_COLUMNS,
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
