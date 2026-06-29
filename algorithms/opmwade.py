from __future__ import annotations

import numpy as np

from .common import (
    DEFAULT_PROBLEM21_TIP_MASS,
    EvalState,
    ProgressReporter,
    RunResult,
    WORKSPACE_ROOT,
    best_and_fearate,
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


def _cauchy(mu: float, gamma: float) -> float:
    return float(mu + gamma * np.tan(np.pi * (np.random.rand() - 0.5)))


def mutation_and_crossover_params(
    memory_len: int,
    mcr: np.ndarray,
    mf: np.ndarray,
    pop_fitness: np.ndarray,
    ftar_method: int,
    state: EvalState,
) -> tuple[float, float, float]:
    r = np.random.randint(memory_len)
    cr = float(np.random.normal(mcr[r], 0.1))
    cr = min(max(cr, 0.0), 1.0)

    f = _cauchy(float(mf[r]), 0.1)
    while f < 0:
        f = _cauchy(float(mf[r]), 0.1)
    f = min(f, 1.0)

    if ftar_method == 1:
        tar = 0.2 + np.sin(np.pi / 6 * (1 + 2 * state.nfes / state.nfes_max))
        ftar = min(tar * f, 1.0)
    elif ftar_method == 2:
        best = np.min(pop_fitness)
        worst = np.max(pop_fitness)
        ita = np.sum((pop_fitness - best) / (worst - best + 1e-8)) / max(pop_fitness.size - 1, 1)
        ftar = float(np.exp(-ita))
    elif ftar_method == 3:
        best = np.min(pop_fitness)
        worst = np.max(pop_fitness)
        ita = np.sum((pop_fitness - best) / (worst - best + 1e-8)) / pop_fitness.size
        ftar = float(1 - np.sin(ita * np.pi / 2))
    else:
        raise ValueError(f"Unsupported Ftar method: {ftar_method}")
    return f, ftar, cr


def _unique_indices(count: int, upper: int) -> np.ndarray:
    replace = upper < count
    return np.random.choice(upper, size=count, replace=replace)


def mutation_results(
    f: float,
    ftar: float,
    pop_best: np.ndarray,
    pop_pbest: np.ndarray,
    pop: np.ndarray,
    i: int,
    class_labels: np.ndarray,
    evals: int,
    pop_max: np.ndarray,
    pop_min: np.ndarray,
    np_g: int,
) -> np.ndarray:
    p1_indices = np.flatnonzero(class_labels == 1)
    p2_indices = np.flatnonzero(class_labels == 2)

    if class_labels[i] == 1:
        r1, r2 = _unique_indices(2, np_g)
        pop_p2 = pop[np.random.choice(p2_indices)] if p2_indices.size else pop[i]
        v = pop[i] + f * (pop[r1] - pop_best) + ftar * (pop[r2] - pop_p2)
    elif class_labels[i] == 2:
        r1 = np.random.randint(np_g)
        pop_p1 = pop[np.random.choice(p1_indices)] if p1_indices.size else pop[i]
        v = pop[i] + f * (pop_pbest[i] - pop[i]) + ftar * (pop[r1] - pop_p1)
    else:
        r1, r2, r3, r4 = _unique_indices(4, np_g)
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
) -> tuple[np.ndarray, float, float, float, float, float]:
    accepted = epsilon_better(u_fitness, u_penalty, person_fitness, person_penalty, eps)

    if accepted:
        return u, float(u_fitness), float(u_penalty), cr, f, float(person_fitness - u_fitness)
    return person, float(person_fitness), float(person_penalty), -1.0, -1.0, -1.0


def generate_epsilon(eps0: float, lamta: float, p: float, state: EvalState) -> float:
    cp = -(np.log10(eps0) + lamta) / np.log10(1 - p)
    if state.nfes / state.nfes_max <= p:
        return float(eps0 * (1 - state.nfes / state.nfes_max) ** cp)
    return 0.0


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
    lamta: float,
    p: float,
    pop_pbest: np.ndarray,
    pbest_fitness: np.ndarray,
    pbest_penalty: np.ndarray,
    state: EvalState,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    eps = generate_epsilon(eps0, lamta, p, state)
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
    denom = np.sum(delta_f + 1e-8)
    weights = delta_f / denom if denom != 0 else np.ones_like(delta_f) / max(delta_f.size, 1)
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
        mcr_new[k] = _weighted_lehmer(scr, delta_f)
    if sf.size:
        mf_new[k] = _weighted_lehmer(sf, delta_f)
    return (k + 1) % memory_len, mcr_new, mf_new


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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    progress = state.nfes / max(state.nfes_max, 1)
    if num_method == 1:
        new_np = round((np_max - np_min) * ef_mean ** max(state.nfes, 1) + np_min)
    elif num_method == 2:
        new_np = round((np_max - np_min) * np.random.rand() + np_min)
    elif num_method == 3:
        new_np = round((np_max - np_min) * np.random.rand() * ef_mean + np_min)
    elif num_method == 4:
        new_np = round((np_max - np_min) * (progress - 1) ** 2 + np_min)
    elif num_method == 5:
        new_np = round((np_max - np_min) * abs(progress - 1) ** (1 + ef_mean) + np_min)
    elif num_method == 6:
        new_np = round((np_max - np_min) * abs(progress - 1) ** (1 + np.exp(-ef_mean)) + np_min)
    elif num_method == 7:
        distance_sum = np.sum(np.sum((pop - pop_best) ** 2, axis=1))
        new_np = round((np_min - np_max) * progress ** (1 + np.exp(-distance_sum)) + np_max)
    else:
        raise ValueError(f"Unsupported population update method: {num_method}")

    state.np_g = int(np.clip(new_np, np_min, np_max))
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


def run(
    evals_range: Iterable[int] = (21,),
    repeat_num: int = 1,
    seed: int | None = None,
    max_nfes: int | None = None,
    save: bool = True,
    num_method: int = 6,
    ftar_method: int = 3,
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
        times: list[float] = []
        fearates: list[float] = []
        process = np.empty((0, 2))

        for repeat_index in range(1, repeat_num + 1):
            start = timed()
            set_problem21_tip_mass(DEFAULT_PROBLEM21_TIP_MASS if tip_mass is None else tip_mass)
            configure_problem_from_init_data(evals, init_data_dir=init_data_dir or PYTHON_INIT_DIR, init_file=init_file)
            pop_max, pop_min, pop_dim = set_initial_scope(evals)
            state = EvalState(nfes=0, nfes_max=max_nfes or iteration_setting(evals, pop_dim))
            reporter = ProgressReporter("OPMWADE", evals, repeat_index, repeat_num, progress_interval, progress_label)
            np_init = 10 * pop_dim if evals == 21 else 18 * pop_dim
            np_min = 10
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
            lamta = 6
            p = 0.5
            pop, fitness, penalty, num_p1, num_p2, _, pop_pbest, pbest_fitness, pbest_penalty, class_labels = sort_by_constraint(
                pop, fitness, penalty, eps0, lamta, p, pop_pbest, pbest_fitness, pbest_penalty, state
            )
            pop_best = pop[0].copy()
            pop_best_fitness = float(fitness[0])
            pop_worst_fitness = float(fitness[-1])
            process = process_best_record(process, pop, fitness, penalty, state.nfes, evals, variant="opmwade")
            current_best, current_fearate = best_and_fearate(pop, fitness, penalty, evals, variant="opmwade")
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
                eps = generate_epsilon(eps0, lamta, p, state)

                for i in range(state.np_g):
                    f, ftar, cr = mutation_and_crossover_params(memory_len, mcr, mf, fitness, ftar_method, state)
                    v = mutation_results(f, ftar, pop_best, pop_pbest, pop, i, class_labels, evals, pop_max, pop_min, state.np_g)
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
                    ) = greedy_choose(u, u_fitness[0], u_penalty[0], penalty[i], fitness[i], f, cr, pop[i], eps)

                state.nfes += state.np_g + int(np.sum(inf_u != 0))
                scr = scr[scr != -1]
                sf = sf[sf != -1]
                delta_f = delta_f[delta_f != -1]

                pop = pop_next.copy()
                fitness = pop_next_fitness.copy()
                penalty = pop_next_penalty.copy()
                pop, fitness, penalty, num_p1, num_p2, _, pop_pbest, pbest_fitness, pbest_penalty, class_labels = sort_by_constraint(
                    pop, fitness, penalty, eps0, lamta, p, pop_pbest, pbest_fitness, pbest_penalty, state
                )
                pop_best = pop[0].copy()
                pop_best_fitness = float(fitness[0])
                pop_worst_fitness = float(fitness[-1])
                eps = generate_epsilon(eps0, lamta, p, state)
                pop_pbest, pbest_fitness, pbest_penalty = update_pop_pbest(pop, fitness, penalty, pop_pbest, pbest_fitness, pbest_penalty, eps)
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
                )
                if state.nfes + state.np_g <= state.nfes_max:
                    pop, fitness, penalty, num_p1, num_p2, _, pop_pbest, pbest_fitness, pbest_penalty, class_labels = sort_by_constraint(
                        pop, fitness, penalty, eps0, lamta, p, pop_pbest, pbest_fitness, pbest_penalty, state
                    )
                current_best, current_fearate = best_and_fearate(pop, fitness, penalty, evals, variant="opmwade")
                reporter.maybe(
                    state,
                    best=current_best,
                    fearate=current_fearate,
                    extra={"iter": iteration_index, "np": state.np_g},
                )

            best, fearate = best_and_fearate(pop, fitness, penalty, evals, variant="opmwade")
            reporter.maybe(state, best=best, fearate=fearate, extra={"iter": iteration_index, "np": state.np_g}, force=True)
            best_values.append(best)
            fearates.append(fearate)
            times.append(timed() - start)

        summary = summarize(best_values, fearates, times)
        result = RunResult("OPMWADE", evals, *summary, process=process)
        results.append(result)
        if save:
            row = np.zeros((21, 8))
            row[evals - 1, :] = np.array([evals, *summary])
            save_mat(WORKSPACE_ROOT / "results" / "opmwade" / f"OPMWADE-P{evals}.mat", everyevalBestMediMeanWorstStdFearateTime=row, testProcessBestFitAndNfes=process)
    return results


if __name__ == "__main__":
    run()
