from __future__ import annotations

from typing import Iterable

import numpy as np
from sklearn.cluster import KMeans

from .common import (
    DEFAULT_PROBLEM21_TIP_MASS,
    EvalState,
    ProgressReporter,
    RunResult,
    WORKSPACE_ROOT,
    best_and_fearate,
    best_individual_by_feasibility,
    best_individual_diagnostics,
    configure_problem_from_init_data,
    enforce_problem21_coupling,
    fearate_calculate,
    generate_population,
    get_fitness_and_penalty,
    iteration_setting,
    load_initial_population,
    population_size,
    process_best_record,
    save_mat,
    set_initial_scope,
    set_problem21_tip_mass,
    summarize,
    timed,
)


PYTHON_INIT_DIR = WORKSPACE_ROOT / "init_data"


def _matlab_round(value: float) -> int:
    return int(np.floor(value + 0.5))


def selection(pop: np.ndarray, tar: float, penalty: np.ndarray) -> tuple[np.ndarray, int, np.ndarray]:
    rank = np.argsort(penalty)
    selected_size = max(1, _matlab_round(pop.shape[0] * tar))
    selected = rank[:selected_size]
    return pop[selected], selected_size, penalty[selected]


def generate_new_pop(pop_min: np.ndarray, pop_max: np.ndarray, evals: int, n: int) -> np.ndarray:
    return generate_population(pop_min, pop_max, n, evals)


def repairing(pop_sampled: np.ndarray, pop_max: np.ndarray, pop_min: np.ndarray, evals: int) -> np.ndarray:
    pop_rep = np.minimum(np.maximum(pop_sampled, pop_min), pop_max)
    return enforce_problem21_coupling(pop_rep) if evals == 21 else pop_rep


def _regularized_covariance(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    dim = points.shape[1]
    if points.shape[0] > 1:
        sigma = np.atleast_2d(np.cov(points, rowvar=False))
    else:
        sigma = np.eye(dim)
    if sigma.shape == (1, 1) and dim > 1:
        sigma = np.eye(dim) * float(sigma[0, 0])
    sigma = np.nan_to_num(sigma, nan=0.0, posinf=0.0, neginf=0.0)
    sigma = (sigma + sigma.T) / 2
    return sigma + 1e-10 * np.eye(dim)


def _make_component(mu: np.ndarray, points: np.ndarray, values: np.ndarray, kind: str = "parent") -> dict[str, np.ndarray | str]:
    return {
        "mu": np.asarray(mu, dtype=float).reshape(-1),
        "x_sel": np.asarray(points, dtype=float),
        "f_sel": np.asarray(values, dtype=float).reshape(-1),
        "sigma": _regularized_covariance(np.asarray(points, dtype=float)),
        "kind": kind,
    }


def learning(
    x_sel: np.ndarray,
    f_sel: np.ndarray,
    alpha: float,
    lamda: float,
    evals: int,
    state: EvalState,
) -> tuple[list[dict[str, np.ndarray | str]], list[dict[str, np.ndarray | str]]]:
    n_sel = x_sel.shape[0]
    if n_sel == 0:
        return [], []

    labels = np.zeros(n_sel, dtype=int)
    centers = x_sel[[0], :].copy()
    for k in range(1, n_sel + 1):
        kmeans = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=None)
        labels = kmeans.fit_predict(x_sel)
        centers = kmeans.cluster_centers_
        _, c_mu, _ = get_fitness_and_penalty(centers, evals, state=state, count=True)
        if np.max(c_mu) == 0:
            break

    phi_big: list[dict[str, np.ndarray | str]] = []
    for k in range(centers.shape[0]):
        mask = labels == k
        if np.any(mask):
            phi_big.append(_make_component(centers[k], x_sel[mask], f_sel[mask], kind="parent"))

    phi: list[dict[str, np.ndarray | str]] = []
    for item in phi_big:
        x_hat = np.asarray(item["x_sel"], dtype=float)
        f_hat = np.asarray(item["f_sel"], dtype=float)
        mu_hat = np.asarray(item["mu"], dtype=float)
        sigma_hat = np.asarray(item["sigma"], dtype=float)
        top_count = max(1, _matlab_round(x_hat.shape[0] * alpha))
        top_indices = np.argsort(f_hat)[:top_count]
        max_eigenvalue = max(float(np.max(np.linalg.eigvalsh(sigma_hat))), 1e-12)
        scale = np.sqrt(max_eigenvalue)

        for idx in top_indices:
            outlier = x_hat[idx]
            z_score = np.linalg.norm(outlier - mu_hat) / scale
            if z_score > lamda:
                diff = np.maximum(np.abs(outlier - mu_hat) / 2.0, 1e-8)
                phi.append({
                    "mu": outlier.copy(),
                    "x_sel": outlier.reshape(1, -1),
                    "f_sel": np.array([f_hat[idx]]),
                    "sigma": np.diag(diff ** 2),
                    "kind": "outlier",
                })
    return phi_big, phi


def sampling(
    phi_big: list[dict[str, np.ndarray | str]],
    phi: list[dict[str, np.ndarray | str]],
    n: int,
) -> np.ndarray:
    components = [*phi_big, *phi]
    if not components:
        raise ValueError("Cannot sample from an empty Phi set")
    samples: list[np.ndarray] = []
    per_cluster = int(np.ceil(n / len(components)))
    for item in components:
        mu = np.asarray(item["mu"], dtype=float)
        sigma = np.asarray(item["sigma"], dtype=float)
        samples.append(np.random.multivariate_normal(mu, sigma, per_cluster))
    return np.vstack(samples)[:n]


def mapping(
    pop_rep: np.ndarray,
    evals: int,
    n_delta: int,
    phi_big: list[dict[str, np.ndarray | str]],
    phi: list[dict[str, np.ndarray | str]],
    pop_min: np.ndarray,
    pop_max: np.ndarray,
    state: EvalState,
    map_type: str = "LD",
) -> np.ndarray:
    components = [*phi_big, *phi]
    if not components:
        return pop_rep

    centroids = np.vstack([np.asarray(item["mu"], dtype=float) for item in components])
    pop_map = repairing(pop_rep.copy(), pop_max, pop_min, evals)
    _, penalties, _ = get_fitness_and_penalty(pop_map, evals, state=state, count=True)

    for row_index in np.where(penalties > 0)[0]:
        start = pop_map[row_index].copy()
        current = start.copy()
        distances = np.linalg.norm(centroids - start, axis=1)
        centroid = centroids[int(np.argmin(distances))]
        linear_delta = (centroid - start) / max(n_delta, 1)

        for _ in range(max(n_delta, 1)):
            if map_type.upper() == "LD":
                candidate = current + linear_delta
            elif map_type.upper() == "LS":
                candidate = current + np.random.rand() * linear_delta
            elif map_type.upper() == "BD":
                candidate = current + (centroid - current) / 2.0
            elif map_type.upper() == "BS":
                midpoint = current + (centroid - current) / 2.0
                radius = np.random.rand() * np.linalg.norm(centroid - current) / 2.0
                direction = np.random.normal(size=current.size)
                norm = np.linalg.norm(direction)
                direction = direction / norm if norm > 0 else direction
                candidate = midpoint + radius * direction
            else:
                raise ValueError(f"Unsupported EDA++ mapping type: {map_type}")

            current = repairing(candidate, pop_max, pop_min, evals).reshape(-1)
            _, penalty_now, _ = get_fitness_and_penalty(current, evals, state=state, count=True)
            if penalty_now[0] <= 0:
                break
        pop_map[row_index] = current

    return pop_map


def replacement(
    x_rep: np.ndarray,
    f_rep: np.ndarray,
    c_rep: np.ndarray,
    x: np.ndarray,
    f: np.ndarray,
    c: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_combined = np.vstack([x, x_rep])
    f_combined = np.concatenate([f, f_rep])
    c_combined = np.concatenate([c, c_rep])
    keep = min(x.shape[0], x_rep.shape[0])
    selected = np.argsort(f_combined)[:keep]
    return x_combined[selected], f_combined[selected], c_combined[selected]


def update_gbest(
    pop: np.ndarray,
    fitness: np.ndarray,
    penalty: np.ndarray,
    pop_gbest: np.ndarray | None,
    pop_gbest_fitness: float | None,
    evals: int,
) -> tuple[np.ndarray | None, float | None]:
    feasible = fearate_calculate(pop, evals, penalty, variant="standard") > 0
    if not np.any(feasible):
        return pop_gbest, pop_gbest_fitness
    feasible_fitness = fitness[feasible]
    feasible_pop = pop[feasible]
    rank = int(np.argmin(feasible_fitness))
    if pop_gbest is None or pop_gbest_fitness is None or feasible_fitness[rank] < pop_gbest_fitness:
        return feasible_pop[rank].copy(), float(feasible_fitness[rank])
    return pop_gbest, pop_gbest_fitness


def seeding(
    evals: int,
    pop_max: np.ndarray,
    pop_min: np.ndarray,
    tar: float,
    np_g: int,
    gama: float,
    lamda: float,
    alpha: float,
    state: EvalState,
    init_data_dir: str | None = None,
    init_file: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    retry = False
    pop = load_initial_population(
        evals,
        np_g,
        init_data_dir=init_data_dir,
        init_file=init_file,
    )
    fitness = np.empty(pop.shape[0])
    penalty = np.empty(pop.shape[0])

    while state.nfes < state.nfes_max:
        if state.nfes == 0:
            pop = load_initial_population(
                evals,
                np_g,
                init_data_dir=init_data_dir,
                init_file=init_file,
            )
            fitness, penalty, _ = get_fitness_and_penalty(pop, evals, state=state, count=True)
        elif retry:
            retry = False
            pop_selected, selected_size, _ = selection(pop, tar, penalty)
            supplement = generate_new_pop(pop_min, pop_max, evals, np_g - selected_size)
            pop = np.vstack([pop_selected, supplement])
            fitness, penalty, _ = get_fitness_and_penalty(pop, evals, state=state, count=True)
        else:
            pop_selected, _, selected_penalty = selection(pop, gama, penalty)
            phi_big, phi = learning(pop_selected, selected_penalty, alpha, lamda, evals, state)
            pop_sampled = sampling(phi_big, phi, np_g)
            pop_rep = repairing(pop_sampled, pop_max, pop_min, evals)
            fit_rep, pen_rep, _ = get_fitness_and_penalty(pop_rep, evals, state=state, count=True)
            pop, fitness, penalty = replacement(pop_rep, fit_rep, pen_rep, pop, fitness, penalty)

        if np.max(penalty) == 0 or state.nfes > state.nfes_max / 4:
            break
        if state.nfes % 100 == 0:
            retry = True
    return pop, fitness, penalty


def run(
    evals_range: Iterable[int] = (21,),
    repeat_num: int = 1,
    seed: int | None = None,
    max_nfes: int | None = None,
    save: bool = True,
    map_type: str = "LD",
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
            reporter = ProgressReporter("EDA++", evals, repeat_index, repeat_num, progress_interval, progress_label)
            np_g, np_base = population_size(evals, pop_dim)
            state.np_g = np_g

            tar = 0.2
            gama = 0.5
            lamda = 1.0
            alpha = 0.01
            n_delta = 10

            pop, fitness, penalty = seeding(
                evals,
                pop_max,
                pop_min,
                tar,
                np_g,
                gama,
                lamda,
                alpha,
                state,
                init_data_dir=init_data_dir or PYTHON_INIT_DIR,
                init_file=init_file,
            )
            pop_gbest: np.ndarray | None = None
            pop_gbest_fitness: float | None = None
            pop_gbest, pop_gbest_fitness = update_gbest(pop, fitness, penalty, pop_gbest, pop_gbest_fitness, evals)
            if pop_gbest_fitness is not None:
                process = np.vstack([process, [pop_gbest_fitness, state.nfes]])
            current_best, current_fearate = best_and_fearate(pop, fitness, penalty, evals)
            if pop_gbest_fitness is not None:
                current_best = pop_gbest_fitness
            reporter.maybe(state, best=current_best, fearate=current_fearate, extra={"iter": 0, "np": state.np_g})

            iteration_index = 0
            while state.nfes <= state.nfes_max:
                iteration_index += 1
                pop_selected, _, fitness_selected = selection(pop, gama, fitness)
                phi_big, phi = learning(pop_selected, fitness_selected, alpha, lamda, evals, state)
                pop_sampled = sampling(phi_big, phi, np_g)
                pop_rep = repairing(pop_sampled, pop_max, pop_min, evals)
                pop_map = mapping(pop_rep, evals, n_delta, phi_big, phi, pop_min, pop_max, state, map_type=map_type)
                fit_map, pen_map, _ = get_fitness_and_penalty(pop_map, evals, state=state, count=True)
                pop, fitness, penalty = replacement(pop_map, fit_map, pen_map, pop, fitness, penalty)
                pop_gbest, pop_gbest_fitness = update_gbest(pop, fitness, penalty, pop_gbest, pop_gbest_fitness, evals)
                if pop_gbest_fitness is not None:
                    process = np.vstack([process, [pop_gbest_fitness, state.nfes]])
                current_best, current_fearate = best_and_fearate(pop, fitness, penalty, evals)
                if pop_gbest_fitness is not None:
                    current_best = pop_gbest_fitness
                reporter.maybe(
                    state,
                    best=current_best,
                    fearate=current_fearate,
                    extra={"iter": iteration_index, "np": state.np_g},
                )

            best, fearate = best_and_fearate(pop, fitness, penalty, evals)
            if pop_gbest_fitness is not None:
                best = pop_gbest_fitness
                best_x = pop_gbest.copy()
                best_x_fitness, best_x_penalty, _ = get_fitness_and_penalty(best_x, evals)
                best_x_fit = float(best_x_fitness[0])
                best_x_pen = float(best_x_penalty[0])
            else:
                best_x, best_x_fit, best_x_pen = best_individual_by_feasibility(pop, fitness, penalty, evals)
            reporter.maybe(state, best=best, fearate=fearate, extra={"iter": iteration_index, "np": state.np_g}, force=True)
            best_individuals.append(best_x)
            best_individual_fitness.append(best_x_fit)
            best_individual_penalty.append(best_x_pen)
            best_values.append(best)
            fearates.append(fearate)
            times.append(timed() - start)

        summary = summarize(best_values, fearates, times)
        result = RunResult("EDA++", evals, *summary, process=process, diagnostics=best_individual_diagnostics(
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
                WORKSPACE_ROOT / "results" / "edapp" / f"EDAPP-P{evals}.mat",
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
