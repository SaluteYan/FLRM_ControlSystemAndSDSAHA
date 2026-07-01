from __future__ import annotations

from typing import Iterable

import numpy as np

from .common import (
    DEFAULT_PROBLEM21_TIP_MASS,
    EvalState,
    ProgressReporter,
    RunResult,
    WORKSPACE_ROOT,
    best_and_fearate,
    best_individual_diagnostics,
    configure_problem_from_init_data,
    enforce_problem21_coupling,
    generate_population,
    get_fitness_and_penalty,
    iteration_setting,
    load_initial_population,
    population_size,
    save_mat,
    set_initial_scope,
    set_problem21_tip_mass,
    summarize,
    timed,
)


PYTHON_INIT_DIR = WORKSPACE_ROOT / "init_data"
DEFAULT_MAX_SURROGATE_SAMPLES = 512
DEFAULT_W_INITIAL = 10
DEFAULT_W_MIN = 5
DEFAULT_W_MAX = 40


class SurrogateModel:
    def __init__(self, x: np.ndarray, y: np.ndarray, zero_when_all_y_zero: bool = False):
        self.x = np.asarray(x, dtype=float)
        self.y = np.asarray(y, dtype=float).reshape(-1)
        self.zero_when_all_y_zero = zero_when_all_y_zero
        self._phi_pinv: np.ndarray | None = None
        self.phi = self._gram(self.x, self.x)
        if zero_when_all_y_zero and np.all(self.y == 0):
            self.weights = np.zeros(self.phi.shape[1])
        else:
            self.weights = np.linalg.lstsq(self.phi, self.y, rcond=None)[0]
        design = np.column_stack([np.ones(self.x.shape[0]), self.x])
        self.beta = np.linalg.lstsq(design, self.y, rcond=None)[0]
        if np.any(~np.isfinite(self.weights)):
            self.weights = np.zeros(self.phi.shape[1])
        if np.any(~np.isfinite(self.beta)):
            self.beta = np.zeros(self.x.shape[1] + 1)

    @staticmethod
    def _gram(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        diff = a[:, None, :] - b[None, :, :]
        return np.linalg.norm(diff, axis=2) ** 3

    @staticmethod
    def _as_2d(x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=float)
        if arr.ndim == 1:
            return arr.reshape(1, -1)
        return arr

    def predict(self, x: np.ndarray) -> float:
        return float(self.predict_many(x).reshape(-1)[0])

    def predict_many(self, x: np.ndarray) -> np.ndarray:
        xx = self._as_2d(x)
        phi_x = self._gram(xx, self.x)
        design = np.column_stack([np.ones(xx.shape[0]), xx])
        return phi_x @ self.weights + design @ self.beta

    def uncertainty(self, x: np.ndarray) -> float:
        return float(self.uncertainty_many(x).reshape(-1)[0])

    def uncertainty_many(self, x: np.ndarray) -> np.ndarray:
        xx = self._as_2d(x)
        phi_x = self._gram(xx, self.x)
        try:
            if self._phi_pinv is None:
                self._phi_pinv = np.linalg.pinv(self.phi)
            values = -np.sum((phi_x @ self._phi_pinv) * phi_x, axis=1)
        except np.linalg.LinAlgError:
            values = np.zeros(xx.shape[0])
        values = np.where(np.isfinite(values) & (values > 0), values, 0.0)
        return values


def feasibility_rule_better(
    fitness_a: float,
    violation_a: float,
    fitness_b: float,
    violation_b: float,
) -> bool:
    """Return True when solution A is better than B by the feasibility rule."""
    a_feasible = violation_a <= 0
    b_feasible = violation_b <= 0
    if a_feasible and b_feasible:
        return fitness_a < fitness_b
    if a_feasible and not b_feasible:
        return True
    if not a_feasible and b_feasible:
        return False
    return violation_a < violation_b


def feasibility_rule_best_index(fitness: np.ndarray, violation: np.ndarray) -> int:
    feasible = violation <= 0
    if np.any(feasible):
        feasible_indices = np.where(feasible)[0]
        return int(feasible_indices[np.argmin(fitness[feasible_indices])])
    return int(np.argmin(violation))


def influence_scores(model: SurrogateModel, x: np.ndarray) -> np.ndarray:
    """Approximate the per-sample influence term I_i from the paper."""
    phi = np.linalg.norm(model.x - np.asarray(x, dtype=float).reshape(1, -1), axis=1) ** 3
    return model.weights * phi


def limit_training_samples(
    samples: np.ndarray,
    values: np.ndarray,
    max_samples: int | None,
    elite_fraction: float = 0.25,
) -> tuple[np.ndarray, np.ndarray]:
    if max_samples is None or max_samples <= 0 or samples.shape[0] <= max_samples:
        return samples, values

    max_samples = int(max_samples)
    finite_indices = np.where(np.isfinite(values))[0]
    elite_count = min(int(max_samples * elite_fraction), finite_indices.size)
    if elite_count > 0:
        elite_indices = finite_indices[np.argsort(values[finite_indices])[:elite_count]]
    else:
        elite_indices = np.empty(0, dtype=int)

    recent_count = max_samples - elite_indices.size
    recent_start = max(0, samples.shape[0] - recent_count)
    recent_indices = np.arange(recent_start, samples.shape[0], dtype=int)
    keep_indices = np.unique(np.concatenate([elite_indices, recent_indices]))

    if keep_indices.size < max_samples:
        keep_set = set(int(idx) for idx in keep_indices)
        fill: list[int] = []
        for idx in range(samples.shape[0] - 1, -1, -1):
            if idx in keep_set:
                continue
            fill.append(idx)
            if len(fill) >= max_samples - keep_indices.size:
                break
        keep_indices = np.unique(np.concatenate([keep_indices, np.array(fill, dtype=int)]))

    return samples[keep_indices], values[keep_indices]


def get_epsilon(nfes: int, nfes_max: int, p: float, eps_init: float) -> float:
    if eps_init <= 0:
        return 0.0
    if nfes / nfes_max <= p:
        cp = -np.log10(eps_init + 6) / np.log(1 - p)
        return float(eps_init * (1 - nfes / nfes_max) ** cp)
    return 0.0


def binomial_crossover(v: np.ndarray, person: np.ndarray, cr: float) -> np.ndarray:
    out = np.zeros_like(v)
    rand_j = np.random.randint(v.size)
    for i in range(v.size):
        out[i] = v[i] if np.random.rand() < cr or i == rand_j else person[i]
    return out


def mutation_and_crossover(
    pop: np.ndarray,
    f_pool: np.ndarray,
    cr_pool: np.ndarray,
    person: np.ndarray,
    gbest: np.ndarray,
    fbest: np.ndarray,
    pop_max: np.ndarray,
    pop_min: np.ndarray,
    evals: int,
    person_idx: int,
) -> np.ndarray:
    rows, cols = pop.shape
    u = np.zeros((3, cols))
    available = np.array([idx for idx in range(rows) if idx != person_idx], dtype=int)
    if available.size == 0:
        available = np.arange(rows)
    rand_idx = np.random.choice(available, size=4, replace=available.size < 4)

    f = float(np.random.choice(f_pool))
    v1 = person + np.random.rand() * (pop[rand_idx[0]] - person) + f * (pop[rand_idx[1]] - pop[rand_idx[2]])
    u[0] = np.minimum(np.maximum(v1, pop_min), pop_max)

    f = float(np.random.choice(f_pool))
    cr = float(np.random.choice(cr_pool))
    v2 = pop[rand_idx[0]] + f * (gbest - pop[rand_idx[1]]) + f * (pop[rand_idx[2]] - pop[rand_idx[3]])
    u[1] = np.minimum(np.maximum(binomial_crossover(v2, person, cr), pop_min), pop_max)

    f = float(np.random.choice(f_pool))
    cr = float(np.random.choice(cr_pool))
    v3 = person + f * (fbest - person) + f * (pop[rand_idx[0]] - pop[rand_idx[1]])
    v3 = np.minimum(np.maximum(v3, pop_min), pop_max)
    u[2] = np.minimum(np.maximum(binomial_crossover(v3, person, cr), pop_min), pop_max)

    return enforce_problem21_coupling(u) if evals == 21 else u


def epsilon_select(
    u_best: np.ndarray,
    u_best_fitness: float,
    u_best_penalty: float,
    person: np.ndarray,
    person_fitness: float,
    person_penalty: float,
    eps: float,
) -> tuple[np.ndarray, float, float]:
    if u_best_penalty <= eps and person_penalty <= eps:
        if u_best_fitness < person_fitness:
            return u_best, u_best_fitness, u_best_penalty
        return person, person_fitness, person_penalty
    if u_best_penalty == person_penalty:
        if u_best_fitness < person_fitness:
            return u_best, u_best_fitness, u_best_penalty
        return person, person_fitness, person_penalty
    if u_best_penalty < person_penalty:
        return u_best, u_best_fitness, u_best_penalty
    return person, person_fitness, person_penalty


def restart_scheme(
    pop: np.ndarray,
    fitness: np.ndarray,
    penalty: np.ndarray,
    mu: float,
    pop_max: np.ndarray,
    pop_min: np.ndarray,
    evals: int,
) -> np.ndarray:
    feasible_rate = np.sum(penalty <= 0) / pop.shape[0]
    if (np.std(fitness) < mu or np.std(penalty) < mu) and feasible_rate == 0:
        return generate_population(pop_min, pop_max, pop.shape[0], evals)
    return enforce_problem21_coupling(pop) if evals == 21 else pop


def c2ode(
    pop: np.ndarray,
    w: int,
    surrogate_model: SurrogateModel,
    np_g: int,
    fitness: np.ndarray,
    penalty: np.ndarray,
    pop_max: np.ndarray,
    pop_min: np.ndarray,
    evals: int,
    surrogate_model_con: SurrogateModel,
    state: EvalState,
) -> tuple[np.ndarray, np.ndarray]:
    pop_next = np.zeros_like(pop)
    f_pool = np.array([0.6, 0.8, 1.0])
    cr_pool = np.array([0.1, 0.2, 1.0])
    eps_init = max(float(np.max(penalty)), 0.0)
    mu = 1e-8
    w_mid = int(np.floor(w / 2 + 0.5))
    pop_mid = pop.copy()
    pop_end = pop.copy()

    for w_index in range(1, int(w) + 1):
        gbest = pop[int(np.argmin(penalty))]
        fbest = pop[int(np.argmin(fitness))]
        eps = get_epsilon(state.nfes, state.nfes_max, 0.5, eps_init)
        for person_idx in range(np_g):
            u = mutation_and_crossover(pop, f_pool, cr_pool, pop[person_idx], gbest, fbest, pop_max, pop_min, evals, person_idx)
            u_fitness = surrogate_model.predict_many(u)
            u_penalty = surrogate_model_con.predict_many(u)
            best_idx = feasibility_rule_best_index(u_fitness, u_penalty)
            pop_next[person_idx], fitness[person_idx], penalty[person_idx] = epsilon_select(
                u[best_idx],
                float(u_fitness[best_idx]),
                float(u_penalty[best_idx]),
                pop[person_idx],
                float(fitness[person_idx]),
                float(penalty[person_idx]),
                eps,
            )
        pop = restart_scheme(pop_next.copy(), fitness, penalty, mu, pop_max, pop_min, evals)
        fitness = surrogate_model.predict_many(pop)
        penalty = surrogate_model_con.predict_many(pop)
        if w_index == w_mid:
            pop_mid = pop.copy()
        elif w_index == int(w):
            pop_end = pop.copy()
    return pop_end, pop_mid


def find_potentially_good_solution(candidates: np.ndarray, model: SurrogateModel, con_model: SurrogateModel) -> tuple[np.ndarray, float]:
    fitness = model.predict_many(candidates)
    penalty = con_model.predict_many(candidates)
    best_idx = feasibility_rule_best_index(fitness, penalty)
    return candidates[best_idx].copy(), float(fitness[best_idx])


def find_most_uncertain_solution(candidates: np.ndarray, model: SurrogateModel) -> tuple[np.ndarray, float]:
    uncertainties = model.uncertainty_many(candidates)
    idx = int(np.argmax(uncertainties))
    return candidates[idx].copy(), float(uncertainties[idx])


def calculate_similarity(send: np.ndarray, db: np.ndarray) -> float:
    distances = np.linalg.norm(send - db[:, :-2], axis=1)
    return float(np.min(distances))


def adaptive_infill_sampling(
    c_p_mid: np.ndarray,
    c_p_end: np.ndarray,
    db: np.ndarray,
    surrogate_model: SurrogateModel,
    surrogate_model_con: SurrogateModel,
) -> np.ndarray:
    send, _ = find_potentially_good_solution(c_p_end, surrogate_model, surrogate_model_con)
    distance = calculate_similarity(send, db)
    if np.isclose(distance, 0.0):
        su, _ = find_most_uncertain_solution(c_p_mid, surrogate_model)
        return su.reshape(1, -1)
    smid, _ = find_potentially_good_solution(c_p_mid, surrogate_model, surrogate_model_con)
    return np.vstack([smid, send])


def search_intensity_adjustment(s_fitness: np.ndarray, s_cons: np.ndarray, w: int, w_max: int, w_min: int) -> int:
    send_better = feasibility_rule_better(s_fitness[1], s_cons[1], s_fitness[0], s_cons[0])
    if send_better:
        new_w = w * 2
    else:
        new_w = int(np.floor(w * 0.5 + 0.5))
    return int(max(min(w_max, new_w), w_min))


def evaluated_best_and_fearate(db: np.ndarray) -> tuple[float, float]:
    feasible = db[:, -1] <= 0
    values = db[feasible, -2]
    best = float(np.min(values)) if values.size else float("inf")
    return best, float(np.sum(feasible) / db.shape[0])


def evaluated_best_individual(db: np.ndarray) -> tuple[np.ndarray, float, float]:
    feasible = db[:, -1] <= 0
    if np.any(feasible):
        feasible_idx = np.flatnonzero(feasible)
        idx = int(feasible_idx[np.argmin(db[feasible_idx, -2])])
    else:
        idx = int(np.argmin(db[:, -1]))
    return db[idx, :-2].copy(), float(db[idx, -2]), float(db[idx, -1])


def process_db_best_record(process: np.ndarray, db: np.ndarray, nfes: int) -> np.ndarray:
    best, _ = evaluated_best_and_fearate(db)
    if not np.isfinite(best):
        return process
    row = np.array([[best, float(nfes)]])
    if process.size == 0:
        return row
    row[0, 0] = min(row[0, 0], process[-1, 0])
    return np.vstack([process, row])


def environmental_select(
    pop: np.ndarray,
    fitness: np.ndarray,
    cons: np.ndarray,
    samples: np.ndarray,
    sample_fitness: np.ndarray,
    sample_cons: np.ndarray,
    np_g: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    candidate_pop = np.vstack([pop, samples])
    candidate_fitness = np.concatenate([fitness, sample_fitness])
    candidate_cons = np.concatenate([cons, sample_cons])
    feasible = candidate_cons <= 0
    if np.any(feasible):
        feasible_indices = np.where(feasible)[0]
        infeasible_indices = np.where(~feasible)[0]
        feasible_order = feasible_indices[np.argsort(candidate_fitness[feasible_indices])]
        infeasible_order = infeasible_indices[np.argsort(candidate_cons[infeasible_indices])]
        order = np.concatenate([feasible_order, infeasible_order])
    else:
        order = np.argsort(candidate_cons)
    selected = order[:np_g]
    return candidate_pop[selected], candidate_fitness[selected], candidate_cons[selected]


def _append_row(matrix: np.ndarray, row: np.ndarray) -> np.ndarray:
    row = row.reshape(1, -1)
    return row if matrix.size == 0 else np.vstack([matrix, row])


def _append_training_samples(
    pop_surrogate: np.ndarray,
    values: np.ndarray,
    samples: np.ndarray,
    sample_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    return np.vstack([pop_surrogate, samples]), np.concatenate([values, sample_values])


def _move_training_sample_to_drm(
    pop_con: np.ndarray,
    con_values: np.ndarray,
    candidate_mask: np.ndarray,
    influence: np.ndarray,
    choose: str,
    drm: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    candidate_indices = np.where(candidate_mask)[0]
    if candidate_indices.size == 0:
        return pop_con, con_values, drm

    candidate_influence = influence[candidate_indices]
    if choose == "max":
        local_idx = int(np.argmax(candidate_influence))
    elif choose == "min":
        local_idx = int(np.argmin(candidate_influence))
    else:
        raise ValueError(f"Unsupported influence choice: {choose}")

    remove_idx = int(candidate_indices[local_idx])
    drm = _append_row(drm, np.r_[pop_con[remove_idx], con_values[remove_idx]])
    pop_con = np.delete(pop_con, remove_idx, axis=0)
    con_values = np.delete(con_values, remove_idx)
    return pop_con, con_values, drm


def _restore_from_drm(
    pop_con: np.ndarray,
    con_values: np.ndarray,
    drm: np.ndarray,
    feasible: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if drm.size == 0:
        return pop_con, con_values, drm

    if feasible:
        candidate_indices = np.where(drm[:, -1] <= 0)[0]
    else:
        candidate_indices = np.where(drm[:, -1] > 0)[0]
    if candidate_indices.size == 0:
        return pop_con, con_values, drm

    selected_idx = int(candidate_indices[np.random.randint(candidate_indices.size)])
    selected = drm[selected_idx]
    pop_con = np.vstack([pop_con, selected[:-1]])
    con_values = np.concatenate([con_values, [selected[-1]]])
    drm = np.delete(drm, selected_idx, axis=0)
    return pop_con, con_values, drm


def data_selection(
    s: np.ndarray,
    s_fitness: np.ndarray,
    s_cons: np.ndarray,
    surrogate_model: SurrogateModel,
    surrogate_model_con: SurrogateModel,
    pop_surrogate: np.ndarray,
    pop_surrogate_fitness: np.ndarray,
    pop_surrogate_con: np.ndarray,
    pop_surrogate_con_cons: np.ndarray,
    pop: np.ndarray,
    fitness: np.ndarray,
    cons: np.ndarray,
    drm: np.ndarray,
    np_g: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if s.shape[0] == 2:
        s_end = s[1]
        s_end_cons = s_cons[1]
        s_end_pred = surrogate_model_con.predict(s_end)
        influence = influence_scores(surrogate_model_con, s_end)

        if s_end_cons > 0 and s_end_pred <= 0:
            # Case 4 in the paper: an infeasible solution is predicted as feasible.
            feasible = pop_surrogate_con_cons <= 0
            pop_surrogate_con, pop_surrogate_con_cons, drm = _move_training_sample_to_drm(
                pop_surrogate_con,
                pop_surrogate_con_cons,
                feasible,
                influence,
                "min",
                drm,
            )
            pop_surrogate_con, pop_surrogate_con_cons, drm = _restore_from_drm(
                pop_surrogate_con,
                pop_surrogate_con_cons,
                drm,
                feasible=False,
            )
        elif s_end_cons <= 0 and s_end_pred > 0:
            # Case 3 in the paper: a feasible solution is predicted as infeasible.
            infeasible = pop_surrogate_con_cons > 0
            pop_surrogate_con, pop_surrogate_con_cons, drm = _move_training_sample_to_drm(
                pop_surrogate_con,
                pop_surrogate_con_cons,
                infeasible,
                influence,
                "max",
                drm,
            )
            pop_surrogate_con, pop_surrogate_con_cons, drm = _restore_from_drm(
                pop_surrogate_con,
                pop_surrogate_con_cons,
                drm,
                feasible=True,
            )

        feasible_s = s_cons <= 0
        if np.sum(feasible_s) == 2 and s_fitness[1] > s_fitness[0] and pop_surrogate.shape[0] > 1:
            remove_idx = int(np.argmax(pop_surrogate_fitness))
            pop_surrogate = np.delete(pop_surrogate, remove_idx, axis=0)
            pop_surrogate_fitness = np.delete(pop_surrogate_fitness, remove_idx)

    pop_surrogate, pop_surrogate_fitness = _append_training_samples(pop_surrogate, pop_surrogate_fitness, s, s_fitness)
    pop_surrogate_con, pop_surrogate_con_cons = _append_training_samples(pop_surrogate_con, pop_surrogate_con_cons, s, s_cons)
    pop_new, fitness_new, cons_new = environmental_select(pop, fitness, cons, s, s_fitness, s_cons, np_g)

    return pop_surrogate, pop_surrogate_fitness, pop_surrogate_con, pop_surrogate_con_cons, pop_new, fitness_new, cons_new, drm


def run(
    evals_range: Iterable[int] = (21,),
    repeat_num: int = 1,
    seed: int | None = None,
    max_nfes: int | None = None,
    save: bool = True,
    init_data_dir: str | None = None,
    init_file: str | None = None,
    tip_mass: float | None = None,
    progress_interval: int = 0,
    progress_label: str | None = None,
    max_surrogate_samples: int | None = DEFAULT_MAX_SURROGATE_SAMPLES,
    w_initial: int = DEFAULT_W_INITIAL,
    w_min: int = DEFAULT_W_MIN,
    w_max: int = DEFAULT_W_MAX,
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
            reporter = ProgressReporter("DSI-C2oDE", evals, repeat_index, repeat_num, progress_interval, progress_label)
            if evals == 21:
                np_g, _ = population_size(evals, pop_dim)
            else:
                np_g, _ = population_size(evals, pop_dim, dsi=True)
            np_g = max(1, min(np_g, state.nfes_max))
            state.np_g = np_g
            sample_limit = None
            if max_surrogate_samples is not None and max_surrogate_samples > 0:
                sample_limit = max(int(max_surrogate_samples), np_g)

            pop = load_initial_population(
                evals,
                np_g,
                init_data_dir=init_data_dir or PYTHON_INIT_DIR,
                init_file=init_file,
            )
            pop_surrogate = pop.copy()
            pop_surrogate_con = pop.copy()
            fitness, cons, _ = get_fitness_and_penalty(pop, evals)
            state.nfes += pop.shape[0]
            pop_surrogate_fitness = fitness.copy()
            pop_surrogate_con_cons = cons.copy()
            db = np.column_stack([pop, fitness, cons])
            drm = np.empty((0, pop_dim + 1))
            process = process_db_best_record(process, db, state.nfes)
            current_best, _ = evaluated_best_and_fearate(db)
            _, current_fearate = best_and_fearate(pop, fitness, cons, evals)
            reporter.maybe(
                state,
                best=current_best,
                fearate=current_fearate,
                extra={"iter": 0, "np": state.np_g, "w": int(w_initial), "train": pop_surrogate.shape[0]},
            )

            w = max(1, int(w_initial))
            w_min = max(1, int(w_min))
            w_max = max(w_min, int(w_max))
            iteration_index = 0
            while state.nfes < state.nfes_max:
                iteration_index += 1
                pop_surrogate, pop_surrogate_fitness = limit_training_samples(
                    pop_surrogate,
                    pop_surrogate_fitness,
                    sample_limit,
                )
                pop_surrogate_con, pop_surrogate_con_cons = limit_training_samples(
                    pop_surrogate_con,
                    pop_surrogate_con_cons,
                    sample_limit,
                )
                surrogate = SurrogateModel(pop_surrogate, pop_surrogate_fitness)
                surrogate_con = SurrogateModel(pop_surrogate_con, pop_surrogate_con_cons, zero_when_all_y_zero=True)
                pop_end, pop_mid = c2ode(pop, w, surrogate, np_g, fitness.copy(), cons.copy(), pop_max, pop_min, evals, surrogate_con, state)
                s = adaptive_infill_sampling(pop_mid, pop_end, db, surrogate, surrogate_con)
                remaining = state.nfes_max - state.nfes
                if s.shape[0] > remaining:
                    s = s[-remaining:]
                s_fitness, s_cons, _ = get_fitness_and_penalty(s, evals)
                state.nfes += s.shape[0]
                db = np.vstack([db, np.column_stack([s, s_fitness, s_cons])])

                if s.shape[0] == 2:
                    w = search_intensity_adjustment(s_fitness, s_cons, w, w_max, w_min)
                (
                    pop_surrogate,
                    pop_surrogate_fitness,
                    pop_surrogate_con,
                    pop_surrogate_con_cons,
                    pop,
                    fitness,
                    cons,
                    drm,
                ) = data_selection(
                    s,
                    s_fitness,
                    s_cons,
                    surrogate,
                    surrogate_con,
                    pop_surrogate,
                    pop_surrogate_fitness,
                    pop_surrogate_con,
                    pop_surrogate_con_cons,
                    pop,
                    fitness,
                    cons,
                    drm,
                    np_g,
                )
                process = process_db_best_record(process, db, state.nfes)
                current_best, _ = evaluated_best_and_fearate(db)
                _, current_fearate = best_and_fearate(pop, fitness, cons, evals)
                reporter.maybe(
                    state,
                    best=current_best,
                    fearate=current_fearate,
                    extra={"iter": iteration_index, "np": state.np_g, "w": w, "train": pop_surrogate.shape[0]},
                )

            best, _ = evaluated_best_and_fearate(db)
            _, fearate = best_and_fearate(pop, fitness, cons, evals)
            best_x, best_x_fit, best_x_pen = evaluated_best_individual(db)
            reporter.maybe(
                state,
                best=best,
                fearate=fearate,
                extra={"iter": iteration_index, "np": state.np_g, "w": w, "train": pop_surrogate.shape[0]},
                force=True,
            )
            best_individuals.append(best_x)
            best_individual_fitness.append(best_x_fit)
            best_individual_penalty.append(best_x_pen)
            best_values.append(best)
            fearates.append(fearate)
            times.append(timed() - start)

        summary = summarize(best_values, fearates, times)
        result = RunResult("DSI-C2oDE", evals, *summary, process=process, diagnostics=best_individual_diagnostics(
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
                WORKSPACE_ROOT / "results" / "dsi_c2ode" / f"DSI-C2oDE-P{evals}.mat",
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
