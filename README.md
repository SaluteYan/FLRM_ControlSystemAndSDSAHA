# FLRM Control System And SDSAHA

Python implementations of the optimization algorithms used for the flexible link manipulator control experiments.

Included algorithms:

- DSI-C2oDE
- EDA++
- OPMWADE
- RND
- TPDE

The repository includes the Python source code, required Problem 21 initialization data, and the requested experiment runner. Runtime outputs are written to `results/` and are intentionally not tracked by git.

## Environment

Create and activate a Python environment, then install the dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

On Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

With the existing conda environment on this machine:

```bash
conda run -n algorithm_py_env python -m pip install -r requirements.txt
```

## Run One Algorithm

Run OPMWADE on Problem 21 with adaptive damping, target angle 1.05 rad, and tip mass 9.78 g:

```bash
python run_algorithm.py --algorithm opmwade --evals 21 --damping-mode adaptive --target-angle 1.05 --tip-mass 0.00978
```

Run all supported algorithms on Problem 21:

```bash
python run_algorithm.py --algorithm all --evals 21 --damping-mode adaptive --target-angle 1.05 --tip-mass 0.00978
```

Use `--max-nfes` for a quick smoke test:

```bash
python run_algorithm.py --algorithm opmwade --evals 21 --damping-mode adaptive --target-angle 1.05 --tip-mass 0.00978 --max-nfes 1 --no-save
```

OPMWADE defaults to population update method `8`, which keeps the population size unchanged. You can pass it explicitly when running comparison checks:

```bash
python run_algorithm.py --algorithm opmwade --evals 21 --damping-mode adaptive --target-angle 1.05 --tip-mass 0.00978 --opmwade-num-method 8
```

DSI-C2oDE uses a surrogate model and can become expensive when every evaluated
sample is retained. The default run keeps the standard Problem 21 population
size, bounds the surrogate archive, and caps the search intensity:

```bash
python run_algorithm.py --algorithm dsi-c2ode --evals 21 --damping-mode adaptive --target-angle 1.05 --tip-mass 0.00978 --dsi-max-surrogate-samples 512 --dsi-w-max 40
```

Use `--dsi-max-surrogate-samples 0 --dsi-w-max 80` to move closer to the previous slower settings.

## Run Requested Experiments

Print the scheduled experiment list without running:

```bash
python run_requested_experiments.py --dry-run
```

Run the full requested experiment batch:

```bash
python run_requested_experiments.py
```

Run the full requested experiment batch in `algorithm_py_env` using multiple CPU processes:

```bash
conda run --no-capture-output -n algorithm_py_env python run_requested_experiments.py --workers 0
```

`--workers 0` uses one process per unique optimization run, capped by CPU count. To avoid oversubscribing BLAS/OpenMP threads inside each process, parallel runs set common inner thread environment variables to `1` by default; use `--inner-threads 0` to keep the current environment unchanged, or choose an explicit per-worker value.

Progress is printed every 500 function evaluations by default. Adjust it with `--progress-interval`, or disable it with `--progress-interval 0`:

```bash
conda run --no-capture-output -n algorithm_py_env python run_requested_experiments.py --workers 0 --progress-interval 1000
```

The requested experiment runner exposes the same DSI-C2oDE controls:

```bash
conda run --no-capture-output -n algorithm_py_env python run_requested_experiments.py --workers 0 --dsi-max-surrogate-samples 512 --dsi-w-max 40
```

`--max-nfes` limits real objective evaluations. For RND this includes the hidden finite-difference objective calls used for numerical gradients and Hessian terms.

The script writes:

- `results/requested_experiments/<timestamp>/summary.csv`
- `results/requested_experiments/<timestamp>/summary.json`
- `results/requested_experiments/<timestamp>/process/*.npz`

## Initialization Data

Problem 21 initialization files are stored in `init_data/`. The `none`, `fixed`, and `adaptive` data files for the same target angle are generated from a shared 16-dimensional master population, so shared variables are identical across damping modes.

Regenerate matched initialization data for one target angle:

```bash
python init_data/generate_init_data.py --evals 21 --seed 1 --target-angle 1.05 --all-damping-modes
```

Regenerate all three target angles used by the requested experiments:

```bash
python init_data/generate_init_data.py --evals 21 --seed 1 --target-angle 1.05 --all-damping-modes
python init_data/generate_init_data.py --evals 21 --seed 1 --target-angle 1.57 --all-damping-modes
python init_data/generate_init_data.py --evals 21 --seed 1 --target-angle 2.09 --all-damping-modes
```
