# Restricted Robust Covariance Estimation: Synthetic and Real-Data Replication

Replication study of restricted robust covariance estimation methodology.

## Overview

This project replicates and extends experiments from the restricted robust covariance estimation literature. The methodology combines Tyler's M-estimator (a robust, elliptically-equivariant scatter estimator) with linear restrictions on the regression coefficient matrix to yield estimators that are both robust to heavy-tailed errors and efficient under correct restrictions. A Stein-type shrinkage-after-selection estimator (SSE+) adaptively combines the unrestricted and restricted estimators based on a test statistic.

### The Five Experiments

| # | Name | Description |
|---|------|-------------|
| 1 | Synthetic Simulation | Monte Carlo study comparing URE, RRE, and SSE+ under Gaussian and heavy-tailed (t₆, t₃) errors across varying sample sizes, dimensions, and restriction correctness |
| 2 | Selected Restriction / Post-Selection | Evaluates the post-selection SSE+ estimator when the restriction is chosen data-adaptively from a candidate set using a model-selection criterion |
| 3 | GMANOVA | Growth curve (Generalized MANOVA) application: fits a polynomial-in-time model to repeated-measures data and estimates the error covariance robustly |
| 4 | Communities and Crime | Real-data application using the UCI Communities and Crime dataset; region indicator design matrix; compares URE vs. RRE vs. SSE+ out-of-sample log-likelihood |
| 5 | Leukemia | High-dimensional genomics application; one-hot subtype design; global probe selection; compares estimators on held-out predictive log-likelihood |

---

## Installation

### Prerequisites

- Python 3.10+

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd project

# Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate      # Linux / macOS
.venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## Repository Structure

```
project/
├── data/                   # Downloaded / generated data (not committed)
│   └── .gitkeep
├── exp/                    # Experiment runner scripts
│   ├── exp_1_synthetic.py
│   ├── exp_2_selection.py
│   ├── exp_3_gmanova.py
│   ├── exp_4_communities.py
│   └── exp_5_leukemia.py
├── results/                # Output figures and summary tables
│   └── RESULTS.md
├── src/                    # Core library modules
│   ├── data_gen.py         # Data generation utilities
│   ├── estimators.py       # URE, RRE, SSE+ estimators
│   ├── metrics.py          # Loss functions and diagnostics
│   ├── regression.py       # OLS and restricted OLS routines
│   ├── shrinkage.py        # Nonlinear shrinkage helpers
│   └── tyler.py            # Tyler M-estimator
├── tests/                  # pytest test suite
│   ├── __init__.py
│   ├── test_data_gen.py
│   ├── test_estimators.py
│   └── test_experiments.py
├── .gitignore
├── AGENTS.md
├── README.md
└── requirements.txt
```

---

## Usage

All experiment scripts are self-contained and write figures / tables to `results/`.

### Experiment 1 — Synthetic Simulation

```bash
python exp/exp_1_synthetic.py
```

Runs a Monte Carlo simulation (default: 500 replications) over a grid of configurations (`n`, `p`, `d`, `q`, error law). Outputs risk comparison tables and plots to `results/exp1_*.{csv,png}`.

### Experiment 2 — Selected Restriction / Post-Selection

```bash
python exp/exp_2_selection.py
```

Evaluates post-selection performance when the restriction matrix is chosen from a candidate set. Outputs `results/exp2_*.{csv,png}`.

### Experiment 3 — GMANOVA

```bash
python exp/exp_3_gmanova.py
```

Fits a growth-curve model to repeated-measures data and compares covariance estimators. Outputs `results/exp3_*.{csv,png}`.

### Experiment 4 — Communities and Crime

```bash
python exp/exp_4_communities.py
```

Downloads the UCI Communities and Crime dataset (falls back to synthetic data on failure), fits a multivariate regression with region indicators, and evaluates out-of-sample log-likelihood. Outputs `results/exp4_*.{csv,png}`.

### Experiment 5 — Leukemia

```bash
python exp/exp_5_leukemia.py
```

Loads the Leukemia gene-expression dataset, selects informative probes globally, fits a one-hot subtype regression, and compares held-out predictive log-likelihood. Outputs `results/exp5_*.{csv,png}`.

### Running All Experiments

```bash
for exp in exp/exp_*.py; do python "$exp"; done
```

### Running Tests

```bash
pytest tests/ -v
```

---

## Expected Results Summary

| Experiment | Key Finding |
|------------|-------------|
| Exp 1 (Synthetic) | SSE+ dominates or matches URE/RRE across all configurations; RRE gains most under correct restriction and heavy tails |
| Exp 2 (Post-Selection) | Post-selection SSE+ retains most of the efficiency gain without prior knowledge of the correct restriction |
| Exp 3 (GMANOVA) | Robust growth-curve estimates are stable under non-Gaussian errors; restricted estimator reduces variance |
| Exp 4 (Communities) | SSE+ achieves best or near-best out-of-sample log-likelihood; regional restriction plausible |
| Exp 5 (Leukemia) | High-dimensional robust estimation benefits from subtype restriction; SSE+ competitive with URE |

---

## Citation

Replication study of restricted robust covariance estimation methodology.

If you use this code, please cite the original paper on restricted robust covariance estimation (see paper reference in `AGENTS.md`).
