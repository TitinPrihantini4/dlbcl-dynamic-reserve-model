# DLBCL Dynamic Reserve Model

## A Regime-Switching Dynamic Reserve Model Linking Relative Dose Intensity and Functional Recovery During First-Line Diffuse Large B-Cell Lymphoma Therapy

This repository contains the computational implementation and reproducibility materials for a theoretical mathematical framework describing the interaction between treatment intensity, hematologic reserve, functional recovery, disease burden, and comorbidity during first-line therapy for diffuse large B-cell lymphoma (DLBCL).

The framework uses a regime-switching dynamical system to represent longitudinal changes in physiological reserve and treatment intensity over repeated treatment cycles. The project is intended as a methodological proof-of-concept and as a foundation for future calibration and validation using longitudinal clinical data.

## Research Objectives

The model was developed to:

- represent treatment tolerance as a time-varying physiological state rather than a static baseline characteristic;
- examine interactions between relative dose intensity and evolving patient reserve;
- identify transitions between treatment-intensity regimes;
- evaluate the sensitivity of model behavior to parameter uncertainty; and
- provide a reproducible mathematical framework for future clinical calibration.

## Mathematical Framework

The model describes four interacting state variables:

- **L(t)** — disease burden;
- **H(t)** — hematologic reserve;
- **F(t)** — functional reserve;
- **D(t)** — treatment-related burden.

A Dynamic Treatment Reserve Index (DTRI) summarizes the evolving reserve state:

**DTRI(t) = sqrt(H(t)F(t)) / [1 + 0.25L(t) + 0.25C]**

where **C** represents comorbidity burden.

Treatment intensity is represented through state-dependent regimes determined by the evolving reserve index. These regime boundaries are theoretical modeling assumptions and should not be interpreted as validated clinical treatment thresholds.

## Computational Methods

The reproducibility workflow includes:

- numerical integration using the fourth-order Runge-Kutta method (RK4);
- simulation over a six-cycle treatment horizon;
- scenario-based analysis;
- reserve-state surface analysis;
- Latin hypercube sampling for parameter uncertainty;
- partial rank correlation coefficient analysis;
- stochastic perturbation analysis; and
- numerical solver robustness assessment.

## Repository Contents

| File | Description |
|---|---|
| `DLBCL_Regime_Switching_Reproducible.py` | Main Python implementation of the mathematical model |
| `DLBCL_Scenario_Results.csv` | Results from predefined treatment and reserve scenarios |
| `DLBCL_Global_Sensitivity.csv` | Global sensitivity analysis outputs |
| `DLBCL_Uncertainty_Summary.csv` | Parameter uncertainty simulation summary |
| `DLBCL_Solver_Robustness.csv` | Numerical solver robustness results |
| `requirements.txt` | Python dependencies required to run the model |
| `LICENSE` | MIT License |

The manuscript PDF will be added after completion of the current revision.

## Requirements

The computational workflow uses:

- Python
- NumPy
- pandas
- SciPy
- Matplotlib

Install the required packages with:

```bash
pip install -r requirements.txt
```

## Running the Model

Clone the repository:

```bash
git clone https://github.com/TitinPrihantini4/dlbcl-dynamic-reserve-model.git
cd dlbcl-dynamic-reserve-model
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the model:

```bash
python DLBCL_Regime_Switching_Reproducible.py
```

## Reproducibility

The computational experiments were designed to permit independent reproduction of the theoretical model behavior. The analysis includes deterministic scenario simulation, parameter sensitivity analysis, uncertainty propagation, and solver robustness evaluation.

The accompanying CSV files provide numerical outputs from the principal computational analyses.

## Research Integrity and Clinical Interpretation

This project is a **theoretical and computational proof-of-concept**.

The numerical results represent outputs generated from the specified mathematical equations, parameter assumptions, initial conditions, regime rules, and numerical procedures. They are **not patient-level observations, clinical trial results, validated treatment-effect estimates, or clinically validated treatment thresholds**.

The model has not yet been calibrated or externally validated using longitudinal patient data. Therefore, the framework should not be used to guide individual treatment decisions.

Future work should include parameter estimation, longitudinal cohort calibration, external validation, uncertainty assessment, and evaluation of clinical utility.

## Manuscript

**A Regime-Switching Dynamic Reserve Model Linking Relative Dose Intensity and Functional Recovery During First-Line Diffuse Large B-Cell Lymphoma Therapy**

Manuscript currently under revision.

## Citation

A formal citation and DOI will be added following archival release of this repository through Zenodo.

## License

The software in this repository is distributed under the MIT License. See `LICENSE` for details.
