#!/usr/bin/env python3
"""
DLBCL_Regime_Switching_Reproducible.py

Reproducibility workflow for:
"A Regime-Switching Dynamic Reserve Model Linking Relative Dose Intensity and
Functional Recovery During First-Line Diffuse Large B-Cell Lymphoma Therapy"

WHAT IS DIRECTLY SPECIFIED BY THE MANUSCRIPT / SUPPLIED CORE CODE
----------------------------------------------------------------
- Four states: lymphoma burden L, hematologic reserve H, functional reserve F,
  cumulative delivered intensity D.
- Dynamic Treatment Reserve Index (DTRI):
      sqrt(H*F) / (1 + 0.25*L + 0.25*C)
- Regimes:
      DTRI >= 0.62       -> u = 1.00
      0.42 <= DTRI<0.62 -> u = 0.75
      DTRI < 0.42        -> u = 0.55
- ODE system and reference parameter values.
- Six treatment-cycle units, RK4, base step 0.01.
- Five reserve scenarios.
- 41x41 initial-reserve surface.
- Continuation along H0 = F0 for an illustrative RDI >= 0.70 boundary.
- 500-set Latin hypercube sampling and PRCC.
- Repeated integration at fourfold coarser resolution.

IMPORTANT TRANSPARENCY NOTE
---------------------------
The manuscript and supplied short Python implementation do NOT state the exact
Latin-hypercube parameter bounds, random seed, or stochastic perturbation law
that produced the archived PRCC/uncertainty CSVs. Therefore this script does
NOT silently pretend those missing choices are known.

The LHS bounds below are explicitly labelled REPRODUCIBILITY ASSUMPTIONS.
They are editable in SENSITIVITY_RANGES. Results from the deterministic core,
five scenarios, reserve surface, continuation boundary, and solver-resolution
check are generated directly from the manuscript equations.

This script is intended to make the computational workflow auditable. Exact
reproduction of archived sensitivity statistics requires the original sampling
bounds/seed if they differed from the assumptions documented here.

Usage
-----
    python DLBCL_Regime_Switching_Reproducible.py

Optional:
    python DLBCL_Regime_Switching_Reproducible.py --output-dir outputs
    python DLBCL_Regime_Switching_Reproducible.py --seed 42
    python DLBCL_Regime_Switching_Reproducible.py --n-lhs 500

Dependencies
------------
numpy
pandas
scipy
matplotlib
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import qmc, rankdata


# ---------------------------------------------------------------------------
# 1. Model specification
# ---------------------------------------------------------------------------

BASE: Dict[str, float] = {
    "g": 0.22,   # lymphoma growth
    "K": 1.00,   # normalized carrying capacity
    "e": 0.62,   # treatment effect
    "rh": 0.18,  # hematologic recovery
    "rf": 0.15,  # functional recovery
    "th": 0.17,  # hematologic treatment toxicity
    "tf": 0.15,  # functional treatment toxicity
    "dh": 0.10,  # lymphoma-associated hematologic depletion
    "df": 0.08,  # lymphoma-associated functional depletion
    "ch": 0.04,  # comorbidity-associated hematologic pressure
    "cf": 0.05,  # comorbidity-associated functional pressure
    "C": 0.35,   # reference comorbidity pressure
}

REFERENCE_X0 = (0.75, 0.72, 0.68, 0.0)

SCENARIOS = {
    "High reserve": {
        "x0": (0.75, 0.85, 0.85, 0.0),
        "mods": {"C": 0.20},
    },
    "Reference": {
        "x0": (0.75, 0.72, 0.68, 0.0),
        "mods": {"C": 0.35},
    },
    "Borderline reserve": {
        "x0": (0.75, 0.60, 0.58, 0.0),
        "mods": {"C": 0.40},
    },
    "Low reserve": {
        "x0": (0.75, 0.55, 0.55, 0.0),
        "mods": {"C": 0.45},
    },
    "Severely constrained": {
        "x0": (0.75, 0.45, 0.45, 0.0),
        "mods": {"C": 0.55},
    },
}

# Reproducibility assumptions because exact LHS bounds are not reported in the
# supplied manuscript. Change these only with a documented reason.
SENSITIVITY_RANGES: Dict[str, Tuple[float, float]] = {
    "H0": (0.45, 0.90),
    "F0": (0.45, 0.90),
    "C":  (0.15, 0.60),
    "g":  (0.16, 0.28),
    "e":  (0.48, 0.76),
    "rh": (0.10, 0.28),
    "rf": (0.08, 0.24),
    "th": (0.09, 0.26),
    "tf": (0.08, 0.25),
    "dh": (0.04, 0.16),
    "df": (0.035, 0.125),
}

SENSITIVITY_ORDER = [
    "H0", "F0", "C", "g", "e", "rh", "rf", "th", "tf", "dh", "df"
]


def dtri(x: np.ndarray, p: Dict[str, float]) -> float:
    """Dynamic Treatment Reserve Index."""
    L, H, F, _ = x
    return np.sqrt(max(H, 1e-12) * max(F, 1e-12)) / (
        1.0 + 0.25 * max(L, 0.0) + 0.25 * p["C"]
    )




def dtri_batch(X: np.ndarray, P: Dict[str, np.ndarray]) -> np.ndarray:
    L, H, F, _ = X.T
    return np.sqrt(np.maximum(H, 1e-12) * np.maximum(F, 1e-12)) / (
        1.0 + 0.25 * np.maximum(L, 0.0) + 0.25 * P["C"]
    )


def treatment_multiplier_batch(ri: np.ndarray) -> np.ndarray:
    return np.where(ri >= 0.62, 1.0, np.where(ri >= 0.42, 0.75, 0.55))


def deriv_batch(X: np.ndarray, P: Dict[str, np.ndarray]) -> np.ndarray:
    L, H, F, D = X.T
    u = treatment_multiplier_batch(dtri_batch(X, P))
    dL = P["g"] * L * (1.0 - L / P["K"]) - P["e"] * u * L
    dH = P["rh"] * (1.0 - H) * F - P["th"] * u * H - P["dh"] * L * H - P["ch"] * P["C"] * H
    dF = P["rf"] * (1.0 - F) * H - P["tf"] * u * F - P["df"] * L * F - P["cf"] * P["C"] * F
    dD = u / 6.0
    return np.column_stack([dL, dH, dF, dD])


def simulate_batch(X0: np.ndarray, parameter_rows: Dict[str, np.ndarray], n_steps: int = 600) -> np.ndarray:
    """Vectorized RK4 simulation for many independent parameter sets."""
    X = np.asarray(X0, dtype=float).copy()
    n = X.shape[0]
    P = {}
    for key, base_value in BASE.items():
        value = parameter_rows.get(key, base_value)
        arr = np.asarray(value, dtype=float)
        if arr.ndim == 0:
            arr = np.full(n, float(arr))
        P[key] = arr
    dt = 6.0 / n_steps
    for _ in range(n_steps):
        k1 = deriv_batch(X, P)
        k2 = deriv_batch(X + dt * k1 / 2.0, P)
        k3 = deriv_batch(X + dt * k2 / 2.0, P)
        k4 = deriv_batch(X + dt * k3, P)
        X = X + dt * (k1 + 2.0*k2 + 2.0*k3 + k4) / 6.0
        X[:, :3] = np.clip(X[:, :3], 0.0, 1.0)
    return X

def treatment_multiplier(reserve_index: float) -> float:
    """Piecewise treatment-delivery multiplier."""
    if reserve_index >= 0.62:
        return 1.00
    if reserve_index >= 0.42:
        return 0.75
    return 0.55


def regime_name(reserve_index: float) -> str:
    if reserve_index >= 0.62:
        return "Preserved"
    if reserve_index >= 0.42:
        return "Compensated"
    return "Intolerant"


def deriv(x: np.ndarray, p: Dict[str, float]) -> np.ndarray:
    """Coupled ODE vector field."""
    L, H, F, D = x
    u = treatment_multiplier(dtri(x, p))

    dL = p["g"] * L * (1.0 - L / p["K"]) - p["e"] * u * L
    dH = (
        p["rh"] * (1.0 - H) * F
        - p["th"] * u * H
        - p["dh"] * L * H
        - p["ch"] * p["C"] * H
    )
    dF = (
        p["rf"] * (1.0 - F) * H
        - p["tf"] * u * F
        - p["df"] * L * F
        - p["cf"] * p["C"] * F
    )
    dD = u / 6.0

    return np.array([dL, dH, dF, dD], dtype=float)


def simulate(
    x0: Iterable[float] = REFERENCE_X0,
    mods: Dict[str, float] | None = None,
    n_steps: int = 600,
    keep_trajectory: bool = False,
):
    """
    Classical RK4 integration over six treatment-cycle units.

    L, H, F are clipped to [0,1] after each full RK4 step, matching the
    supplied core implementation. D is not clipped.
    """
    p = BASE.copy()
    p.update(mods or {})

    horizon = 6.0
    dt = horizon / n_steps
    x = np.asarray(x0, dtype=float).copy()

    if keep_trajectory:
        times = np.linspace(0.0, horizon, n_steps + 1)
        states = np.empty((n_steps + 1, 4), dtype=float)
        ris = np.empty(n_steps + 1, dtype=float)
        regimes = np.empty(n_steps + 1, dtype=object)
        states[0] = x
        ris[0] = dtri(x, p)
        regimes[0] = regime_name(ris[0])

    regime_counts = {"Preserved": 0, "Compensated": 0, "Intolerant": 0}

    for i in range(n_steps):
        ri = dtri(x, p)
        regime_counts[regime_name(ri)] += 1

        k1 = deriv(x, p)
        k2 = deriv(x + dt * k1 / 2.0, p)
        k3 = deriv(x + dt * k2 / 2.0, p)
        k4 = deriv(x + dt * k3, p)
        x = x + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        x[:3] = np.clip(x[:3], 0.0, 1.0)

        if keep_trajectory:
            states[i + 1] = x
            ris[i + 1] = dtri(x, p)
            regimes[i + 1] = regime_name(ris[i + 1])

    fractions = {k: v / n_steps for k, v in regime_counts.items()}

    result = {
        "state": x,
        "params": p,
        "initial_dtri": dtri(np.asarray(x0, dtype=float), p),
        "final_dtri": dtri(x, p),
        "fractions": fractions,
    }

    if keep_trajectory:
        result.update(
            {
                "time": times,
                "trajectory": states,
                "dtri": ris,
                "regime": regimes,
            }
        )
    return result


# ---------------------------------------------------------------------------
# 2. Deterministic scenario analysis
# ---------------------------------------------------------------------------

def scenario_analysis(output_dir: Path) -> pd.DataFrame:
    rows = []
    for name, spec in SCENARIOS.items():
        result = simulate(spec["x0"], spec["mods"], keep_trajectory=True)
        L, H, F, D = result["state"]
        fr = result["fractions"]
        rows.append(
            {
                "Scenario": name,
                "L0": spec["x0"][0],
                "H0": spec["x0"][1],
                "F0": spec["x0"][2],
                "C": result["params"]["C"],
                "Initial_RI": result["initial_dtri"],
                "Final_L": L,
                "Final_H": H,
                "Final_F": F,
                "Final_RI": result["final_dtri"],
                "Modeled_RDI": D,
                "Preserved_Fraction": fr["Preserved"],
                "Compensated_Fraction": fr["Compensated"],
                "Intolerant_Fraction": fr["Intolerant"],
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "DLBCL_Scenario_Results.csv", index=False)
    return df


def selected_trajectories():
    selected = ["High reserve", "Reference", "Low reserve", "Severely constrained"]
    return {
        name: simulate(
            SCENARIOS[name]["x0"],
            SCENARIOS[name]["mods"],
            keep_trajectory=True,
        )
        for name in selected
    }


# ---------------------------------------------------------------------------
# 3. Figures 1-4
# ---------------------------------------------------------------------------

def plot_trajectories(output_dir: Path) -> None:
    tr = selected_trajectories()

    def plot_state(index: int, ylabel: str, filename: str):
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        for name, res in tr.items():
            ax.plot(res["time"], res["trajectory"][:, index], label=name)
        ax.set_xlabel("Treatment time (cycle units)")
        ax.set_ylabel(ylabel)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=300)
        plt.close(fig)

    plot_state(0, "Normalized lymphoma burden", "Figure_1_Lymphoma_Burden.png")
    plot_state(1, "Hematologic reserve", "Figure_2_Hematologic_Reserve.png")
    plot_state(2, "Functional reserve", "Figure_3_Functional_Reserve.png")

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for name, res in tr.items():
        ax.plot(res["time"], res["dtri"], label=name)
    ax.axhline(0.62, linestyle="--")
    ax.axhline(0.42, linestyle="--")
    ax.set_xlabel("Treatment time (cycle units)")
    ax.set_ylabel("Dynamic Treatment Reserve Index")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "Figure_4_DTRI_Regime_Boundaries.png", dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4. 41x41 reserve surface and continuation boundary
# ---------------------------------------------------------------------------

def reserve_surface(output_dir: Path) -> pd.DataFrame:
    grid = np.linspace(0.40, 0.90, 41)
    rows = []
    Z = np.empty((len(grid), len(grid)), dtype=float)

    Hmesh, Fmesh = np.meshgrid(grid, grid, indexing="ij")
    X0 = np.column_stack([
        np.full(Hmesh.size, 0.75), Hmesh.ravel(), Fmesh.ravel(), np.zeros(Hmesh.size)
    ])
    Xf = simulate_batch(X0, {"C": np.full(Hmesh.size, 0.35)})
    Z = Xf[:, 3].reshape(Hmesh.shape)
    Pf = {k: np.full(Hmesh.size, v) for k, v in BASE.items()}
    Pf["C"] = np.full(Hmesh.size, 0.35)
    final_ri = dtri_batch(Xf, Pf)
    for idx, (H0, F0) in enumerate(zip(Hmesh.ravel(), Fmesh.ravel())):
        rows.append({
            "H0": H0, "F0": F0, "C": 0.35,
            "Modeled_RDI": Xf[idx, 3], "Final_L": Xf[idx, 0],
            "Final_H": Xf[idx, 1], "Final_F": Xf[idx, 2],
            "Final_DTRI": final_ri[idx],
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "DLBCL_Reserve_Surface_41x41.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.2, 5.7))
    image = ax.imshow(
        Z,
        origin="lower",
        extent=[grid.min(), grid.max(), grid.min(), grid.max()],
        aspect="auto",
    )
    cs = ax.contour(grid, grid, Z, levels=[0.60, 0.70], linewidths=1.0)
    ax.clabel(cs, inline=True, fontsize=8)
    ax.set_xlabel("Initial functional reserve")
    ax.set_ylabel("Initial hematologic reserve")
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Modeled relative dose intensity")
    fig.tight_layout()
    fig.savefig(output_dir / "Figure_5_Reserve_RDI_Surface.png", dpi=300)
    plt.close(fig)

    return df


def continuation_boundary(output_dir: Path) -> pd.DataFrame:
    """
    Fine one-dimensional continuation along H0=F0.

    The manuscript reports an illustrative RDI>=0.70 boundary near equal
    baseline reserves of approximately 0.745.
    """
    reserve = np.linspace(0.40, 0.90, 1001)
    X0 = np.column_stack([np.full(reserve.size, 0.75), reserve, reserve, np.zeros(reserve.size)])
    Xf = simulate_batch(X0, {"C": np.full(reserve.size, 0.35)})
    df = pd.DataFrame({"Equal_Reserve": reserve, "Modeled_RDI": Xf[:, 3]})
    eligible = df[df["Modeled_RDI"] >= 0.70]
    boundary = float(eligible.iloc[0]["Equal_Reserve"]) if len(eligible) else np.nan

    df["RDI70_Boundary"] = boundary
    df.to_csv(output_dir / "DLBCL_Continuation_RDI70.csv", index=False)
    return df


# ---------------------------------------------------------------------------
# 5. Latin hypercube uncertainty + PRCC
# ---------------------------------------------------------------------------

def _rank_residual(values: np.ndarray, covariates: np.ndarray) -> np.ndarray:
    y = rankdata(values)
    if covariates.size == 0:
        return y - y.mean()
    X = np.column_stack(
        [np.ones(len(values)), np.apply_along_axis(rankdata, 0, covariates)]
    )
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ beta


def partial_rank_correlations(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    out = np.empty(X.shape[1], dtype=float)
    for j in range(X.shape[1]):
        mask = [k for k in range(X.shape[1]) if k != j]
        residual_x = _rank_residual(X[:, j], X[:, mask])
        residual_y = _rank_residual(y, X[:, mask])
        out[j] = np.corrcoef(residual_x, residual_y)[0, 1]
    return out


def lhs_analysis(
    output_dir: Path,
    n_samples: int = 500,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    sampler = qmc.LatinHypercube(d=len(SENSITIVITY_ORDER), seed=seed)
    unit = sampler.random(n=n_samples)

    X = np.empty_like(unit)
    for j, name in enumerate(SENSITIVITY_ORDER):
        lo, hi = SENSITIVITY_RANGES[name]
        X[:, j] = lo + unit[:, j] * (hi - lo)

    input_df = pd.DataFrame(X, columns=SENSITIVITY_ORDER)

    X0 = np.column_stack([
        np.full(n_samples, 0.75), input_df["H0"].to_numpy(),
        input_df["F0"].to_numpy(), np.zeros(n_samples)
    ])
    parameter_rows = {k: input_df[k].to_numpy() for k in ["C","g","e","rh","rf","th","tf","dh","df"]}
    Xf = simulate_batch(X0, parameter_rows)
    Pfinal = {k: np.full(n_samples, v) for k, v in BASE.items()}
    for k, v in parameter_rows.items(): Pfinal[k] = v
    final_ri = dtri_batch(Xf, Pfinal)
    output_df = pd.DataFrame({
        "Modeled_RDI": Xf[:,3],
        "Final_Lymphoma_Burden": Xf[:,0],
        "Final_Reserve_Index": final_ri,
    })
    full_df = pd.concat([input_df, output_df], axis=1)
    full_df.to_csv(output_dir / "DLBCL_LHS_500_Full.csv", index=False)

    X_np = input_df.to_numpy()
    prcc_rdi = partial_rank_correlations(
        X_np, output_df["Modeled_RDI"].to_numpy()
    )
    prcc_l = partial_rank_correlations(
        X_np, output_df["Final_Lymphoma_Burden"].to_numpy()
    )

    sensitivity = pd.DataFrame(
        {
            "Parameter": SENSITIVITY_ORDER,
            "PRCC_RDI": prcc_rdi,
            "PRCC_Final_Lymphoma_Burden": prcc_l,
            "Abs": np.abs(prcc_rdi),
        }
    ).sort_values("Abs", ascending=False)
    sensitivity.to_csv(output_dir / "DLBCL_Global_Sensitivity.csv", index=False)

    def summary_row(label: str, series: pd.Series):
        return {
            "Outcome": label,
            "Median": float(series.median()),
            "P05": float(series.quantile(0.05)),
            "P95": float(series.quantile(0.95)),
        }

    uncertainty = pd.DataFrame(
        [
            summary_row("Modeled RDI", output_df["Modeled_RDI"]),
            summary_row(
                "Final lymphoma burden",
                output_df["Final_Lymphoma_Burden"],
            ),
            summary_row(
                "Final reserve index",
                output_df["Final_Reserve_Index"],
            ),
        ]
    )
    uncertainty.to_csv(output_dir / "DLBCL_Uncertainty_Summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    plot_df = sensitivity.sort_values("PRCC_RDI")
    ax.barh(plot_df["Parameter"], plot_df["PRCC_RDI"])
    ax.axvline(0.0, linewidth=0.8)
    ax.set_xlabel("PRCC with modeled relative dose intensity")
    fig.tight_layout()
    fig.savefig(output_dir / "Figure_6_Global_Sensitivity_RDI.png", dpi=300)
    plt.close(fig)

    return full_df, sensitivity, uncertainty


# ---------------------------------------------------------------------------
# 6. Solver-resolution robustness
# ---------------------------------------------------------------------------

def solver_robustness(output_dir: Path) -> pd.DataFrame:
    strict = simulate(REFERENCE_X0, {"C": 0.35}, n_steps=600)["state"]
    relaxed = simulate(REFERENCE_X0, {"C": 0.35}, n_steps=150)["state"]

    names = ["Final L", "Final H", "Final F", "Modeled RDI"]
    rows = []
    for i, name in enumerate(names):
        rows.append(
            {
                "Outcome": name,
                "Strict": strict[i],
                "Relaxed": relaxed[i],
                "Absolute_Difference": abs(strict[i] - relaxed[i]),
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "DLBCL_Solver_Robustness.csv", index=False)
    return df


# ---------------------------------------------------------------------------
# 7. Machine-readable metadata and console report
# ---------------------------------------------------------------------------

def save_metadata(output_dir: Path, seed: int, n_lhs: int) -> None:
    metadata = {
        "model": "DLBCL regime-switching dynamic reserve model",
        "status": "theoretical mechanistic proof-of-concept",
        "patient_level_fitting": False,
        "clinical_decision_tool": False,
        "integration": {
            "method": "classical fourth-order Runge-Kutta",
            "horizon_cycle_units": 6.0,
            "reference_steps": 600,
            "reference_dt": 0.01,
            "coarse_steps": 150,
            "coarse_dt": 0.04,
        },
        "regime_thresholds": {
            "preserved": "DTRI >= 0.62",
            "compensated": "0.42 <= DTRI < 0.62",
            "intolerant": "DTRI < 0.42",
        },
        "treatment_multipliers": {
            "preserved": 1.0,
            "compensated": 0.75,
            "intolerant": 0.55,
        },
        "lhs": {
            "n_samples": n_lhs,
            "seed": seed,
            "ranges_status": (
                "reproducibility assumptions; exact bounds were not specified "
                "in the supplied manuscript/core script"
            ),
            "ranges": {k: list(v) for k, v in SENSITIVITY_RANGES.items()},
        },
    }
    (output_dir / "DLBCL_Reproducibility_Metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


def print_report(
    scenarios: pd.DataFrame,
    continuation: pd.DataFrame,
    sensitivity: pd.DataFrame,
    uncertainty: pd.DataFrame,
    solver: pd.DataFrame,
    output_dir: Path,
) -> None:
    ref = scenarios.loc[scenarios["Scenario"] == "Reference"].iloc[0]
    boundary = continuation["RDI70_Boundary"].iloc[0]

    print("\nDLBCL regime-switching reproducibility workflow")
    print("=" * 55)
    print(f"Reference modeled RDI : {ref['Modeled_RDI']:.6f}")
    print(f"Reference final L     : {ref['Final_L']:.6f}")
    print(f"Reference final H     : {ref['Final_H']:.6f}")
    print(f"Reference final F     : {ref['Final_F']:.6f}")
    print(f"Reference final DTRI  : {ref['Final_RI']:.6f}")
    print(f"RDI >= 0.70 boundary  : {boundary:.4f} along H0=F0")
    print(
        "Max solver difference : "
        f"{solver['Absolute_Difference'].max():.9f}"
    )

    print("\nGenerated LHS uncertainty (using documented assumed bounds):")
    print(uncertainty.to_string(index=False))

    print("\nGenerated PRCC ranking:")
    print(
        sensitivity[
            ["Parameter", "PRCC_RDI", "PRCC_Final_Lymphoma_Burden"]
        ].to_string(index=False)
    )

    print(f"\nFiles written to: {output_dir.resolve()}")
    print(
        "\nNOTE: deterministic results follow the manuscript equations. "
        "Exact archived PRCC/uncertainty values cannot be guaranteed unless "
        "the original unreported LHS bounds and seed are known."
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Reproduce the DLBCL regime-switching computational workflow."
    )
    parser.add_argument(
        "--output-dir",
        default="DLBCL_reproducibility_outputs",
        help="Directory for generated CSV, PNG, and JSON files.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for Latin hypercube sampling.",
    )
    parser.add_argument(
        "--n-lhs",
        type=int,
        default=500,
        help="Number of Latin hypercube parameter sets.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scenarios = scenario_analysis(output_dir)
    plot_trajectories(output_dir)
    reserve_surface(output_dir)
    continuation = continuation_boundary(output_dir)
    _, sensitivity, uncertainty = lhs_analysis(
        output_dir,
        n_samples=args.n_lhs,
        seed=args.seed,
    )
    solver = solver_robustness(output_dir)
    save_metadata(output_dir, args.seed, args.n_lhs)

    print_report(
        scenarios,
        continuation,
        sensitivity,
        uncertainty,
        solver,
        output_dir,
    )


if __name__ == "__main__":
    main()
