"""
HIT140 Assessment 3 - Member 4 (Nahidul)
LR 2.2 Modelling + Cross-model Comparison

Purpose
-------
1. Build and evaluate LR 2.2 models that predict goals scored by a team.
2. Keep the two rows from the same match in the same validation fold.
3. Compare raw-goal and log(goals + 1) target formulations.
4. Run diagnostics and coefficient inference for the full OLS model.
5. Compare the LR 2.2-derived goal difference with Member 2's LR 2.1 model.

Expected repository layout
--------------------------
repository_root/
    lr21_dataset_104.csv
    lr22_member3/lr22_dataset_208.csv
    Task_2_lr21_Md_Foysal/model_comparison.csv
    Task_4_lr22_Nahidul/lr22_modelling_nahidul.py   <- this file

Outputs are written to:
    repository_root/modelling_outputs_for_lr22_nahidul/
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

warnings.filterwarnings("ignore")
plt.rcParams["figure.dpi"] = 140
plt.rcParams["font.family"] = "DejaVu Sans"

# -----------------------------------------------------------------------------
# PATHS
# -----------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
LR22_PATH = ROOT / "lr22_member3" / "lr22_dataset_208.csv"
LR21_PATH = ROOT / "lr21_dataset_104.csv"
LR21_RESULTS_PATH = ROOT / "Task_2_lr21_Md_Foysal" / "model_comparison.csv"
OUT = ROOT / "modelling_outputs_for_lr22_nahidul"
OUT.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# LR 2.2 COLUMNS
# -----------------------------------------------------------------------------
TARGET = "Goals Scored (Target)"
GROUP_COL = "Match #"
FEATURES = [
    "X1 Team FIFA Rank",
    "X2 Opponent FIFA Rank",
    "X3 Team Squad Value EURm",
    "X4 Team Is Host",
    "X5 Team Qual Goals Scored per Match",
    "X6 Opponent Qual Goals Conceded per Match",
    "X7 Knockout Match",
    "X8 Team WC Appearances Before 2026",
]
SHORT = {
    "X1 Team FIFA Rank": "X1 Team rank",
    "X2 Opponent FIFA Rank": "X2 Opponent rank",
    "X3 Team Squad Value EURm": "X3 Squad value",
    "X4 Team Is Host": "X4 Host",
    "X5 Team Qual Goals Scored per Match": "X5 Qual goals/match",
    "X6 Opponent Qual Goals Conceded per Match": "X6 Opp qual conceded",
    "X7 Knockout Match": "X7 Knockout",
    "X8 Team WC Appearances Before 2026": "X8 WC appearances",
}

# Member 3's EDA recommends testing log(X3) for the transformed-feature OLS.
TRANSFORMED_FEATURES = FEATURES.copy()
REDUCED_FEATURES = [f for f in FEATURES if f != "X8 Team WC Appearances Before 2026"]

# Hyperparameter grids. Small enough for the assignment-sized dataset but broad
# enough to cover weak-to-strong regularisation.
ALPHAS = [0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 50, 100]
L1_RATIOS = [0.1, 0.3, 0.5, 0.7, 0.9]

N_SPLITS = 5
N_REPEATS = 10
RANDOM_SEED = 42


# -----------------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------------
@dataclass
class EvalResult:
    name: str
    model_key: str
    target_mode: str
    fold_scores: pd.DataFrame
    oof_prediction: np.ndarray

    @property
    def summary(self) -> dict:
        fs = self.fold_scores
        return {
            "Model": self.name,
            "Target Scale": self.target_mode,
            "MAE": fs["MAE"].mean(),
            "RMSE": fs["RMSE"].mean(),
            "R2": fs["R2"].mean(),
            "MAE_std": fs["MAE"].std(ddof=1),
            "RMSE_std": fs["RMSE"].std(ddof=1),
            "R2_std": fs["R2"].std(ddof=1),
        }


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def metrics(y_true, y_pred) -> tuple[float, float, float]:
    return (
        float(mean_absolute_error(y_true, y_pred)),
        rmse(y_true, y_pred),
        float(r2_score(y_true, y_pred)),
    )


def make_match_splits(match_ids: np.ndarray) -> list[dict]:
    """Create 10 repeats of shuffled 5-fold splits at the MATCH level."""
    unique_matches = np.array(sorted(pd.unique(match_ids)))
    splits = []
    for repeat in range(N_REPEATS):
        kf = KFold(
            n_splits=N_SPLITS,
            shuffle=True,
            random_state=RANDOM_SEED + repeat,
        )
        # Split the unique match IDs first, then map the selected match IDs back
        # to both team rows. This works across older scikit-learn versions too.
        for fold, (tr, te) in enumerate(kf.split(unique_matches), start=1):
            splits.append(
                {
                    "repeat": repeat + 1,
                    "fold": fold,
                    "train_matches": unique_matches[tr],
                    "test_matches": unique_matches[te],
                }
            )
    return splits


def grouped_row_splits(groups: np.ndarray, n_splits: int, seed: int):
    """Return row-index CV splits while keeping each group entirely together."""
    unique_groups = np.array(sorted(pd.unique(groups)))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    row_splits = []
    for train_g_idx, test_g_idx in kf.split(unique_groups):
        train_groups = unique_groups[train_g_idx]
        test_groups = unique_groups[test_g_idx]
        train_idx = np.where(np.isin(groups, train_groups))[0]
        test_idx = np.where(np.isin(groups, test_groups))[0]
        row_splits.append((train_idx, test_idx))
    return row_splits


def feature_frame(df: pd.DataFrame, model_key: str) -> pd.DataFrame:
    """Return the appropriate feature matrix for a model variant."""
    if model_key == "ols_transformed":
        X = df[FEATURES].copy()
        # log1p is safe because squad value is non-negative.
        X["X3 Team Squad Value EURm"] = np.log1p(X["X3 Team Squad Value EURm"])
        return X
    if model_key == "ols_reduced":
        return df[REDUCED_FEATURES].copy()
    return df[FEATURES].copy()


def build_untuned_model(model_key: str):
    if model_key == "baseline":
        return DummyRegressor(strategy="mean")
    if model_key in {"ols_full", "ols_transformed", "ols_reduced"}:
        return LinearRegression()
    if model_key == "ridge":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge()),
        ])
    if model_key == "lasso":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", Lasso(max_iter=20000)),
        ])
    if model_key == "elasticnet":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", ElasticNet(max_iter=20000)),
        ])
    raise ValueError(f"Unknown model_key: {model_key}")


def tune_model(
    model_key: str,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    seed: int,
):
    """Tune regularised models using grouped CV inside the training data only."""
    base = build_untuned_model(model_key)
    if model_key not in {"ridge", "lasso", "elasticnet"}:
        return base, {}

    inner_cv = grouped_row_splits(groups_train, n_splits=5, seed=seed)
    if model_key in {"ridge", "lasso"}:
        param_grid = {"model__alpha": ALPHAS}
    else:
        param_grid = {
            "model__alpha": ALPHAS,
            "model__l1_ratio": L1_RATIOS,
        }

    gs = GridSearchCV(
        estimator=base,
        param_grid=param_grid,
        scoring="neg_root_mean_squared_error",
        cv=inner_cv,
        n_jobs=-1,
        refit=True,
    )
    gs.fit(X_train, y_train)
    return gs.best_estimator_, gs.best_params_


def fit_predict_one_fold(
    model_key: str,
    X_train: pd.DataFrame,
    y_train_raw: np.ndarray,
    groups_train: np.ndarray,
    X_test: pd.DataFrame,
    target_mode: str,
    seed: int,
):
    """Fit one fold, optionally on log1p(target), then return predictions in goal units."""
    if target_mode not in {"raw", "log1p"}:
        raise ValueError("target_mode must be 'raw' or 'log1p'")

    if target_mode == "log1p":
        y_train_fit = np.log1p(y_train_raw)
    else:
        y_train_fit = y_train_raw

    model, params = tune_model(
        model_key, X_train, y_train_fit, groups_train, seed
    )
    model.fit(X_train, y_train_fit)
    pred_fit = np.asarray(model.predict(X_test), dtype=float)

    if target_mode == "log1p":
        pred = np.expm1(pred_fit)
        # Goals cannot be negative. Clipping is applied only after inverse
        # transformation for evaluation on the original goal scale.
        pred = np.maximum(pred, 0.0)
    else:
        pred = pred_fit

    return model, pred, params


def evaluate_lr22_model(
    df: pd.DataFrame,
    splits: list[dict],
    model_key: str,
    display_name: str,
    target_mode: str = "raw",
) -> EvalResult:
    """Nested repeated grouped CV and averaged out-of-fold predictions."""
    y_all = df[TARGET].to_numpy(dtype=float)
    groups_all = df[GROUP_COL].to_numpy()

    pred_sum = np.zeros(len(df), dtype=float)
    pred_count = np.zeros(len(df), dtype=int)
    fold_records = []

    for split_no, split in enumerate(splits, start=1):
        train_mask = df[GROUP_COL].isin(split["train_matches"]).to_numpy()
        test_mask = df[GROUP_COL].isin(split["test_matches"]).to_numpy()
        train_idx = np.where(train_mask)[0]
        test_idx = np.where(test_mask)[0]

        X_full = feature_frame(df, model_key)
        X_train = X_full.iloc[train_idx]
        X_test = X_full.iloc[test_idx]
        y_train = y_all[train_idx]
        y_test = y_all[test_idx]
        groups_train = groups_all[train_idx]

        _, pred, params = fit_predict_one_fold(
            model_key=model_key,
            X_train=X_train,
            y_train_raw=y_train,
            groups_train=groups_train,
            X_test=X_test,
            target_mode=target_mode,
            seed=1000 + split_no,
        )

        mae_v, rmse_v, r2_v = metrics(y_test, pred)
        fold_records.append(
            {
                "Repeat": split["repeat"],
                "Fold": split["fold"],
                "Model": display_name,
                "Target Scale": target_mode,
                "MAE": mae_v,
                "RMSE": rmse_v,
                "R2": r2_v,
                "Tuning Parameters": str(params),
            }
        )
        pred_sum[test_idx] += pred
        pred_count[test_idx] += 1

    if not np.all(pred_count == N_REPEATS):
        raise RuntimeError(
            "Each row should receive exactly one out-of-fold prediction per repeat. "
            f"Observed prediction counts: {np.unique(pred_count, return_counts=True)}"
        )

    return EvalResult(
        name=display_name,
        model_key=model_key,
        target_mode=target_mode,
        fold_scores=pd.DataFrame(fold_records),
        oof_prediction=pred_sum / pred_count,
    )


def tune_final_lr22_model(df: pd.DataFrame, model_key: str, target_mode: str):
    """Tune (if needed) and fit the selected model on the full LR 2.2 dataset."""
    X = feature_frame(df, model_key)
    y_raw = df[TARGET].to_numpy(dtype=float)
    y_fit = np.log1p(y_raw) if target_mode == "log1p" else y_raw
    groups = df[GROUP_COL].to_numpy()

    model, params = tune_model(
        model_key=model_key,
        X_train=X,
        y_train=y_fit,
        groups_train=groups,
        seed=9999,
    )
    model.fit(X, y_fit)
    return model, params


def parse_lr21_best_model(results_path: Path) -> tuple[str, dict]:
    """Read Member 2's model table and reconstruct the selected LR 2.1 model."""
    res = pd.read_csv(results_path)
    best = res.loc[res["RMSE"].idxmin(), "Model"]

    if best.startswith("Ridge"):
        m = re.search(r"a=([0-9.]+)", best)
        alpha = float(m.group(1)) if m else 1.0
        return "ridge", {"alpha": alpha, "label": best}
    if best.startswith("Lasso"):
        m = re.search(r"a=([0-9.]+)", best)
        alpha = float(m.group(1)) if m else 1.0
        return "lasso", {"alpha": alpha, "label": best}
    if best.startswith("ElasticNet"):
        ma = re.search(r"a=([0-9.]+)", best)
        ml = re.search(r"l1=([0-9.]+)", best)
        alpha = float(ma.group(1)) if ma else 1.0
        l1 = float(ml.group(1)) if ml else 0.5
        return "elasticnet", {"alpha": alpha, "l1_ratio": l1, "label": best}
    if best.startswith("OLS (transformed"):
        return "ols_transformed", {"label": best}
    if best.startswith("OLS (reduced"):
        return "ols_reduced", {"label": best}
    if best.startswith("OLS"):
        return "ols_full", {"label": best}
    return "baseline", {"label": best}


def build_lr21_model(kind: str, params: dict):
    if kind == "ridge":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=params["alpha"])),
        ])
    if kind == "lasso":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", Lasso(alpha=params["alpha"], max_iter=20000)),
        ])
    if kind == "elasticnet":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", ElasticNet(
                alpha=params["alpha"],
                l1_ratio=params["l1_ratio"],
                max_iter=20000,
            )),
        ])
    if kind in {"ols_full", "ols_transformed", "ols_reduced"}:
        return LinearRegression()
    return DummyRegressor(strategy="mean")


def lr21_feature_frame(df21: pd.DataFrame, kind: str) -> pd.DataFrame:
    features21 = [
        "X1 FIFA Rank Diff (Away-Home)",
        "X2 Squad Value Diff EURm (Home-Away)",
        "X3 Squad Age Diff (Home-Away)",
        "X4 Prior WC Best Result Diff (Home-Away)",
        "X5 Qualifying GD Diff (Home-Away)",
        "X6 Host Advantage Diff (Home-Away)",
        "X7 Confederation Strength Diff (Home-Away)",
        "X8 Rest Days Diff (Home-Away)",
    ]
    if kind == "ols_reduced":
        return df21[[f for f in features21 if not f.startswith("X4")]].copy()
    if kind == "ols_transformed":
        X = df21[features21].copy()
        X["X1_x_X2"] = (
            X["X1 FIFA Rank Diff (Away-Home)"]
            * X["X2 Squad Value Diff EURm (Home-Away)"]
        )
        return X
    return df21[features21].copy()


def lr21_oof_predictions(df21: pd.DataFrame, splits: list[dict]):
    """Generate aligned repeated OOF predictions using Member 2's chosen model."""
    kind, params = parse_lr21_best_model(LR21_RESULTS_PATH)
    X_all = lr21_feature_frame(df21, kind)
    y_all = df21["Goal Difference (Target)"].to_numpy(dtype=float)

    pred_sum = np.zeros(len(df21), dtype=float)
    pred_count = np.zeros(len(df21), dtype=int)
    fold_records = []

    for split in splits:
        train_mask = df21[GROUP_COL].isin(split["train_matches"]).to_numpy()
        test_mask = df21[GROUP_COL].isin(split["test_matches"]).to_numpy()
        tr = np.where(train_mask)[0]
        te = np.where(test_mask)[0]

        model = build_lr21_model(kind, params)
        model.fit(X_all.iloc[tr], y_all[tr])
        pred = np.asarray(model.predict(X_all.iloc[te]), dtype=float)

        pred_sum[te] += pred
        pred_count[te] += 1
        mae_v, rmse_v, r2_v = metrics(y_all[te], pred)
        fold_records.append({
            "Repeat": split["repeat"],
            "Fold": split["fold"],
            "Model": params["label"],
            "MAE": mae_v,
            "RMSE": rmse_v,
            "R2": r2_v,
        })

    if not np.all(pred_count == N_REPEATS):
        raise RuntimeError("LR 2.1 OOF prediction count mismatch")

    return pred_sum / pred_count, pd.DataFrame(fold_records), params["label"]


def validate_inputs(df: pd.DataFrame):
    assert len(df) == 208, f"LR 2.2 dataset must have 208 rows, found {len(df)}"
    assert len(FEATURES) == 8, "Exactly 8 explanatory variables are required"
    assert df[GROUP_COL].nunique() == 104, "Expected 104 distinct matches"
    assert df.groupby(GROUP_COL).size().eq(2).all(), "Each match must have exactly 2 team rows"
    assert df[FEATURES + [TARGET]].isna().sum().sum() == 0, "Missing model values found"
    assert (df[TARGET] >= 0).all(), "Goal target must be non-negative"


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def main():
    print("=" * 76)
    print("HIT140 Assessment 3 - Member 4 Nahidul - LR 2.2 Modelling")
    print("=" * 76)

    df = pd.read_csv(LR22_PATH)
    df21 = pd.read_csv(LR21_PATH)
    validate_inputs(df)

    print(f"LR 2.2 rows: {len(df)} | matches: {df[GROUP_COL].nunique()} | features: {len(FEATURES)}")
    print("Validation: exactly 208 rows, 104 matches, 8 predictors, no missing model values")

    splits = make_match_splits(df[GROUP_COL].to_numpy())
    print(f"Outer evaluation: {N_SPLITS}-fold grouped CV x {N_REPEATS} repeats = {len(splits)} folds")
    print("Both team rows from one match always stay in the same fold.")

    # ------------------------------------------------------------------
    # A. RAW-TARGET MODEL COMPARISON
    # ------------------------------------------------------------------
    raw_specs = [
        ("baseline", "Baseline (mean)"),
        ("ols_full", "OLS (full)"),
        ("ols_transformed", "OLS (transformed X3)"),
        ("ols_reduced", "OLS (reduced, no X8)"),
        ("ridge", "Ridge"),
        ("lasso", "Lasso"),
        ("elasticnet", "ElasticNet"),
    ]

    raw_results: dict[str, EvalResult] = {}
    all_fold_tables = []

    print("\n[A] RAW GOALS: model comparison")
    for key, label in raw_specs:
        print(f"  Evaluating {label} ...")
        result = evaluate_lr22_model(df, splits, key, label, target_mode="raw")
        raw_results[label] = result
        all_fold_tables.append(result.fold_scores)
        s = result.summary
        print(
            f"    MAE={s['MAE']:.4f} | RMSE={s['RMSE']:.4f} | R2={s['R2']:.4f}"
        )

    raw_summary = pd.DataFrame([r.summary for r in raw_results.values()])
    raw_summary = raw_summary.sort_values("RMSE").reset_index(drop=True)
    best_raw_name = raw_summary.iloc[0]["Model"]
    best_raw_result = raw_results[best_raw_name]
    print(f"  Best raw-target model by RMSE: {best_raw_name}")

    # ------------------------------------------------------------------
    # B. LOG-TARGET TEST
    # ------------------------------------------------------------------
    print("\n[B] TARGET TRANSFORMATION TEST: log(goals + 1)")
    key_by_label = {label: key for key, label in raw_specs}
    best_raw_key = key_by_label[best_raw_name]

    log_results: dict[str, EvalResult] = {}
    # Always test full OLS on log target.
    log_ols = evaluate_lr22_model(
        df, splits, "ols_full", "OLS (full, log target)", target_mode="log1p"
    )
    log_results[log_ols.name] = log_ols
    all_fold_tables.append(log_ols.fold_scores)

    # Also test the family that won on the raw target (unless it is already OLS full).
    if best_raw_key != "ols_full":
        log_best = evaluate_lr22_model(
            df,
            splits,
            best_raw_key,
            f"{best_raw_name} + log target",
            target_mode="log1p",
        )
        log_results[log_best.name] = log_best
        all_fold_tables.append(log_best.fold_scores)
    else:
        log_best = log_ols

    # Candidate final models: best raw + the tested log variants.
    candidates = [best_raw_result] + list(log_results.values())
    # Deduplicate by (model_key, target_mode)
    unique_candidates = {}
    for c in candidates:
        unique_candidates[(c.model_key, c.target_mode)] = c
    candidates = list(unique_candidates.values())
    final_result = min(candidates, key=lambda r: r.summary["RMSE"])

    target_comparison = pd.DataFrame([c.summary for c in candidates]).sort_values("RMSE")
    print(target_comparison[["Model", "Target Scale", "MAE", "RMSE", "R2"]].to_string(index=False))
    print(
        f"  Selected final LR 2.2 formulation: {final_result.name} "
        f"({final_result.target_mode})"
    )

    # ------------------------------------------------------------------
    # C. PAIRED t-TEST: FINAL MODEL VS FULL RAW OLS
    # ------------------------------------------------------------------
    print("\n[C] Paired t-test: final LR 2.2 model fold RMSE vs raw OLS")
    ols_folds = raw_results["OLS (full)"].fold_scores.sort_values(["Repeat", "Fold"])
    final_folds = final_result.fold_scores.sort_values(["Repeat", "Fold"])
    t_stat, p_val = stats.ttest_rel(ols_folds["RMSE"], final_folds["RMSE"])
    paired_test = pd.DataFrame([{
        "Comparison": f"{final_result.name} vs OLS (full)",
        "OLS Mean Fold RMSE": ols_folds["RMSE"].mean(),
        "Final Mean Fold RMSE": final_folds["RMSE"].mean(),
        "t statistic": t_stat,
        "p value": p_val,
        "Significant at 0.05": bool(p_val < 0.05),
    }])
    print(paired_test.to_string(index=False))

    # ------------------------------------------------------------------
    # D. OLS INFERENCE + ASSUMPTION DIAGNOSTICS
    # ------------------------------------------------------------------
    print("\n[D] OLS coefficient inference and diagnostics")
    X_ols = df[FEATURES].astype(float)
    y = df[TARGET].astype(float).to_numpy()
    X_sm = sm.add_constant(X_ols, has_constant="add")
    ols_sm = sm.OLS(y, X_sm).fit()
    residuals = np.asarray(ols_sm.resid)
    fitted = np.asarray(ols_sm.fittedvalues)

    with open(OUT / "ols_full_summary_lr22.txt", "w", encoding="utf-8") as f:
        f.write(str(ols_sm.summary()))

    ci = ols_sm.conf_int(alpha=0.05)
    coef_df = pd.DataFrame({
        "Feature": ["Intercept"] + [SHORT[f] for f in FEATURES],
        "Coefficient": ols_sm.params.to_numpy(),
        "Std Error": ols_sm.bse.to_numpy(),
        "t statistic": ols_sm.tvalues.to_numpy(),
        "p value": ols_sm.pvalues.to_numpy(),
        "CI Lower 95%": ci.iloc[:, 0].to_numpy(),
        "CI Upper 95%": ci.iloc[:, 1].to_numpy(),
        "Significant at 0.05": (ols_sm.pvalues.to_numpy() < 0.05),
    })

    # VIF should be calculated with an intercept in the auxiliary regressions.
    # This matches Member 3's EDA method and avoids inflated VIF values that can
    # occur when the auxiliary regressions are forced through the origin.
    X_vif = sm.add_constant(X_ols, has_constant="add").to_numpy()
    vif_df = pd.DataFrame({
        "Feature": [SHORT[f] for f in FEATURES],
        "VIF": [
            variance_inflation_factor(X_vif, i)
            for i in range(1, X_vif.shape[1])
        ],
    }).sort_values("VIF", ascending=False)

    influence = ols_sm.get_influence()
    cooks_d = np.asarray(influence.cooks_distance[0])
    cooks_threshold = 4 / len(df)
    infl_idx = np.where(cooks_d > cooks_threshold)[0]
    influential = df.loc[
        infl_idx,
        ["Row ID", "Match #", "Team", "Opponent", TARGET],
    ].copy()
    influential["Cooks Distance"] = cooks_d[infl_idx]
    influential["Threshold 4/n"] = cooks_threshold
    influential = influential.sort_values("Cooks Distance", ascending=False)

    # ------------------------------------------------------------------
    # E. FIT FINAL LR 2.2 MODEL ON ALL DATA
    # ------------------------------------------------------------------
    print("\n[E] Fitting selected final LR 2.2 model on all 208 rows")
    final_model, final_params = tune_final_lr22_model(
        df, final_result.model_key, final_result.target_mode
    )
    print(f"  Final model: {final_result.name}")
    print(f"  Full-data tuning parameters: {final_params}")

    # OOF predictions are preferred for performance charts/tables.
    lr22_oof = final_result.oof_prediction
    pred_table = df[[
        "Row ID", "Match #", "Stage", "Team", "Opponent", "Fixture Slot", TARGET
    ]].copy()
    pred_table["Predicted Goals (OOF mean)"] = lr22_oof
    pred_table["Residual"] = pred_table[TARGET] - pred_table["Predicted Goals (OOF mean)"]
    pred_table["Absolute Error"] = np.abs(pred_table["Residual"])

    # Save selected-model coefficients where directly available.
    final_coef_rows = []
    if final_result.model_key in {"ridge", "lasso", "elasticnet"}:
        model_step = final_model.named_steps["model"]
        for feat, coef in zip(FEATURES, model_step.coef_):
            final_coef_rows.append({
                "Feature": SHORT[feat],
                "Standardised Coefficient": coef,
            })
    elif final_result.model_key in {"ols_full", "ols_transformed", "ols_reduced"}:
        feats = list(feature_frame(df, final_result.model_key).columns)
        coefs = np.asarray(final_model.coef_)
        for feat, coef in zip(feats, coefs):
            final_coef_rows.append({
                "Feature": SHORT.get(feat, feat),
                "Coefficient": coef,
            })
    final_coef_df = pd.DataFrame(final_coef_rows)

    # ------------------------------------------------------------------
    # F. CROSS-MODEL COMPARISON: LR 2.1 vs LR 2.2-derived GD
    # ------------------------------------------------------------------
    print("\n[F] Cross-model comparison: LR 2.1 direct GD vs LR 2.2-derived GD")
    lr21_pred, lr21_fold_scores, lr21_label = lr21_oof_predictions(df21, splits)

    team_pred = pred_table.pivot(
        index="Match #",
        columns="Fixture Slot",
        values="Predicted Goals (OOF mean)",
    )
    team_actual = pred_table.pivot(
        index="Match #",
        columns="Fixture Slot",
        values=TARGET,
    )

    cross = df21[[
        "Match #", "Stage", "Home", "Away", "Goal Difference (Target)"
    ]].copy()
    cross = cross.set_index("Match #")
    cross["LR2.1 Predicted Goal Difference"] = pd.Series(
        lr21_pred, index=df21["Match #"]
    )
    cross["LR2.2 Predicted Home Goals"] = team_pred["Home slot"]
    cross["LR2.2 Predicted Away Goals"] = team_pred["Away slot"]
    cross["LR2.2 Derived Goal Difference"] = (
        cross["LR2.2 Predicted Home Goals"] - cross["LR2.2 Predicted Away Goals"]
    )
    cross["LR2.1 Absolute Error"] = np.abs(
        cross["Goal Difference (Target)"] - cross["LR2.1 Predicted Goal Difference"]
    )
    cross["LR2.2 Absolute Error"] = np.abs(
        cross["Goal Difference (Target)"] - cross["LR2.2 Derived Goal Difference"]
    )
    cross = cross.reset_index()

    y_gd = cross["Goal Difference (Target)"].to_numpy(dtype=float)
    p21 = cross["LR2.1 Predicted Goal Difference"].to_numpy(dtype=float)
    p22 = cross["LR2.2 Derived Goal Difference"].to_numpy(dtype=float)

    m21 = metrics(y_gd, p21)
    m22 = metrics(y_gd, p22)
    cross_summary = pd.DataFrame([
        {
            "Approach": f"LR 2.1 direct ({lr21_label})",
            "MAE": m21[0], "RMSE": m21[1], "R2": m21[2],
        },
        {
            "Approach": f"LR 2.2 derived ({final_result.name})",
            "MAE": m22[0], "RMSE": m22[1], "R2": m22[2],
        },
    ])

    # Paired comparison at match level using absolute error.
    cross_t, cross_p = stats.ttest_rel(
        cross["LR2.1 Absolute Error"], cross["LR2.2 Absolute Error"]
    )
    cross_test = pd.DataFrame([{
        "Comparison": "Absolute error: LR2.1 direct vs LR2.2-derived",
        "Mean LR2.1 absolute error": cross["LR2.1 Absolute Error"].mean(),
        "Mean LR2.2 absolute error": cross["LR2.2 Absolute Error"].mean(),
        "t statistic": cross_t,
        "p value": cross_p,
        "Significant at 0.05": bool(cross_p < 0.05),
    }])

    print(cross_summary.to_string(index=False))
    print(cross_test.to_string(index=False))

    # ------------------------------------------------------------------
    # G. SAVE TABLES
    # ------------------------------------------------------------------
    print("\n[G] Saving tables")
    raw_summary.to_csv(OUT / "model_comparison_lr22.csv", index=False)
    target_comparison.to_csv(OUT / "target_transform_comparison.csv", index=False)
    pd.concat(all_fold_tables, ignore_index=True).to_csv(
        OUT / "cv_fold_scores_lr22.csv", index=False
    )
    paired_test.to_csv(OUT / "paired_ttest_final_vs_ols.csv", index=False)
    coef_df.to_csv(OUT / "ols_coefficients_lr22.csv", index=False)
    vif_df.to_csv(OUT / "vif_lr22.csv", index=False)
    influential.to_csv(OUT / "influential_observations_lr22.csv", index=False)
    pred_table.to_csv(OUT / "lr22_oof_predictions_208.csv", index=False)
    final_coef_df.to_csv(OUT / "final_model_coefficients_lr22.csv", index=False)
    lr21_fold_scores.to_csv(OUT / "lr21_aligned_cv_fold_scores.csv", index=False)
    cross.to_csv(OUT / "cross_model_predictions_104.csv", index=False)
    cross_summary.to_csv(OUT / "cross_model_summary.csv", index=False)
    cross_test.to_csv(OUT / "cross_model_paired_test.csv", index=False)

    # ------------------------------------------------------------------
    # H. CHARTS
    # ------------------------------------------------------------------
    print("\n[H] Generating charts")

    # Chart 1: Raw model comparison
    plot_df = raw_summary.sort_values("RMSE", ascending=False)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars = ax.barh(plot_df["Model"], plot_df["RMSE"])
    ax.set_xlabel("Repeated grouped-CV RMSE (goals, lower is better)")
    ax.set_title("LR 2.2 Model Comparison: RMSE")
    for b, v in zip(bars, plot_df["RMSE"]):
        ax.text(v + 0.01, b.get_y() + b.get_height()/2, f"{v:.3f}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "chart1_model_comparison_rmse.png", bbox_inches="tight")
    plt.close(fig)

    # Chart 2: OOF predicted vs actual for selected LR 2.2 formulation
    fig, ax = plt.subplots(figsize=(6.8, 6.2))
    ax.scatter(y, lr22_oof, alpha=0.58, s=35, edgecolors="white", linewidth=0.35)
    lim_low = min(y.min(), lr22_oof.min()) - 0.25
    lim_high = max(y.max(), lr22_oof.max()) + 0.25
    ax.plot([lim_low, lim_high], [lim_low, lim_high], linestyle="--", linewidth=1.2)
    ax.set_xlim(lim_low, lim_high)
    ax.set_ylim(lim_low, lim_high)
    ax.set_xlabel("Actual goals")
    ax.set_ylabel("Out-of-fold predicted goals")
    fs = final_result.summary
    ax.set_title(f"LR 2.2 Predicted vs Actual\n{final_result.name}")
    ax.text(
        0.03, 0.97,
        f"CV MAE={fs['MAE']:.3f}  RMSE={fs['RMSE']:.3f}  R2={fs['R2']:.3f}",
        transform=ax.transAxes, va="top", fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )
    fig.tight_layout()
    fig.savefig(OUT / "chart2_predicted_vs_actual_lr22.png", bbox_inches="tight")
    plt.close(fig)

    # Chart 3: Residuals vs fitted (OLS)
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    ax.scatter(fitted, residuals, alpha=0.6, s=35, edgecolors="white", linewidth=0.35)
    ax.axhline(0, linestyle="--", linewidth=1.2)
    ax.set_xlabel("OLS fitted goals")
    ax.set_ylabel("Residual")
    ax.set_title("LR 2.2 OLS Residuals vs Fitted")
    fig.tight_layout()
    fig.savefig(OUT / "chart3_residuals_vs_fitted.png", bbox_inches="tight")
    plt.close(fig)

    # Chart 4: Q-Q plot
    fig, ax = plt.subplots(figsize=(6.4, 5.5))
    stats.probplot(residuals, dist="norm", plot=ax)
    ax.set_title("LR 2.2 OLS Residual Q-Q Plot")
    fig.tight_layout()
    fig.savefig(OUT / "chart4_qq_plot.png", bbox_inches="tight")
    plt.close(fig)

    # Chart 5: Cook's distance
    fig, ax = plt.subplots(figsize=(9.5, 4.5))
    ax.stem(np.arange(len(df)), cooks_d, basefmt=" ", markerfmt=".")
    ax.axhline(cooks_threshold, linestyle="--", linewidth=1.2,
               label=f"4/n = {cooks_threshold:.4f}")
    ax.set_xlabel("Observation index")
    ax.set_ylabel("Cook's distance")
    ax.set_title("LR 2.2 OLS Influential Observations")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "chart5_cooks_distance.png", bbox_inches="tight")
    plt.close(fig)

    # Chart 6: OLS coefficients with 95% CI
    cp = coef_df[coef_df["Feature"] != "Intercept"].copy().sort_values("Coefficient")
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.errorbar(
        cp["Coefficient"], cp["Feature"],
        xerr=[
            cp["Coefficient"] - cp["CI Lower 95%"],
            cp["CI Upper 95%"] - cp["Coefficient"],
        ],
        fmt="o", capsize=4,
    )
    ax.axvline(0, linestyle="--", linewidth=1)
    ax.set_xlabel("OLS coefficient with 95% confidence interval")
    ax.set_title("LR 2.2 OLS Coefficients")
    fig.tight_layout()
    fig.savefig(OUT / "chart6_ols_coefficients_ci.png", bbox_inches="tight")
    plt.close(fig)

    # Chart 7: raw vs log target test
    tc = target_comparison.sort_values("RMSE", ascending=False)
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    bars = ax.barh(tc["Model"], tc["RMSE"])
    ax.set_xlabel("Repeated grouped-CV RMSE (goals)")
    ax.set_title("LR 2.2 Target-Scale Comparison")
    for b, v in zip(bars, tc["RMSE"]):
        ax.text(v + 0.01, b.get_y() + b.get_height()/2, f"{v:.3f}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "chart7_target_transform_comparison.png", bbox_inches="tight")
    plt.close(fig)

    # Chart 8: Cross-model RMSE comparison
    cs = cross_summary.sort_values("RMSE", ascending=False)
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    bars = ax.barh(cs["Approach"], cs["RMSE"])
    ax.set_xlabel("OOF RMSE of goal difference")
    ax.set_title("Cross-model Comparison: Goal Difference Prediction")
    for b, v in zip(bars, cs["RMSE"]):
        ax.text(v + 0.01, b.get_y() + b.get_height()/2, f"{v:.3f}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "chart8_cross_model_comparison.png", bbox_inches="tight")
    plt.close(fig)

    # Chart 9: LR2.1 vs LR2.2 predicted goal difference
    fig, ax = plt.subplots(figsize=(6.5, 6.0))
    ax.scatter(
        cross["LR2.1 Predicted Goal Difference"],
        cross["LR2.2 Derived Goal Difference"],
        alpha=0.65, s=38, edgecolors="white", linewidth=0.35,
    )
    low = min(
        cross["LR2.1 Predicted Goal Difference"].min(),
        cross["LR2.2 Derived Goal Difference"].min(),
    ) - 0.25
    high = max(
        cross["LR2.1 Predicted Goal Difference"].max(),
        cross["LR2.2 Derived Goal Difference"].max(),
    ) + 0.25
    ax.plot([low, high], [low, high], linestyle="--", linewidth=1.2)
    ax.set_xlabel("LR 2.1 direct predicted goal difference")
    ax.set_ylabel("LR 2.2-derived predicted goal difference")
    ax.set_title("LR 2.1 vs LR 2.2 Goal-Difference Predictions")
    fig.tight_layout()
    fig.savefig(OUT / "chart9_lr21_vs_lr22_predictions.png", bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------------
    # I. RESULTS SUMMARY FOR MEMBER 3 / REPORT LEAD
    # ------------------------------------------------------------------
    best_raw_s = best_raw_result.summary
    final_s = final_result.summary
    sig_text = (
        "statistically significant" if p_val < 0.05 else "not statistically significant"
    )
    better_cross = cross_summary.sort_values("RMSE").iloc[0]

    summary_lines = [
        "# Member 4 LR 2.2 modelling results summary",
        "",
        "This file is a factual results handover for Member 3. Rewrite the final report section in the group's own words.",
        "",
        "## Validation design",
        f"- {N_REPEATS} repeats of {N_SPLITS}-fold grouped cross-validation were used.",
        "- Both team rows from the same match were kept in the same fold using Match # as the group.",
        "- Ridge, Lasso and ElasticNet hyperparameters were tuned inside the training portion of each outer fold.",
        "",
        "## Raw-target model comparison",
        f"- Best raw-target model by mean CV RMSE: {best_raw_name}.",
        f"- Mean CV MAE: {best_raw_s['MAE']:.4f}.",
        f"- Mean CV RMSE: {best_raw_s['RMSE']:.4f}.",
        f"- Mean CV R2: {best_raw_s['R2']:.4f}.",
        "",
        "## Raw goals vs log(goals + 1)",
        f"- Selected final formulation: {final_result.name} ({final_result.target_mode}).",
        f"- Mean CV MAE: {final_s['MAE']:.4f}.",
        f"- Mean CV RMSE: {final_s['RMSE']:.4f}.",
        f"- Mean CV R2: {final_s['R2']:.4f}.",
        "",
        "## Paired model comparison",
        f"- Paired fold-RMSE t-test against raw full OLS: t={t_stat:.4f}, p={p_val:.4f}.",
        f"- At alpha=0.05, the difference is {sig_text}.",
        "",
        "## OLS diagnostics",
        f"- Maximum VIF: {vif_df['VIF'].max():.3f}.",
        f"- Cook's-distance threshold 4/n: {cooks_threshold:.4f}.",
        f"- Observations above the Cook's-distance threshold: {len(influential)}.",
        "- See the residual, Q-Q, VIF, Cook's-distance and coefficient output files for interpretation.",
        "",
        "## Cross-model comparison",
        f"- LR 2.1 model used: {lr21_label}.",
        f"- LR 2.1 direct goal-difference RMSE: {m21[1]:.4f}.",
        f"- LR 2.2-derived goal-difference RMSE: {m22[1]:.4f}.",
        f"- Lower-RMSE approach: {better_cross['Approach']}.",
        f"- Paired absolute-error test: t={cross_t:.4f}, p={cross_p:.4f}.",
        "",
        "## Limitations to mention",
        "- Ordinary linear regression can produce fractional goals and, for raw-target models, negative predictions.",
        "- The target is a non-negative integer count, so linear regression is an approximation rather than a count-data model.",
        "- Each match contributes two related team rows; grouped validation was therefore used to avoid putting opposite sides of the same match into train and test folds.",
        "- The dataset contains only 104 matches (208 team rows), so coefficient estimates and validation results can be sensitive to influential matches.",
    ]
    (OUT / "results_summary_for_report.md").write_text("\n".join(summary_lines), encoding="utf-8")

    recommendation_text = "\n".join([
        "HIT140 Assessment 3 - Member 4 Nahidul",
        "LR 2.2 Final Model Recommendation",
        "=" * 60,
        f"Selected model: {final_result.name}",
        f"Target mode: {final_result.target_mode}",
        f"Mean grouped-CV MAE: {final_s['MAE']:.4f}",
        f"Mean grouped-CV RMSE: {final_s['RMSE']:.4f}",
        f"Mean grouped-CV R2: {final_s['R2']:.4f}",
        f"Full-data tuning parameters: {final_params}",
        f"Paired t-test vs raw OLS: t={t_stat:.4f}, p={p_val:.4f}",
        "",
        "Selection rule: lowest repeated grouped-CV RMSE among the best raw-target",
        "candidate and the tested log-target formulations. MAE and R2 are reported",
        "as supporting metrics. Grouped validation prevents two rows from one match",
        "appearing on opposite sides of a train/test split.",
    ])
    (OUT / "final_recommendation.txt").write_text(recommendation_text, encoding="utf-8")

    print("\n" + "=" * 76)
    print("FINAL LR 2.2 RECOMMENDATION")
    print("=" * 76)
    print(f"Selected model: {final_result.name}")
    print(f"Target scale: {final_result.target_mode}")
    print(f"CV MAE:  {final_s['MAE']:.4f}")
    print(f"CV RMSE: {final_s['RMSE']:.4f}")
    print(f"CV R2:   {final_s['R2']:.4f}")
    print(f"Paired t-test vs OLS: t={t_stat:.4f}, p={p_val:.4f}")
    print(f"Outputs: {OUT}")
    print("=" * 76)


if __name__ == "__main__":
    main()
