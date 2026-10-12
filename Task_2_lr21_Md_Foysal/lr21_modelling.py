"""
HIT140 Assessment 3 -- LR 2.1 Modelling Pipeline
Predicting Goal Difference for 2026 FIFA World Cup matches (104 rows)

Author : Md Foysal  (LR 2.1 Modelling)
"""
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedKFold, GridSearchCV, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.dummy import DummyRegressor
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
import statsmodels.api as sm

warnings.filterwarnings('ignore')
sns.set_theme(style='whitegrid', font_scale=0.95)
plt.rcParams['figure.dpi'] = 140
plt.rcParams['font.family'] = 'DejaVu Sans'

OUT = 'modelling_outputs_for_lr21_md_foysal'
os.makedirs(OUT, exist_ok=True)

# ================================================================
# 0. LOAD DATA
# ================================================================
df = pd.read_csv('lr21_dataset_104.csv')

TARGET = 'Goal Difference (Target)'
FEATURES = [
    'X1 FIFA Rank Diff (Away-Home)',
    'X2 Squad Value Diff EURm (Home-Away)',
    'X3 Squad Age Diff (Home-Away)',
    'X4 Prior WC Best Result Diff (Home-Away)',
    'X5 Qualifying GD Diff (Home-Away)',
    'X6 Host Advantage Diff (Home-Away)',
    'X7 Confederation Strength Diff (Home-Away)',
    'X8 Rest Days Diff (Home-Away)',
]
SHORT = {
    'X1 FIFA Rank Diff (Away-Home)':            'X1 FIFA Rank',
    'X2 Squad Value Diff EURm (Home-Away)':     'X2 Squad Value',
    'X3 Squad Age Diff (Home-Away)':            'X3 Squad Age',
    'X4 Prior WC Best Result Diff (Home-Away)': 'X4 Prior WC',
    'X5 Qualifying GD Diff (Home-Away)':        'X5 Qual GD',
    'X6 Host Advantage Diff (Home-Away)':       'X6 Host',
    'X7 Confederation Strength Diff (Home-Away)':'X7 Conf',
    'X8 Rest Days Diff (Home-Away)':            'X8 Rest',
}

X_all  = df[FEATURES].values
y      = df[TARGET].values
n, p   = X_all.shape

print('=' * 65)
print('LR 2.1 Modelling  |  Member 2  |  Md Foysal  S395914')
print(f'Rows: {n}   Features: {p}   Target: {TARGET}')
print('=' * 65)

# ================================================================
# 1. EVALUATION PIPELINE
#    Repeated 5-fold CV (10 repeats) -- shared with Member 4
# ================================================================
CV = RepeatedKFold(n_splits=5, n_repeats=10, random_state=42)

def cv_scores(estimator, X, y, cv):
    """Return dict of mean MAE, RMSE, R2 from cross-validation."""
    res = cross_validate(
        estimator, X, y, cv=cv,
        scoring={
            'mae':  'neg_mean_absolute_error',
            'rmse': 'neg_root_mean_squared_error',
            'r2':   'r2',
        },
        return_train_score=False,
        n_jobs=-1,
    )
    return {
        'MAE':  -res['test_mae'].mean(),
        'RMSE': -res['test_rmse'].mean(),
        'R2':    res['test_r2'].mean(),
        'MAE_std':  res['test_mae'].std(),
        'RMSE_std': res['test_rmse'].std(),
        'R2_std':   res['test_r2'].std(),
        'fold_rmse': -res['test_rmse'],   # 50 individual fold scores for paired t-test
    }

print('\n[1] Evaluation pipeline: Repeated 5-fold CV  (5 splits x 10 repeats = 50 scores per model)')

# ================================================================
# 2. FEATURE ENGINEERING
#    Based on EDA handover: standardise for regularised models.
#    No log transforms needed. One interaction term tested.
# ================================================================

# Reduced feature set: drop X4 (highest VIF, overlaps X1 and X2)
# as suggested in EDA handover
FEATURES_REDUCED = [f for f in FEATURES if not f.startswith('X4')]
X_reduced = df[FEATURES_REDUCED].values

# Transformed features: add X1*X2 interaction (strength agreement amplifier)
# Suggested because X1 and X2 both measure team quality -- when both agree
# (strong FIFA rank AND high squad value), the effect should be amplified.
df['X1_x_X2'] = (df['X1 FIFA Rank Diff (Away-Home)'] *
                  df['X2 Squad Value Diff EURm (Home-Away)'])
FEATURES_TRANS = FEATURES + ['X1_x_X2']
X_trans = df[FEATURES_TRANS].values

print('\n[2] Feature sets prepared:')
print(f'    Full (8 features):           X_all  {X_all.shape}')
print(f'    Reduced (7, drop X4):        X_reduced {X_reduced.shape}')
print(f'    Transformed (8 + X1*X2):     X_trans {X_trans.shape}')

# ================================================================
# 3. MODEL DEFINITIONS
# ================================================================

# Pipelines for regularised models (StandardScaler -> model)
def reg_pipe(model):
    return Pipeline([('scaler', StandardScaler()), ('model', model)])

# Baseline: always predict training mean
baseline    = DummyRegressor(strategy='mean')

# OLS full
ols_full    = LinearRegression()

# OLS transformed (interaction term added)
ols_trans   = LinearRegression()

# OLS reduced (X4 dropped)
ols_reduced = LinearRegression()

# Regularised (scaled inside pipeline)
ridge_model  = reg_pipe(Ridge())
lasso_model  = reg_pipe(Lasso(max_iter=10000))
en_model     = reg_pipe(ElasticNet(max_iter=10000))

# ================================================================
# 4. HYPERPARAMETER TUNING (Ridge, Lasso, ElasticNet)
#    Grid search within CV to find best alpha
# ================================================================
print('\n[3] Tuning hyperparameters via GridSearchCV (5-fold inner CV)...')

alphas = [0.001, 0.01, 0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 200]
l1_ratios = [0.1, 0.3, 0.5, 0.7, 0.9]

# Ridge tuning
ridge_gs = GridSearchCV(
    reg_pipe(Ridge()),
    {'model__alpha': alphas},
    cv=5, scoring='neg_root_mean_squared_error', n_jobs=-1
)
ridge_gs.fit(X_all, y)
best_ridge_alpha = ridge_gs.best_params_['model__alpha']
print(f'    Ridge  best alpha = {best_ridge_alpha}')

# Lasso tuning
lasso_gs = GridSearchCV(
    reg_pipe(Lasso(max_iter=10000)),
    {'model__alpha': alphas},
    cv=5, scoring='neg_root_mean_squared_error', n_jobs=-1
)
lasso_gs.fit(X_all, y)
best_lasso_alpha = lasso_gs.best_params_['model__alpha']
print(f'    Lasso  best alpha = {best_lasso_alpha}')

# ElasticNet tuning
en_gs = GridSearchCV(
    reg_pipe(ElasticNet(max_iter=10000)),
    {'model__alpha': alphas, 'model__l1_ratio': l1_ratios},
    cv=5, scoring='neg_root_mean_squared_error', n_jobs=-1
)
en_gs.fit(X_all, y)
best_en_alpha    = en_gs.best_params_['model__alpha']
best_en_l1ratio  = en_gs.best_params_['model__l1_ratio']
print(f'    ElasticNet best alpha = {best_en_alpha}, l1_ratio = {best_en_l1ratio}')

# Rebuild tuned models
ridge_tuned = reg_pipe(Ridge(alpha=best_ridge_alpha))
lasso_tuned = reg_pipe(Lasso(alpha=best_lasso_alpha, max_iter=10000))
en_tuned    = reg_pipe(ElasticNet(alpha=best_en_alpha, l1_ratio=best_en_l1ratio, max_iter=10000))

# ================================================================
# 5. CROSS-VALIDATION -- ALL MODELS
# ================================================================
print('\n[4] Running cross-validation on all 7 models...')

model_specs = [
    ('Baseline (mean)',            baseline,    X_all),
    ('OLS (full)',                 ols_full,    X_all),
    ('OLS (transformed)',          ols_trans,   X_trans),
    ('OLS (reduced, no X4)',       ols_reduced, X_reduced),
    (f'Ridge (a={best_ridge_alpha})', ridge_tuned, X_all),
    (f'Lasso (a={best_lasso_alpha})', lasso_tuned, X_all),
    (f'ElasticNet (a={best_en_alpha}, l1={best_en_l1ratio})', en_tuned, X_all),
]

results = {}
for name, model, X in model_specs:
    scores = cv_scores(model, X, y, CV)
    results[name] = scores
    print(f'    {name:<45s}  MAE={scores["MAE"]:.4f}  RMSE={scores["RMSE"]:.4f}  R2={scores["R2"]:.4f}')

# ================================================================
# 6. RESULTS TABLE
# ================================================================
res_df = pd.DataFrame([
    {
        'Model': name,
        'MAE':  round(scores['MAE'],  4),
        'RMSE': round(scores['RMSE'], 4),
        'R2':   round(scores['R2'],   4),
        'MAE_std':  round(scores['MAE_std'],  4),
        'RMSE_std': round(scores['RMSE_std'], 4),
        'R2_std':   round(scores['R2_std'],   4),
    }
    for name, scores in results.items()
])
res_df.to_csv(f'{OUT}/model_comparison.csv', index=False)
print(f'\n[5] Model comparison saved to {OUT}/model_comparison.csv')

# Identify best model by RMSE
best_name = res_df.loc[res_df['RMSE'].idxmin(), 'Model']
print(f'    Best model by RMSE: {best_name}')

# ================================================================
# 7. PAIRED t-TEST: best model vs OLS (fold-level errors)
# ================================================================
print('\n[6] Paired t-test: best model CV errors vs OLS (full) CV errors')

ols_rmse_folds   = results['OLS (full)']['fold_rmse']
best_rmse_folds  = results[best_name]['fold_rmse']

t_stat, p_val = stats.ttest_rel(ols_rmse_folds, best_rmse_folds)
print(f'    OLS mean RMSE  = {ols_rmse_folds.mean():.4f}')
print(f'    Best mean RMSE = {best_rmse_folds.mean():.4f}')
print(f'    t = {t_stat:.4f},  p = {p_val:.4f}')
if p_val < 0.05:
    print('    => p < 0.05: best model is significantly better than plain OLS')
else:
    print('    => p >= 0.05: no statistically significant improvement over plain OLS')

# ================================================================
# 8. FIT FINAL MODEL ON ALL DATA + ASSUMPTION CHECKS
# ================================================================
print(f'\n[7] Fitting final model ({best_name}) on full dataset for diagnostics...')

# Determine which X and model object to use for the best model
spec_map = {spec[0]: (spec[1], spec[2]) for spec in model_specs}
final_model_obj, final_X = spec_map[best_name]
final_model_obj.fit(final_X, y)
y_pred = final_model_obj.predict(final_X)
residuals = y - y_pred

# Also fit OLS with statsmodels for coefficient inference
sm_X = add_constant(X_all)
sm_model = sm.OLS(y, sm_X).fit()
print('\n--- OLS Full Model Summary (statsmodels) ---')
print(sm_model.summary())
sm_model.summary().tables[1].as_csv()

# Save OLS summary
with open(f'{OUT}/ols_full_summary.txt', 'w') as f:
    f.write(str(sm_model.summary()))

# Coefficients table with 95% CI
ci = sm_model.conf_int()
coef_df = pd.DataFrame({
    'Feature':     ['Intercept'] + [SHORT[f] for f in FEATURES],
    'Coefficient': np.round(sm_model.params, 4),
    'Std Error':   np.round(sm_model.bse, 4),
    't-stat':      np.round(sm_model.tvalues, 4),
    'p-value':     np.round(sm_model.pvalues, 4),
    'CI Lower':    np.round(ci[:, 0] if isinstance(ci, np.ndarray) else ci.iloc[:, 0].values, 4),
    'CI Upper':    np.round(ci[:, 1] if isinstance(ci, np.ndarray) else ci.iloc[:, 1].values, 4),
    'Significant': (sm_model.pvalues < 0.05),
})
coef_df.to_csv(f'{OUT}/ols_coefficients.csv', index=False)
print('\nCoefficients and 95% CI:')
print(coef_df.to_string(index=False))

# VIF check
print('\n--- VIF (full 8 features) ---')
vif_df = pd.DataFrame({
    'Feature': [SHORT[f] for f in FEATURES],
    'VIF': [round(variance_inflation_factor(sm_X[:, 1:], i), 3)
            for i in range(len(FEATURES))]
}).sort_values('VIF', ascending=False)
vif_df.to_csv(f'{OUT}/vif_final.csv', index=False)
print(vif_df.to_string(index=False))

# Cook's distance
influence  = sm_model.get_influence()
cooks_d, _ = influence.cooks_distance
influential_threshold = 4 / n
influential_idx = np.where(cooks_d > influential_threshold)[0]
print(f"\nCook's distance threshold (4/n = {influential_threshold:.4f}): "
      f"{len(influential_idx)} potentially influential observations")
for idx in influential_idx[:10]:
    print(f'  Row {idx+1}: {df.iloc[idx]["Home"]} vs {df.iloc[idx]["Away"]}  '
          f'GD={int(y[idx])}  Cook\'s D={cooks_d[idx]:.4f}')

# ================================================================
# 9. CHARTS
# ================================================================
print(f'\n[8] Generating charts to {OUT}/')

# --- Chart 1: Model comparison (MAE, RMSE, R2) ---
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('LR 2.1: Model Comparison (Repeated 5-Fold CV)\nMd Foysal S395914', fontsize=13, fontweight='bold')

short_names = [
    'Baseline', 'OLS\n(full)', 'OLS\n(trans)', 'OLS\n(reduced)',
    'Ridge', 'Lasso', 'ElasticNet'
]
colors = ['#95A5A6' if n != best_name else '#E74C3C' for n in res_df['Model']]

for ax, metric in zip(axes, ['MAE', 'RMSE', 'R2']):
    bars = ax.bar(short_names, res_df[metric], color=colors, edgecolor='white', linewidth=0.8)
    ax.set_title(metric, fontsize=12, fontweight='bold')
    ax.set_ylabel(metric)
    ax.tick_params(axis='x', labelsize=8)
    for bar, val in zip(bars, res_df[metric]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{val:.3f}', ha='center', va='bottom', fontsize=7.5)
    if metric == 'R2':
        ax.axhline(0, color='gray', linewidth=0.8, linestyle='--')

axes[0].set_ylabel('Goals (lower = better)')
axes[2].set_ylabel('R² (higher = better)')
fig.text(0.5, -0.02, 'Red bar = best model by RMSE', ha='center', fontsize=9, color='#E74C3C')
plt.tight_layout()
plt.savefig(f'{OUT}/chart1_model_comparison.png', bbox_inches='tight')
plt.close()

# --- Chart 2: Predicted vs Actual (OLS full, fitted on all data) ---
sm_y_pred = sm_model.fittedvalues
fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(sm_y_pred, y, alpha=0.6, color='#2E5C8A', s=40, edgecolors='white', linewidth=0.4)
lims = [min(sm_y_pred.min(), y.min()) - 0.5, max(sm_y_pred.max(), y.max()) + 0.5]
ax.plot(lims, lims, 'r--', linewidth=1.5, label='Perfect prediction')
ax.set_xlabel('Predicted Goal Difference', fontsize=11)
ax.set_ylabel('Actual Goal Difference', fontsize=11)
ax.set_title('LR 2.1: Predicted vs Actual Goal Difference\n(OLS Full Model, n=104)', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
mae_full = mean_absolute_error(y, sm_y_pred)
rmse_full = mean_squared_error(y, sm_y_pred) ** 0.5
r2_full = r2_score(y, sm_y_pred)
ax.text(0.04, 0.96, f'MAE={mae_full:.3f}  RMSE={rmse_full:.3f}  R²={r2_full:.3f}',
        transform=ax.transAxes, fontsize=9, va='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
plt.tight_layout()
plt.savefig(f'{OUT}/chart2_predicted_vs_actual.png', bbox_inches='tight')
plt.close()

# --- Chart 3: Residual plot (OLS full) ---
sm_residuals = sm_model.resid
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('LR 2.1: Residual Diagnostics (OLS Full Model)', fontsize=13, fontweight='bold')

axes[0].scatter(sm_y_pred, sm_residuals, alpha=0.6, color='#2E5C8A', s=40, edgecolors='white', lw=0.4)
axes[0].axhline(0, color='red', linewidth=1.5, linestyle='--')
axes[0].set_xlabel('Fitted Values', fontsize=11)
axes[0].set_ylabel('Residuals', fontsize=11)
axes[0].set_title('Residuals vs Fitted', fontsize=11)

stats.probplot(sm_residuals, dist='norm', plot=axes[1])
axes[1].set_title('Q-Q Plot of Residuals', fontsize=11)
axes[1].get_lines()[1].set(color='red', linewidth=1.5)

plt.tight_layout()
plt.savefig(f'{OUT}/chart3_residual_diagnostics.png', bbox_inches='tight')
plt.close()

# --- Chart 4: Biggest upsets (matches where model was most wrong) ---
upset_df = df[['Home', 'Away', TARGET]].copy()
upset_df['Predicted'] = np.array(sm_y_pred)
upset_df['Abs Error']  = np.abs(upset_df[TARGET] - upset_df['Predicted'])
upset_df['Matchup']    = upset_df['Home'] + ' vs ' + upset_df['Away']
top_upsets = upset_df.nlargest(10, 'Abs Error').sort_values('Abs Error', ascending=True)

fig, ax = plt.subplots(figsize=(9, 6))
bars = ax.barh(top_upsets['Matchup'], top_upsets['Abs Error'],
               color='#E74C3C', edgecolor='white', linewidth=0.5)
for bar, (_, row) in zip(bars, top_upsets.iterrows()):
    ax.text(bar.get_width() + 0.05,
            bar.get_y() + bar.get_height()/2,
            f'Actual={int(row[TARGET])}  Pred={row["Predicted"]:.1f}',
            va='center', fontsize=8.5)
ax.set_xlabel('Absolute Prediction Error (goals)', fontsize=11)
ax.set_title('LR 2.1: 10 Biggest Upsets\n(Matches the Model Got Most Wrong)', fontsize=12, fontweight='bold')
ax.set_xlim(0, top_upsets['Abs Error'].max() + 1.5)
plt.tight_layout()
plt.savefig(f'{OUT}/chart4_biggest_upsets.png', bbox_inches='tight')
plt.close()

# --- Chart 5: Cook's distance ---
fig, ax = plt.subplots(figsize=(11, 4))
markerline, stemlines, baseline_line = ax.stem(
    range(n), cooks_d, linefmt='C0-', markerfmt='C0o', basefmt='k-'
)
markerline.set_markersize(3)
ax.axhline(influential_threshold, color='red', linestyle='--', linewidth=1.5,
           label=f"Threshold = 4/n = {influential_threshold:.3f}")
ax.set_xlabel("Observation index", fontsize=11)
ax.set_ylabel("Cook's Distance", fontsize=11)
ax.set_title("LR 2.1: Cook's Distance (Influential Observations)", fontsize=12, fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig(f'{OUT}/chart5_cooks_distance.png', bbox_inches='tight')
plt.close()

# --- Chart 6: Coefficient plot (OLS full, 95% CI) ---
coef_plot = coef_df[coef_df['Feature'] != 'Intercept'].copy()
coef_plot = coef_plot.sort_values('Coefficient')
colors_coef = ['#E74C3C' if sig else '#95A5A6' for sig in coef_plot['Significant']]

fig, ax = plt.subplots(figsize=(9, 5.5))
ax.barh(coef_plot['Feature'], coef_plot['Coefficient'], color=colors_coef,
        edgecolor='white', linewidth=0.5)
ax.errorbar(coef_plot['Coefficient'], coef_plot['Feature'],
            xerr=[coef_plot['Coefficient'] - coef_plot['CI Lower'],
                  coef_plot['CI Upper'] - coef_plot['Coefficient']],
            fmt='none', color='black', capsize=4, linewidth=1.2)
ax.axvline(0, color='black', linewidth=1, linestyle='--')
ax.set_xlabel('Coefficient (effect on goal difference per unit)', fontsize=11)
ax.set_title('LR 2.1: OLS Coefficients with 95% Confidence Intervals\n(Red = significant at p < 0.05)',
             fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT}/chart6_coefficients.png', bbox_inches='tight')
plt.close()

# --- Chart 7: Lasso coefficient paths (feature selection view) ---
lasso_alphas = np.logspace(-3, 2, 100)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_all)
coef_paths = []
for a in lasso_alphas:
    m = Lasso(alpha=a, max_iter=10000).fit(X_scaled, y)
    coef_paths.append(m.coef_)
coef_paths = np.array(coef_paths)

fig, ax = plt.subplots(figsize=(10, 5))
for i, feat in enumerate(FEATURES):
    ax.plot(lasso_alphas, coef_paths[:, i], label=SHORT[feat])
ax.axvline(best_lasso_alpha, color='red', linestyle='--', linewidth=1.5,
           label=f'Best alpha = {best_lasso_alpha}')
ax.set_xscale('log')
ax.set_xlabel('Alpha (regularisation strength)', fontsize=11)
ax.set_ylabel('Coefficient', fontsize=11)
ax.set_title('LR 2.1: Lasso Coefficient Paths\n(Features shrink to zero as alpha increases)',
             fontsize=11, fontweight='bold')
ax.legend(fontsize=8.5, loc='upper right')
plt.tight_layout()
plt.savefig(f'{OUT}/chart7_lasso_paths.png', bbox_inches='tight')
plt.close()

print('    chart1_model_comparison.png')
print('    chart2_predicted_vs_actual.png')
print('    chart3_residual_diagnostics.png')
print('    chart4_biggest_upsets.png')
print('    chart5_cooks_distance.png')
print('    chart6_coefficients.png')
print('    chart7_lasso_paths.png')

# ================================================================
# 10. LASSO / ELASTICNET SELECTED FEATURES
# ================================================================
print('\n[9] Lasso feature selection (tuned model):')
lasso_tuned.fit(X_all, y)
lasso_coefs = lasso_tuned.named_steps['model'].coef_
for feat, coef in zip(FEATURES, lasso_coefs):
    status = 'KEPT' if abs(coef) > 1e-6 else 'ZEROED OUT'
    print(f'    {SHORT[feat]:<20s}  coef={coef:+.4f}  [{status}]')

print('\n[10] ElasticNet feature selection (tuned model):')
en_tuned.fit(X_all, y)
en_coefs = en_tuned.named_steps['model'].coef_
for feat, coef in zip(FEATURES, en_coefs):
    status = 'KEPT' if abs(coef) > 1e-6 else 'ZEROED OUT'
    print(f'    {SHORT[feat]:<20s}  coef={coef:+.4f}  [{status}]')

# ================================================================
# 11. FINAL RECOMMENDATION
# ================================================================
print('\n' + '=' * 65)
print('FINAL MODEL RECOMMENDATION')
print('=' * 65)
best_row = res_df[res_df['Model'] == best_name].iloc[0]
print(f'  Recommended model : {best_name}')
print(f'  CV MAE            : {best_row["MAE"]:.4f}  (+/- {best_row["MAE_std"]:.4f})')
print(f'  CV RMSE           : {best_row["RMSE"]:.4f}  (+/- {best_row["RMSE_std"]:.4f})')
print(f'  CV R2             : {best_row["R2"]:.4f}  (+/- {best_row["R2_std"]:.4f})')
print(f'  Paired t-test vs OLS (full):  t={t_stat:.4f}  p={p_val:.4f}')
print(f'  {"Significant improvement over OLS" if p_val < 0.05 else "No significant improvement over OLS -- OLS may be preferred for interpretability"}')
print('=' * 65)
print(f'\nAll outputs written to ./{OUT}/')
print('\nDone. Md Foysal  S395914')
