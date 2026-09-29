"""
Helper functions for the PovertyPredictionPeru project.

Sections
--------
1. Google Earth Engine  : zonal statistics at block (manzana) level.
2. Outlier detection    : IQR rule, DBSCAN, Local Outlier Factor, Isolation Forest.
3. Imputation           : mean, KNN and MissForest imputers.
4. Model training       : district-level (UBIGEO) splits, pipelines, nested cross-validation.
5. Ensembles            : out-of-fold predictions, weighted average and stacking.
6. Reporting            : box plots and Excel export.
"""
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed
from scipy.optimize import minimize

from sklearn.base import clone
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Google Earth Engine is only needed by the remote-sensing notebooks.
# Install it with `pip install earthengine-api` (NOT the unrelated `ee` package).
try:
    import ee
except ImportError:
    ee = None


# =============================================================================
# 1. GOOGLE EARTH ENGINE
# =============================================================================

def shp_to_ee_fmt(geodf):
    """GeoDataFrame -> list of GeoJSON features, to build ee.Geometry objects."""
    data = json.loads(geodf.to_json())
    return data['features']


def _reduce_region(img, shapefile, band, scales=(30, 50, 100, 500)):
    """Mean of `band` over `shapefile`, retrying with a coarser scale (m) if EE fails."""
    for scale in scales[:-1]:
        try:
            return img.reduceRegion(reducer=ee.Reducer.mean(), geometry=shapefile,
                                    scale=scale, maxPixels=1e9).get(band)
        except Exception:
            continue
    return img.reduceRegion(reducer=ee.Reducer.mean(), geometry=shapefile,
                            scale=scales[-1], maxPixels=1e9).get(band)


def reduce_mean(img, shapefile, var_name: str, remote_var_name: str):
    """Mean of band `remote_var_name` over the geometry, stored as property `var_name` (with date)."""
    mean = _reduce_region(img, shapefile, remote_var_name)
    return img.set('date', img.date().format()).set(var_name, mean)


def reduce_mean_slope(img, shapefile, var_name: str):
    """Mean slope over the geometry, stored as property `var_name`."""
    return img.set(var_name, _reduce_region(img, shapefile, 'slope'))


def reduce_mean_temperature(img, shapefile, var_name: str):
    """Mean annual temperature (WorldClim bio01) over the geometry, stored as property `var_name`."""
    return img.set(var_name, _reduce_region(img, shapefile, 'bio01'))


# =============================================================================
# 2. OUTLIER DETECTION
# =============================================================================

def detect_outliers_iqr(df, variables: list = []):
    """
    Interquartile-range rule: values outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR] are set to NaN.
    Creates a new column `{variable}OutInter` for each variable.
    """
    for variable in variables:
        Q1 = df[variable].quantile(0.25)
        Q3 = df[variable].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        df[f'{variable}OutInter'] = np.where(
            (df[variable] < lower_bound) | (df[variable] > upper_bound),
            np.nan,
            df[variable]
        )

    return df


def _outliers_in_block(X_block, idx, method, params):
    if method == "dbscan":
        labels = DBSCAN(**params).fit_predict(X_block)
    else:
        labels = LocalOutlierFactor(**params).fit_predict(X_block)
    return idx, (labels == -1).astype(np.int8)


def detect_outliers_multivariate(df, variables, n_blocks=10, eps=0.1, min_samples=10, seed=42):
    """
    Multivariate outlier flags (1 = outlier) with DBSCAN, Local Outlier Factor and Isolation Forest.
    Returns a DataFrame with columns out_dbscan, out_lof, out_iforest (same index as df).
    """
    # Median imputation + scaling (DBSCAN and LOF are distance-based)
    X = df[variables].fillna(df[variables].median())
    X = MinMaxScaler(feature_range=(0, 1)).fit_transform(X)

    # DBSCAN / LOF are run on random blocks (each one representative of the full sample)
    # to keep memory and time manageable with ~400k blocks
    rng = np.random.default_rng(seed)
    blocks = np.array_split(rng.permutation(len(X)), n_blocks)

    out = pd.DataFrame(index=df.index)

    for name, method, params in [
        ("out_dbscan", "dbscan", dict(eps=eps, min_samples=min_samples)),
        ("out_lof",    "lof",    dict(n_neighbors=4, contamination=.1)),
    ]:
        res = Parallel(n_jobs=n_blocks)(
            delayed(_outliers_in_block)(X[idx], idx, method, params) for idx in blocks
        )
        flag = np.zeros(len(X), dtype=np.int8)
        for idx, d in res:          # each block goes back to its original position
            flag[idx] = d
        out[name] = flag

    # Isolation Forest is already fast and parallel (n_jobs=-1): fitted on the full sample
    clf = IsolationForest(n_estimators=100, max_samples='auto', contamination=.12,
                          max_features=7, bootstrap=False, n_jobs=-1,
                          random_state=42, verbose=0)
    out["out_iforest"] = (clf.fit_predict(X) == -1).astype(np.int8)

    return out


# =============================================================================
# 3. IMPUTATION
# =============================================================================

def impute_values(df, variable, method='mean', value=None):
    """Simple imputation of a single column: 'mean', 'median', 'mode' or 'constant' (uses `value`)."""
    if method == 'mean':
        fill = df[variable].mean()
    elif method == 'median':
        fill = df[variable].median()
    elif method == 'mode':
        fill = df[variable].mode()[0]
    elif method == 'constant':
        if value is None:
            raise ValueError("A `value` must be provided for constant imputation.")
        fill = value
    else:
        raise ValueError("Unknown method. Use 'mean', 'median', 'mode' or 'constant'.")

    return df[variable].fillna(fill)


def impute(df_group, method):
    """
    Imputes all columns of `df_group` with 'mean', 'knn' or 'missforest'.
    Returns a DataFrame with the suffix `_{method}` added to each column.
    """
    from missforest import MissForest   # imported here: only needed for this method

    cols, idx = df_group.columns, df_group.index

    if method == "mean":
        res = df_group.fillna(df_group.mean())

    elif method == "knn":
        scaler = MinMaxScaler()                 # KNN is distance-based: scale first
        Xs = scaler.fit_transform(df_group)     # the scaler ignores and keeps NaNs
        Xi = KNNImputer(n_neighbors=10, weights="uniform").fit_transform(Xs)
        res = pd.DataFrame(scaler.inverse_transform(Xi), columns=cols, index=idx)

    else:  # missforest
        Xi = np.asarray(MissForest().fit_transform(df_group))
        res = pd.DataFrame(Xi, columns=cols, index=idx)

    return res.add_suffix(f"_{method}")


# =============================================================================
# 4. MODEL TRAINING
# =============================================================================

def split_by_groups(groups, n_splits, seed):
    """
    Splits the rows into `n_splits` folds, assigning each FULL district (UBIGEO)
    to a single fold. The UBIGEO -> fold assignment is random (controlled by `seed`).

    Returns a list of (idx_train, idx_test) tuples with POSITIONAL indices,
    the format accepted by the `cv` argument of RandomizedSearchCV.

    Note: folds are balanced in number of districts, not rows.
    """
    groups = pd.Series(np.asarray(groups))
    ubigeos = groups.unique()

    rng = np.random.default_rng(seed)
    rng.shuffle(ubigeos)                                 # random order of districts

    # i-th (shuffled) district -> fold i % n_splits
    fold_by_ubigeo = pd.Series(np.arange(len(ubigeos)) % n_splits, index=ubigeos)
    fold = groups.map(fold_by_ubigeo).to_numpy()        # fold of each row

    return [(np.where(fold != k)[0], np.where(fold == k)[0]) for k in range(n_splits)]


def pipe_linear(model):
    """Linear models: median imputation + standardization."""
    return Pipeline([("imputer", SimpleImputer(strategy="median")),
                     ("scaler",  StandardScaler()),
                     ("model",   model)])


def pipe_tree(model):
    """Tree-based models: median imputation, no scaling needed."""
    return Pipeline([("imputer", SimpleImputer(strategy="median")),
                     ("model",   model)])


def regression_metrics(y_true, y_pred):
    return {"r2":   r2_score(y_true, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
            "mae":  mean_absolute_error(y_true, y_pred)}


def nested_cv(name, estimator, params, X, y, groups,
              n_outer=10, n_inner=3, n_iter=20, n_jobs=-1, seed=42):
    """
    OUTER: `n_outer` random folds by district -> out-of-sample R².
    INNER: within the training part of each outer fold, RandomizedSearchCV with
           `n_inner` folds by district -> hyperparameter selection.
    Returns a DataFrame with one row per outer fold.
    """
    outer_splits = split_by_groups(groups, n_outer, seed=seed)
    rows = []

    for i, (tr, te) in enumerate(outer_splits, start=1):
        t0 = time.time()
        X_tr, y_tr, g_tr = X.iloc[tr], y.iloc[tr], groups.iloc[tr]
        X_te, y_te       = X.iloc[te], y.iloc[te]

        if params:
            # Inner folds, also by district (different seed in each outer fold)
            inner_splits = split_by_groups(g_tr, n_inner, seed=seed + i)

            search = RandomizedSearchCV(
                estimator=estimator,
                param_distributions=params,
                n_iter=n_iter,
                scoring="r2",
                cv=inner_splits,        # list of (train, test) -> respects districts
                n_jobs=n_jobs,
                random_state=seed,
                refit=True,             # refits with the best params on all of X_tr
            )
            search.fit(X_tr, y_tr)
            model       = search.best_estimator_
            best_params = search.best_params_
            r2_inner    = search.best_score_
        else:
            # Linear Regression: nothing to tune
            model       = clone(estimator).fit(X_tr, y_tr)
            best_params = {}
            r2_inner    = np.nan

        # Evaluation on the outer fold (data never seen during tuning)
        y_pred = model.predict(X_te)
        rows.append({
            "model":       name,
            "fold":        i,
            "r2_outer":    r2_score(y_te, y_pred),
            "rmse_outer":  np.sqrt(mean_squared_error(y_te, y_pred)),
            "r2_inner":    r2_inner,        # mean inner-CV R² of the best set
            "best_params": str({k.replace("model__", ""): (round(v, 5) if isinstance(v, float) else v)
                                for k, v in best_params.items()}),
            "n_test":      len(te),
            "seconds":     round(time.time() - t0, 1),
        })
        print(f"   {name:<18} fold {i:>2}/{n_outer}  R²={rows[-1]['r2_outer']:.4f}  "
              f"({rows[-1]['seconds']}s)")

    return pd.DataFrame(rows)


def run_nested_cv(models, X, y, groups, folder, **kwargs):
    """Runs `nested_cv` for every model, with a CSV checkpoint per model in `folder`."""
    results = []
    for name, (estimator, params) in models.items():
        path = Path(folder) / f"nested_{name.replace(' ', '_')}.csv"

        if path.exists():                       # checkpoint: already run
            print(f" {name}: loaded from {path}")
            results.append(pd.read_csv(path))
            continue

        print(f"\n {name}")
        df_m = nested_cv(name, estimator, params, X, y, groups, **kwargs)
        df_m.to_csv(path, index=False)          # save as soon as each model finishes
        results.append(df_m)

    return pd.concat(results, ignore_index=True)


def summarize_nested_cv(df_folds, n_outer):
    """
    Summary table of the nested CV (one row per model, sorted by mean R²) and
    model selection with the one-standard-error rule: among the models whose mean
    R² is within 1 SE of the best one, pick the most stable (lowest std).

    Returns (summary, selected_model).
    """
    summary = (df_folds.groupby("model")
               .agg(r2_mean   =("r2_outer", "mean"),
                    r2_std    =("r2_outer", "std"),
                    r2_min    =("r2_outer", "min"),
                    r2_max    =("r2_outer", "max"),
                    rmse_mean =("rmse_outer", "mean"),
                    r2_inner  =("r2_inner", "mean"),
                    n_distinct_params=("best_params", "nunique"),   # hyperparameter stability
                    minutes   =("seconds", lambda s: round(s.sum() / 60, 1)))
               .reset_index())

    # Standard error of the mean outer R²
    summary["r2_se"] = summary["r2_std"] / np.sqrt(n_outer)

    # Tuning optimism: inner R² - outer R².
    # A large gap means the model overfitted the hyperparameter search.
    summary["gap_inner_outer"] = summary["r2_inner"] - summary["r2_mean"]

    # ----- 1-SE rule -----
    best      = summary.loc[summary["r2_mean"].idxmax()]
    threshold = best["r2_mean"] - best["r2_se"]
    summary["candidate_1SE"] = summary["r2_mean"] >= threshold

    selected = (summary[summary["candidate_1SE"]]
                .sort_values(["r2_std", "r2_mean"], ascending=[True, False])
                .iloc[0]["model"])

    summary = summary.sort_values("r2_mean", ascending=False).reset_index(drop=True)
    print(f"Highest mean R² : {best['model']} ({best['r2_mean']:.4f} ± {best['r2_se']:.4f} SE)")
    print(f"1-SE threshold  : mean R² >= {threshold:.4f}")
    print(f"SELECTED MODEL  : {selected}  (lowest std among the candidates)")
    return summary, selected


def fit_final_models(models, X, y, groups, folder, n_folds=5, n_iter=20, n_jobs=-1, seed=42):
    """
    One RandomizedSearchCV per model on the FULL training set (folds by district) to get
    its final hyperparameters. Checkpoint per model in `folder`.

    Returns (fitted_models, final_params): dicts keyed by model name.
    """
    cv_final = split_by_groups(groups, n_folds, seed=seed)

    fitted, final_params = {}, {}
    for name, (estimator, params) in models.items():
        path = Path(folder) / f"final_{name.replace(' ', '_')}.joblib"
        if path.exists():
            fitted[name], final_params[name] = joblib.load(path)
            print(f" {name}: loaded")
            continue

        t0 = time.time()
        if params:
            search = RandomizedSearchCV(estimator, params, n_iter=n_iter, scoring="r2",
                                        cv=cv_final, n_jobs=n_jobs, random_state=seed, refit=True)
            search.fit(X, y)
            fitted[name]       = search.best_estimator_   # already refitted on the full train
            final_params[name] = search.best_params_
        else:
            fitted[name]       = clone(estimator).fit(X, y)
            final_params[name] = {}

        joblib.dump((fitted[name], final_params[name]), path)
        print(f" {name}: done in {(time.time() - t0) / 60:.1f} min")

    return fitted, final_params


def evaluate_on_test(fitted_models, X_test, y_test):
    """Metrics of each fitted model on the FULL test set. Returns (table, predictions)."""
    preds = {name: m.predict(X_test) for name, m in fitted_models.items()}
    table = (pd.DataFrame([{"model": n, **regression_metrics(y_test, p)} for n, p in preds.items()])
             .sort_values("r2", ascending=False).reset_index(drop=True))
    return table, preds


# =============================================================================
# 5. ENSEMBLES (simple average, weighted average, stacking)
# =============================================================================

ENSEMBLES = ["Simple average", "Weighted average", "Stacking"]


def unfitted_copy(fitted_models, name, n_jobs=-1):
    """
    Unfitted copy of a pipeline with its final hyperparameters. There is no parallel
    RandomizedSearchCV here, so the model itself uses all cores.
    """
    m = clone(fitted_models[name])
    if "n_jobs" in m.named_steps["model"].get_params():
        m.set_params(model__n_jobs=n_jobs)
    return m


def base_predictions(base_models, fitted_models, X_tr, y_tr, g_tr, X_ev,
                     seed, n_folds=5, n_jobs=-1, already_fitted=None):
    """
    Returns:
      oof     : (n_tr, n_models) out-of-fold predictions on X_tr (folds by district)
      pred_ev : (n_ev, n_models) predictions on X_ev with models fitted on all of X_tr
    `already_fitted`: optional dict of models ALREADY fitted on X_tr (avoids refitting).
    """
    inner   = split_by_groups(g_tr, n_folds, seed=seed)
    oof     = np.zeros((len(X_tr), len(base_models)))
    pred_ev = np.zeros((len(X_ev), len(base_models)))

    for j, name in enumerate(base_models):
        for tr_i, va_i in inner:
            m = unfitted_copy(fitted_models, name, n_jobs).fit(X_tr.iloc[tr_i], y_tr.iloc[tr_i])
            oof[va_i, j] = m.predict(X_tr.iloc[va_i])

        if already_fitted is not None and name in already_fitted:
            m = already_fitted[name]
        else:
            m = unfitted_copy(fitted_models, name, n_jobs).fit(X_tr, y_tr)
        pred_ev[:, j] = m.predict(X_ev)

    return oof, pred_ev


def optimal_weights(P, y):
    """Weighted-average weights: minimize the MSE of the OOF predictions s.t. w >= 0 and sum(w) = 1."""
    y = np.asarray(y, dtype=float)
    k = P.shape[1]
    res = minimize(lambda w: np.mean((y - P @ w) ** 2),
                   x0=np.full(k, 1 / k),
                   bounds=[(0, 1)] * k,
                   constraints={"type": "eq", "fun": lambda w: w.sum() - 1},
                   method="SLSQP")
    return res.x


def fit_ensembles(oof, y_tr):
    """Learns the weights (weighted average) and the meta-model (stacking) on the OOF predictions."""
    w    = optimal_weights(oof, y_tr)
    meta = LinearRegression(positive=True).fit(oof, y_tr)   # coef >= 0: stable combination
    return w, meta


def predict_all(pred, base_models, w, meta):
    """Predictions of the base models and of the 3 ensembles."""
    out = {n: pred[:, j] for j, n in enumerate(base_models)}
    out["Simple average"]   = pred.mean(axis=1)
    out["Weighted average"] = pred @ w
    out["Stacking"]         = meta.predict(pred)
    return out


def weights_row(label, base_models, w, meta):
    row = {"set": label}
    for j, n in enumerate(base_models):
        row[f"w_avg_{n}"] = w[j]
    row["stack_intercept"] = meta.intercept_
    for j, n in enumerate(base_models):
        row[f"stack_coef_{n}"] = meta.coef_[j]
    return row


def ensemble_cv(base_models, fitted_models, X, y, groups, folder,
                n_outer=10, n_folds=5, n_jobs=-1, seed=42):
    """
    Evaluates base models and ensembles on the SAME outer folds (by district) as the nested CV,
    with fixed hyperparameters. Checkpoint per fold in `folder`.

    Returns (df_folds, weight_rows).
    """
    outer_splits = split_by_groups(groups, n_outer, seed=seed)   # identical to the nested CV

    rows, weight_rows = [], []
    for i, (tr, te) in enumerate(outer_splits, start=1):
        path = Path(folder) / f"cv_fold_{i:02d}.joblib"
        if path.exists():                                 # checkpoint per fold
            f_cv, f_w = joblib.load(path)
            rows += f_cv; weight_rows.append(f_w)
            print(f"   fold {i:>2}/{n_outer}: loaded")
            continue

        t0 = time.time()
        X_tr, y_tr, g_tr = X.iloc[tr], y.iloc[tr], groups.iloc[tr]
        X_te, y_te       = X.iloc[te], y.iloc[te]

        oof, pred_te = base_predictions(base_models, fitted_models, X_tr, y_tr, g_tr, X_te,
                                        seed=seed + i, n_folds=n_folds, n_jobs=n_jobs)
        w, meta = fit_ensembles(oof, y_tr)
        preds   = predict_all(pred_te, base_models, w, meta)

        f_cv = [{"model": n, "type": "Base" if n in base_models else "Ensemble",
                 "fold": i, **regression_metrics(y_te, p), "n_test": len(te)}
                for n, p in preds.items()]
        f_w = weights_row(f"CV fold {i}", base_models, w, meta)
        joblib.dump((f_cv, f_w), path)
        rows += f_cv; weight_rows.append(f_w)

        r2s = "  ".join(f"{d['model'][:10]}={d['r2']:.3f}" for d in f_cv)
        print(f"   fold {i:>2}/{n_outer} ({time.time() - t0:.0f}s)  {r2s}")

    return pd.DataFrame(rows), weight_rows


# =============================================================================
# 6. REPORTING
# =============================================================================

MEANPROPS = {"marker": "D", "markerfacecolor": "darkred", "markeredgecolor": "darkred"}


def boxplot_r2(df, x, y, order, title, path=None, test_r2=None, highlight=None,
               ylabel="R²", figsize=(12, 6)):
    """
    Box plot of the out-of-sample R² per model (one point per fold; red diamond = mean).
    `test_r2`  : optional dict {model: R² on the full test set}, drawn as an orange star.
    `highlight`: optional model name shown in bold on the x-axis.
    """
    fig, ax = plt.subplots(figsize=figsize)
    sns.boxplot(data=df, x=x, y=y, order=order, color="lightsteelblue",
                showmeans=True, meanprops=MEANPROPS, ax=ax)
    sns.stripplot(data=df, x=x, y=y, order=order, color="black", size=4, alpha=0.6, ax=ax)

    if test_r2 is not None:
        ax.scatter(range(len(order)), [test_r2.get(m, np.nan) for m in order],
                   marker="*", s=260, color="darkorange", edgecolor="black",
                   zorder=5, label="Full test set")
        ax.legend(loc="lower left")

    if highlight is not None and highlight in order:
        ax.get_xticklabels()[order.index(highlight)].set_fontweight("bold")

    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    if path is not None:
        plt.savefig(path, dpi=150)
    return fig, ax


def save_excel(sheets, path):
    """Writes a dict {sheet_name: DataFrame} to Excel with bold headers and 4-decimal floats."""
    from openpyxl.styles import Font

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for sheet_name, df in sheets.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            ws = writer.sheets[sheet_name]
            ws.freeze_panes = "A2"
            for cell in ws[1]:
                cell.font = Font(bold=True)
            for col in ws.columns:
                for c in col[1:]:
                    if isinstance(c.value, float):
                        c.number_format = "0.0000"
                width = max(len(str(c.value)) if c.value is not None else 0 for c in col[:200])
                ws.column_dimensions[col[0].column_letter].width = min(width + 2, 45)
