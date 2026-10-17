"""House-price prediction - EVERYTHING IN ONE FILE.

How to run (just this one file; no train.csv needed):

    pip install numpy pandas scikit-learn scipy matplotlib seaborn joblib
    python housing_all_in_one.py

The first run downloads the California Housing data (needs internet once;
scikit-learn keeps a local copy afterwards). Without internet the script
falls back to scikit-learn's built-in Diabetes data and says so loudly.

What it does, in order:
  1. loads the data and inspects it
  2. splits it 80% train / 20% test
  3. draws exploratory charts                -> outputs/figures/
  4. baseline (always predict the average price)
  5. linear regression, then polynomial regression (degree 2, 3, 5)
  6. an extra: degree-5 polynomial + Ridge (a fix for over-fitting)
  7. picks the best model with cross-validation on the TRAIN data only,
     saves it, and evaluates ONCE on the untouched test set
  8. predicts a few houses (you can also describe your own):

    python housing_all_in_one.py --set MedInc=5 --set HouseAge=20 --set Latitude=34
"""
import argparse
import json
import sys
import time
import traceback
import warnings
from pathlib import Path

import matplotlib

if "--show" not in sys.argv:          # no pop-up windows unless you ask for them
    matplotlib.use("Agg")

# --------------------------------------------------------------------------
# Paths and small helpers
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "outputs"
FIG_DIR = OUT_DIR / "figures"
MODEL_DIR = OUT_DIR / "models"
for _d in (FIG_DIR, MODEL_DIR):
    _d.mkdir(parents=True, exist_ok=True)

SEED = 42
RESULTS = {}
SHOW = "--show" in sys.argv
STAGE = "housing"


def header(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def save_fig(name, fig=None):
    import matplotlib.pyplot as plt
    fig = fig or plt.gcf()
    path = FIG_DIR / f"{name}.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    if SHOW:
        plt.show()
    plt.close(fig)
    print(f"   [figure saved] {path.relative_to(ROOT)}")


def log_result(stage, key, value):
    RESULTS.setdefault(stage, {})[key] = value


def write_results():
    path = OUT_DIR / "results.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(RESULTS, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nResults written to {path.relative_to(ROOT)}")


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


# --------------------------------------------------------------------------
# Data loading (online first, offline fallback so the demo never crashes)
# --------------------------------------------------------------------------
def load_data():
    """California Housing. If it cannot be downloaded, use the Diabetes data
    that ships inside scikit-learn so the pipeline can still be demonstrated."""
    try:
        from sklearn.datasets import fetch_california_housing
        d = fetch_california_housing(as_frame=True)
        return d.data, d.target, "California Housing", False
    except Exception as e:
        print("\n" + "!" * 70)
        print(f"! California Housing could not be downloaded ({type(e).__name__}).")
        print("! Using the built-in DIABETES data instead - the numbers below are NOT house prices.")
        print("! Connect to the internet and run again to get the real housing results.")
        print("!" * 70)
        from sklearn.datasets import load_diabetes
        d = load_diabetes(as_frame=True)
        return d.data, d.target, "Diabetes (offline fallback)", True


def money(value, dataset):
    """California's target is in units of $100,000."""
    if dataset.startswith("California"):
        return f"${value * 100_000:,.0f}"
    return f"{value:.1f}"


# --------------------------------------------------------------------------
# Metrics (the five used in the notebook)
# --------------------------------------------------------------------------
def metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    nz = y_true != 0
    rel = (y_true[nz] - y_pred[nz]) / y_true[nz]
    return {
        "R2": round(float(r2_score(y_true, y_pred)), 4),
        "MAE": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "RMSE": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
        "MAPE_%": round(float(np.mean(np.abs(rel)) * 100), 2),
        "RMSPE_%": round(float(np.sqrt(np.mean(rel ** 2)) * 100), 2),
    }


def show_metrics(label, m):
    print(f"  {label:<34} R2={m['R2']:>9.4f}  MAE={m['MAE']:>8.3f}  RMSE={m['RMSE']:>8.3f}"
          f"  MAPE={m['MAPE_%']:>7.1f}%  RMSPE={m['RMSPE_%']:>7.1f}%")


# --------------------------------------------------------------------------
# Stage steps
# --------------------------------------------------------------------------
def load_and_inspect():
    header("1  Load and inspect the data")
    X, y, name, fallback = load_data()
    print(f"Dataset: {name}")
    print(f"Shape: {X.shape[0]} rows x {X.shape[1]} features (+ the target)")
    print("\nFeatures:", list(X.columns))
    print("\nMissing values:", int(X.isnull().sum().sum()) + int(y.isnull().sum()))
    print("\nSummary statistics of the features:")
    print(X.describe().round(2).to_string())
    print(f"\nTarget: mean={y.mean():.3f}  median={y.median():.3f}  min={y.min():.3f}  max={y.max():.3f}")
    corr = X.assign(target=y).corr()["target"].drop("target").sort_values(ascending=False)
    print("\nCorrelation of each feature with the target:")
    print(corr.round(3).to_string())
    log_result(STAGE, "dataset", name)
    log_result(STAGE, "fallback_data", fallback)
    log_result(STAGE, "rows", int(X.shape[0]))
    log_result(STAGE, "features", list(X.columns))
    log_result(STAGE, "target_mean", round(float(y.mean()), 4))
    log_result(STAGE, "correlation_with_target", {k: round(float(v), 3) for k, v in corr.items()})
    return X, y, name, fallback


def split_data(X, y):
    header("2  Split: 80% train / 20% test")
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=SEED)
    print(f"  train: {len(X_tr)} rows   test: {len(X_te)} rows")
    log_result(STAGE, "split_sizes", {"train": len(X_tr), "test": len(X_te)})
    return X_tr, X_te, y_tr, y_te


def explore(X_tr, y_tr, name):
    """EDA on the TRAIN set only (the test set must stay untouched)."""
    header("3  Exploratory data analysis (figures)")
    train = X_tr.assign(target=y_tr)

    plt.figure(figsize=(7, 4))
    sns.histplot(y_tr, bins=40, kde=True, color="#4c72b0")
    plt.xlabel("Target (house value)")
    plt.title(f"Distribution of the target - {name}")
    save_fig("housing_01_target")

    n = X_tr.shape[1]
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 2.6 * rows))
    for ax, c in zip(np.ravel(axes), X_tr.columns):
        ax.hist(X_tr[c], bins=30, color="#55a868")
        ax.set_title(c, fontsize=10)
    for ax in np.ravel(axes)[n:]:
        ax.axis("off")
    fig.suptitle("Distribution of every feature", y=1.0)
    fig.tight_layout()
    save_fig("housing_02_features", fig)

    corr = train.corr()
    plt.figure(figsize=(8, 6.5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, annot_kws={"size": 8})
    plt.title("Correlation between features and the target")
    save_fig("housing_03_correlation")

    ct = corr["target"].drop("target").sort_values()
    plt.figure(figsize=(6.5, 4.5))
    ct.plot.barh(color=["#c44e52" if v < 0 else "#4c72b0" for v in ct])
    plt.axvline(0, color="black", lw=0.8)
    plt.title("Which features move with the target?")
    plt.xlabel("Correlation with the target")
    save_fig("housing_04_corr_with_target")

    if {"Latitude", "Longitude"} <= set(X_tr.columns):
        plt.figure(figsize=(7, 6))
        sc = plt.scatter(X_tr["Longitude"], X_tr["Latitude"], c=y_tr, s=4, cmap="viridis", alpha=0.6)
        plt.colorbar(sc, label="House value")
        plt.xlabel("Longitude"); plt.ylabel("Latitude")
        plt.title("Where the expensive houses are")
        save_fig("housing_05_map")


def baseline(X_tr, y_tr, X_te, y_te):
    header("4  Baseline: always predict the average price (any model must beat this)")
    dummy = DummyRegressor(strategy="mean").fit(X_tr, y_tr)
    m = metrics(y_te, dummy.predict(X_te))
    show_metrics("Predict the mean (test)", m)
    log_result(STAGE, "baseline_test", m)
    return m


def build_candidates():
    """Every model is one Pipeline: scaling is learned on the training part only."""
    cands = {"Linear": make_pipeline(StandardScaler(), LinearRegression())}
    for deg in (2, 3, 5):
        cands[f"Polynomial deg {deg}"] = make_pipeline(
            StandardScaler(), PolynomialFeatures(degree=deg, include_bias=False), LinearRegression())
    cands["Poly deg 5 + Ridge (extra)"] = make_pipeline(
        StandardScaler(), PolynomialFeatures(degree=5, include_bias=False), StandardScaler(),
        RidgeCV(alphas=np.logspace(-2, 4, 13)))
    return cands


def compare_models(X_tr, y_tr, X_te, y_te):
    header("5  Linear and polynomial regression (train vs test vs cross-validation)")
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    print("  (the degree-5 models have ~1,300 features: on the real housing data this step can take 1-3 minutes)\n")
    table, fitted = [], {}
    for label, model in build_candidates().items():
        t0 = time.time()
        cv = cross_val_score(model, X_tr, y_tr, cv=kf, scoring="r2", n_jobs=1).mean()
        model.fit(X_tr, y_tr)
        m_tr = metrics(y_tr, model.predict(X_tr))
        m_te = metrics(y_te, model.predict(X_te))
        show_metrics(f"{label} (train)", m_tr)
        show_metrics(f"{label} (test)", m_te)
        print(f"  {'':<34} 5-fold CV R2 on train = {cv:.4f}   ({time.time() - t0:.0f}s)\n")
        table.append({"model": label, "cv_r2": round(float(cv), 4), "train": m_tr, "test": m_te})
        fitted[label] = model
    print("  -> TEST and CV scores are the honest ones; the TRAIN score alone can mislead.")
    poly_rows = [r for r in table if r["model"].startswith("Polynomial")]
    if len(poly_rows) >= 2 and poly_rows[-1]["train"]["R2"] < poly_rows[0]["train"]["R2"]:
        print("  NOTE: a higher degree scored WORSE even on train. That is not normal overfitting: with ~1,300\n"
              "  polynomial columns and a few extreme houses (e.g. AveOccup up to 1243), the numbers get so huge\n"
              "  that the least-squares solver loses precision. Ridge (the extra row) fixes this.")
    log_result(STAGE, "models", table)
    return table, fitted


def model_figures(table, fitted, X_tr, X_te, y_te, name):
    best_lin = fitted["Linear"]
    pred = best_lin.predict(X_te)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(y_te, pred, s=6, alpha=0.4)
    lo, hi = min(np.min(y_te), pred.min()), max(np.max(y_te), pred.max())
    axes[0].plot([lo, hi], [lo, hi], "r--", label="Perfect prediction")
    axes[0].set_xlabel("Actual"); axes[0].set_ylabel("Predicted")
    axes[0].set_title("Linear regression on the test set"); axes[0].legend()
    resid = np.asarray(y_te) - pred
    axes[1].scatter(pred, resid, s=6, alpha=0.4, color="#dd8452")
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("Error (actual - predicted)")
    axes[1].set_title("Where the linear model is wrong")
    save_fig("housing_06_actual_vs_predicted", fig)

    coefs = pd.Series(best_lin.named_steps["linearregression"].coef_, index=X_tr.columns).sort_values()
    plt.figure(figsize=(6.5, 4.5))
    coefs.plot.barh(color=["#c44e52" if v < 0 else "#4c72b0" for v in coefs])
    plt.axvline(0, color="black", lw=0.8)
    plt.title("Linear model: effect of each feature (standardized)")
    save_fig("housing_07_coefficients")
    log_result(STAGE, "linear_coefficients", {k: round(float(v), 4) for k, v in coefs[::-1].items()})

    labels = [t["model"] for t in table]
    tr = [t["train"]["R2"] for t in table]
    te = [t["test"]["R2"] for t in table]
    floor = -1.0
    x = np.arange(len(labels))
    plt.figure(figsize=(10, 5))
    plt.bar(x - 0.2, tr, 0.4, label="Train R2", color="#4c72b0")
    plt.bar(x + 0.2, np.maximum(te, floor), 0.4, label="Test R2", color="#dd8452")
    for i, v in enumerate(te):
        if v < floor:
            plt.text(i + 0.2, floor + 0.05, f"{v:.1f}\n(cut)", ha="center", fontsize=8)
        else:
            plt.text(i + 0.2, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
        plt.text(i - 0.2, tr[i] + 0.02, f"{tr[i]:.2f}", ha="center", fontsize=8)
    plt.axhline(0, color="black", lw=0.8)
    plt.ylim(floor - 0.1, 1.15)
    plt.xticks(x, labels, rotation=15, ha="right")
    plt.ylabel("R2 (1 = perfect, 0 = same as guessing the average)")
    plt.title("Train vs test: more complex is not always better")
    plt.legend()
    save_fig("housing_08_complexity")


def finalize(table, fitted, X_te, y_te, dataset):
    header("6  Pick the best model by cross-validation, save it, evaluate ONCE on the test set")
    best = max(table, key=lambda t: t["cv_r2"])
    print(f"  Best by cross-validation (train data only): {best['model']}  (CV R2 = {best['cv_r2']:.4f})")
    path = MODEL_DIR / "best_housing_model.joblib"
    joblib.dump(fitted[best["model"]], path)
    print(f"  Saved -> {path}")
    loaded = joblib.load(path)
    m = metrics(y_te, loaded.predict(X_te))
    show_metrics("FINAL test (never seen before)", m)
    if dataset.startswith("California"):
        print(f"  On average the model is off by about {money(m['MAE'], dataset)} per house (MAE).")
    log_result(STAGE, "best_model", best["model"])
    log_result(STAGE, "final_test", m)
    return loaded, best["model"], m


def predict_samples(model, X_te, y_te, dataset):
    header("7  Prediction demo on 5 unseen houses")
    sample = X_te.head(5)
    pred = model.predict(sample)
    out = []
    for (idx, _), p, truth in zip(sample.iterrows(), pred, y_te.head(5)):
        print(f"  house #{idx:<6}  predicted={money(p, dataset):>12}   actual={money(truth, dataset):>12}")
        out.append({"id": int(idx), "predicted": round(float(p), 3), "actual": round(float(truth), 3)})
    log_result(STAGE, "prediction_demo", out)


def predict_custom(model, X_tr, overrides, dataset):
    header("8  Your own house")
    row = X_tr.median().to_dict()
    for item in overrides:
        key, _, val = item.partition("=")
        if key not in row:
            sys.exit(f"Unknown feature '{key}'. Choose from: {', '.join(X_tr.columns)}")
        row[key] = float(val)
    p = float(model.predict(pd.DataFrame([row]))[0])
    print("  Features used (missing ones = typical value of the training data):")
    for k, v in row.items():
        print(f"    {k:<12} {v:g}")
    print(f"\n  Predicted value: {money(p, dataset)}")
    log_result(STAGE, "custom_house", {"features": row, "predicted": round(p, 3)})


def run():
    X, y, name, fallback = load_and_inspect()
    X_tr, X_te, y_tr, y_te = split_data(X, y)
    explore(X_tr, y_tr, name)
    baseline(X_tr, y_tr, X_te, y_te)
    table, fitted = compare_models(X_tr, y_tr, X_te, y_te)
    model_figures(table, fitted, X_tr, X_te, y_te, name)
    model, best_name, final = finalize(table, fitted, X_te, y_te, name)
    predict_samples(model, X_te, y_te, name)
    return model, X_tr, name


# --------------------------------------------------------------------------
# One-page HTML report
# --------------------------------------------------------------------------
CAPTIONS = {
    "housing_01_target": "Distribution of the house values (what we want to predict)",
    "housing_02_features": "Distribution of every input feature",
    "housing_03_correlation": "Correlation between all features and the target",
    "housing_04_corr_with_target": "Which features move most with the target",
    "housing_05_map": "Location of the houses, coloured by value",
    "housing_06_actual_vs_predicted": "Linear model: predicted vs actual (left) and the errors (right)",
    "housing_07_coefficients": "Linear model: how much each feature pushes the prediction",
    "housing_08_complexity": "Train vs test score as the model gets more complex (over-fitting)",
}


def build_report():
    import base64
    import html
    R = RESULTS.get(STAGE, {})
    final = R.get("final_test", {})
    base = R.get("baseline_test", {})
    models = R.get("models", [])

    def mrow(label, m, cv=""):
        return (f"<tr><td>{html.escape(label)}</td><td>{m['R2']:.4f}</td><td>{m['MAE']:.3f}</td>"
                f"<td>{m['RMSE']:.3f}</td><td>{m['MAPE_%']:.1f}%</td><td>{m['RMSPE_%']:.1f}%</td><td>{cv}</td></tr>")

    rows = "".join(mrow(f"{t['model']} (train)", t["train"]) + mrow(f"{t['model']} (test)", t["test"], f"{t['cv_r2']:.4f}")
                   for t in models)
    dataset = R.get("dataset", "?")
    demo = "".join(f"<tr><td>#{d['id']}</td><td>{d['predicted']:.3f}</td><td>{d['actual']:.3f}</td></tr>"
                   for d in R.get("prediction_demo", []))
    figs = ""
    for f in sorted(FIG_DIR.glob("housing_*.png")):
        b64 = base64.b64encode(f.read_bytes()).decode()
        figs += (f"<figure><img src='data:image/png;base64,{b64}'>"
                 f"<figcaption>{html.escape(CAPTIONS.get(f.stem, f.stem))}</figcaption></figure>")
    banner = ""
    if R.get("fallback_data"):
        banner = ("<div class='warn'><b>These numbers are NOT house prices.</b> The California data could not be "
                  "downloaded, so the built-in Diabetes data was used to demonstrate the pipeline. "
                  "Run again with internet for the real housing results.</div>")
    custom = R.get("custom_house")
    custom_html = ""
    if custom:
        custom_html = ("<h2>Your own house</h2><p>Predicted value: <b>%s</b></p>"
                       % html.escape(money(custom["predicted"], dataset)))
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>House price project - results</title><style>
body{{font-family:Segoe UI,Arial,sans-serif;max-width:1000px;margin:24px auto;padding:0 16px;color:#222}}
h1{{margin-bottom:4px}} h2{{margin-top:34px;border-bottom:2px solid #eee;padding-bottom:4px}}
.kpis{{display:flex;gap:12px;flex-wrap:wrap}} .kpi{{background:#f4f6fa;border-radius:10px;padding:14px 20px;min-width:150px}}
.kpi b{{display:block;font-size:26px}} table{{border-collapse:collapse;width:100%;font-size:14px}}
td,th{{border:1px solid #ddd;padding:6px 10px;text-align:left}} th{{background:#f4f6fa}}
figure{{margin:18px 0}} img{{max-width:100%;border:1px solid #eee}} figcaption{{color:#555;font-size:14px}}
.warn{{background:#fff3cd;border:1px solid #e0c36a;padding:12px 16px;border-radius:8px;margin:14px 0}}
</style></head><body>
<h1>House price prediction - results of this run</h1>
<p>Dataset: <b>{html.escape(dataset)}</b> &middot; {R.get('rows','?')} rows. Produced by running <code>housing_all_in_one.py</code>.</p>
{banner}
<div class="kpis">
<div class="kpi"><b>{base.get('R2','?')}</b>baseline R&sup2; (guess the average)</div>
<div class="kpi"><b>{final.get('R2','?')}</b>FINAL test R&sup2;</div>
<div class="kpi"><b>{final.get('MAE','?')}</b>average error (MAE)</div>
<div class="kpi"><b style="font-size:18px">{html.escape(str(R.get('best_model','?')))}</b>best model (by CV)</div></div>
<h2>1. All models: train vs test</h2>
<table><tr><th>Model</th><th>R&sup2;</th><th>MAE</th><th>RMSE</th><th>MAPE</th><th>RMSPE</th><th>CV R&sup2; (train)</th></tr>{rows}</table>
<p>R&sup2; = 1 is perfect and 0 is no better than guessing the average. A model that is great on train and bad on test is over-fitting.
The best model is chosen with cross-validation on the training data only; the test set is used once at the end.</p>
<h2>2. Final result on data the model never saw</h2>
<p style="font-size:20px">Test R&sup2;: <b>{final.get('R2','?')}</b> &middot; MAE: <b>{final.get('MAE','?')}</b> &middot; RMSE: <b>{final.get('RMSE','?')}</b></p>
<h2>3. Prediction demo on 5 unseen houses</h2>
<table><tr><th>House</th><th>Predicted</th><th>Actual</th></tr>{demo}</table>
{custom_html}
<h2>4. All figures</h2>{figs}
</body></html>"""
    path = OUT_DIR / "report.html"
    path.write_text(page, encoding="utf-8")
    print(f"\nReport page (numbers + all figures) -> {path}")
    return path


def open_report(path):
    try:
        import webbrowser
        webbrowser.open(Path(path).resolve().as_uri())
    except Exception:
        pass


def main():
    p = argparse.ArgumentParser(description="House prices: train, compare, save and predict - one file.")
    p.add_argument("--show", action="store_true", help="show figures in windows too")
    p.add_argument("--no-open", action="store_true", help="do not open the report page at the end")
    p.add_argument("--set", action="append", default=[], metavar="Feature=value",
                   help="describe your own house, e.g. --set MedInc=5 --set HouseAge=20")
    a = p.parse_args()

    t0 = time.time()
    try:
        model, X_tr, name = run()
    except Exception:
        traceback.print_exc()
        sys.exit("Run failed - see the error above.")
    if a.set:
        predict_custom(model, X_tr, a.set, name)
    write_results()
    print(f"\nDONE in {time.time() - t0:.0f}s   |   figures: {FIG_DIR}   |   model: {MODEL_DIR}")
    report_path = build_report()
    if not a.no_open and "--show" not in sys.argv:
        open_report(report_path)


if __name__ == "__main__":
    main()
