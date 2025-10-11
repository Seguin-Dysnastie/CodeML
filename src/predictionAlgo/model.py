import numpy as np
import pandas as pd
import ast
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_absolute_error
import joblib
import warnings
warnings.filterwarnings("ignore")

# Optional: LightGBM
try:
    import lightgbm as lgb
    HAS_LGB = True
except Exception:
    HAS_LGB = False

# ============================================================
# UTILITIES
# ============================================================
def ensure_date(dt):
    if isinstance(dt, str):
        return pd.to_datetime(dt).date()
    if isinstance(dt, pd.Timestamp):
        return dt.date()
    if isinstance(dt, datetime):
        return dt.date()
    return dt

def stack_embeddings(df, emb_col="embedding"):
    return np.vstack(df[emb_col].apply(lambda x: np.array(x, dtype=np.float32)).values)

# ============================================================
# FEATURE ENGINEERING
# ============================================================
def prepare_daily_aggregates(articles_df, emb_col="embedding", date_col="date", n_lags=7, decay=0.7):
    df = articles_df.copy()
    df[date_col] = pd.to_datetime(df[date_col]).dt.date
    emb_arr = stack_embeddings(df, emb_col)
    emb_dim = emb_arr.shape[1]

    rows = []
    for day, sub in df.groupby(date_col):
        emb_mat = np.vstack(sub[emb_col].values)
        emb_mean = emb_mat.mean(axis=0)
        emb_std = emb_mat.std(axis=0)

        mean_norm = emb_mean / (np.linalg.norm(emb_mean) + 1e-9)
        projs = emb_mat.dot(mean_norm)
        att = np.exp(projs - projs.max())
        att = att / (att.sum() + 1e-9)
        att_mean_emb = (att[:, None] * emb_mat).sum(axis=0)

        rows.append({
            "date": day,
            "num_articles": len(sub),
            "emb_mean": emb_mean,
            "emb_std_norm": np.linalg.norm(emb_std),
            "emb_att_mean": att_mean_emb,
            "top_proj_mean": projs.mean(),
        })

    daily = pd.DataFrame(rows).set_index("date").sort_index()

    # Expand embedding arrays
    emb_cols_mean = [f"emb_mean_{i}" for i in range(emb_dim)]
    emb_cols_att = [f"emb_att_{i}" for i in range(emb_dim)]
    daily = pd.concat([
        daily.drop(columns=["emb_mean", "emb_att_mean"]),
        pd.DataFrame(np.vstack(daily["emb_mean"]), index=daily.index, columns=emb_cols_mean),
        pd.DataFrame(np.vstack(daily["emb_att_mean"]), index=daily.index, columns=emb_cols_att)
    ], axis=1)

    # Lag features
    for lag in range(1, n_lags+1):
        prev = daily.shift(lag)
        weight = (decay ** (lag-1))
        cols = [f"emb_att_{i}" for i in range(emb_dim)]
        daily[[f"lag{lag}_emb_{i}" for i in range(emb_dim)]] = prev[cols].fillna(0) * weight

    daily["weekday"] = [d.weekday() for d in daily.index]
    daily["is_weekend"] = daily["weekday"].isin([5,6]).astype(int)

    lag_cols = [f"lag{lag}_emb_{i}" for lag in range(1, n_lags+1) for i in range(emb_dim)]
    return daily, emb_dim, emb_cols_mean, emb_cols_att, lag_cols

# ============================================================
# MODEL TRAINING
# ============================================================
def train_ensemble(X, y, use_lgb=True):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    models = {}

    if use_lgb and HAS_LGB:
        params = {"objective": "regression", "metric": "l1", "verbosity": -1}
        gbm = lgb.train(params, lgb.Dataset(Xs, label=y), num_boost_round=200)
        models["lgb"] = gbm
    else:
        from sklearn.ensemble import RandomForestRegressor
        rf = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
        rf.fit(Xs, y)
        models["rf"] = rf

    sgd = SGDRegressor(max_iter=1000, tol=1e-3, learning_rate="invscaling", eta0=0.01, random_state=42)
    sgd.partial_fit(Xs[:min(100, Xs.shape[0])], y[:min(100, Xs.shape[0])])
    models["sgd_resid"] = sgd
    models["scaler"] = scaler
    return models

def predict_ensemble(models, X):
    scaler = models["scaler"]
    Xs = scaler.transform(X.reshape(1,-1))
    if "lgb" in models:
        base = models["lgb"].predict(Xs)[0]
    else:
        base = models["rf"].predict(Xs)[0]
    resid = models["sgd_resid"].predict(Xs)[0]
    return base + resid

# ============================================================
# PIPELINE FUNCTION
# ============================================================
def run_prediction_pipeline():
    # --- Load news ---
    news = pd.read_csv("train_gas_news_embedded.csv")
    news["date"] = pd.to_datetime(news["date"]).dt.date
    news["embedding"] = news["embedded"].apply(lambda x: np.array(ast.literal_eval(x), dtype=np.float32))
    news["published_at"] = pd.to_datetime(news["date"])
    news["country"] = "us"

    # --- Load price ---
    price = pd.read_csv("gas_prices_with_returns1_3_7.csv")
    price["date"] = pd.to_datetime(price["date"]).dt.date
    price = price.rename(columns={
        "return_1day": "ret_1d",
        "return_3days": "ret_3d",
        "return_7days": "ret_7d"
    }).set_index("date")

    # --- Prepare features ---
    daily, emb_dim, emb_cols_mean, emb_cols_att, lag_cols = prepare_daily_aggregates(news)
    feature_cols = [c for c in daily.columns if (c.startswith("emb_att_") or c.startswith("lag") or c in ['num_articles','weekday','is_weekend','emb_std_norm','top_proj_mean'])]
    target = "ret_3d"

    # --- Align and train ---
    joined = daily.join(price[["ret_1d","ret_3d","ret_7d"]], how="left").dropna(subset=["ret_3d"])
    X, y = joined[feature_cols].values, joined[target].values
    models = train_ensemble(X, y, use_lgb=HAS_LGB)

    # --- Predict for all days (2020–2025) ---
    all_dates = pd.date_range("2020-01-01", "2025-09-22").date
    preds = []
    for d in all_dates:
        if d in daily.index:
            x = daily.loc[d, feature_cols].values.astype(float)
            yhat = predict_ensemble(models, x)
            preds.append({"id": d.isoformat(), "price_usd_per_mmbtu": float(yhat)})
        else:
            preds.append({"id": d.isoformat(), "price_usd_per_mmbtu": np.nan})

    # --- Save predictions ---
    output = pd.DataFrame(preds)
    output.to_csv("gas_price_predictions.csv", index=False)
    print("✅ gas_price_predictions.csv generated successfully.")
    return output

# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    run_prediction_pipeline()
