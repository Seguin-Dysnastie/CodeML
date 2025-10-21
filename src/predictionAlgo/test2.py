import numpy as np
import pandas as pd
from datetime import timedelta
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
import lightgbm as lgb
from sklearn.metrics.pairwise import cosine_similarity
import joblib

# -------------------------
# Helper utilities
# -------------------------
def ensure_datetime(df, col='published_at'):
    df[col] = pd.to_datetime(df[col], utc=True)
    return df

def day_floor(ts):
    return ts.dt.tz_convert(None).dt.floor('D')  # naive date

# -------------------------
# 1) Aggregate news into daily rows with decayed embeddings & stats
# news_df must contain columns:
#   id,title,description,category,country,published_at,embedding (np.array), etc.
# prices_df must contain daily price and returns: columns index=date -> price, and ret_1d,... ret_7d
# -------------------------
def prepare_daily_features(news_df, prices_df,
                           emb_dim=None,
                           decay_windows = [1,3,7,14,30],
                           decay_half_life_days = 7):
    """
    news_df: each row single article; embedding column contains numpy array
    prices_df: DataFrame with index as dates (daily) and 'price' and ret_1d,... columns
    Returns: daily_df with features and target columns merged
    """
    news_df = news_df.copy()
    news_df = ensure_datetime(news_df, 'published_at')
    news_df['date'] = day_floor(news_df['published_at'])
    if emb_dim is None:
        emb_dim = len(news_df['embedding'].iloc[0])
    # Expand embedding dims into columns for easier aggregation
    emb_matrix = np.vstack(news_df['embedding'].values)
    emb_cols = [f'emb_{i}' for i in range(emb_dim)]
    emb_df = pd.DataFrame(emb_matrix, columns=emb_cols, index=news_df.index)
    news_df = pd.concat([news_df.reset_index(drop=True), emb_df.reset_index(drop=True)], axis=1)
    # Group by day and compute day-level stats
    daily = news_df.groupby('date').agg({
        'id':'count',
        'category': lambda s: s.mode().iloc[0] if len(s)>0 else 'unknown',
        'country': lambda s: s.mode().iloc[0] if len(s)>0 else 'unknown',
        **{c: 'mean' for c in emb_cols}
    }).rename(columns={'id':'articles_count'})
    # Add per-day variance of embeddings (vector norm variance)
    emb_norms = news_df[emb_cols].pow(2).sum(axis=1).pow(0.5)
    daily['emb_norm_std'] = news_df.groupby('date')[emb_norms.name if hasattr(emb_norms,'name') else emb_norms.index].std() if False else news_df.groupby('date').apply(lambda g: g[emb_cols].values.std())
    # For decay aggregates, build a time-indexed mapping of embeddings per article
    # We'll compute for each date t: decayed sum of embeddings of articles with published_at <= t
    all_dates = pd.date_range(start=prices_df.index.min(), end=prices_df.index.max(), freq='D')
    # Precompute per-article date and embedding numpy
    article_dates = news_df['date'].values
    article_embs = news_df[emb_cols].values  # shape (N, D)
    # decay function: weight = 2 ** (-(t - t_i)/half_life)
    half = float(decay_half_life_days)
    daily_features = []
    for t in all_dates:
        # mask of articles published at or before t
        mask = article_dates <= pd.Timestamp(t)
        if mask.sum() == 0:
            # empty features
            base = {'date': t, 'articles_count': 0}
            # zero emb sums
            for i in range(emb_dim):
                base[f'dec_emb_mean_{i}'] = 0.0
            base['decayed_count'] = 0.0
            daily_features.append(base)
            continue
        dt_days = (pd.Timestamp(t) - article_dates[mask]) / np.timedelta64(1,'D')
        weights = 2 ** ( - dt_days / half )
        weights = weights.reshape(-1,1)  # (M,1)
        weighted_emb = (article_embs[mask] * weights).sum(axis=0)
        norm_weight = weights.sum()
        mean_emb = weighted_emb / (norm_weight + 1e-9)
        row = {'date': t, 'articles_count': int(mask.sum()), 'decayed_count': float(norm_weight)}
        for i in range(emb_dim):
            row[f'dec_emb_mean_{i}'] = float(mean_emb[i])
        daily_features.append(row)
    daily_feat_df = pd.DataFrame(daily_features).set_index('date')
    # Merge with daily (direct per-day) and prices_df
    daily_full = pd.concat([daily_feat_df, daily.reindex(daily_feat_df.index)], axis=1)
    # Fill missing article stats with zeros or modes
    daily_full['articles_count'] = daily_full['articles_count'].fillna(0)
    # Merge prices (targets)
    df = daily_full.merge(prices_df[['price','ret_1d','ret_3d','ret_7d']], left_index=True, right_index=True, how='left')
    # Drop days where targets are NaN (end of series)
    df = df.dropna(subset=['ret_1d','ret_3d','ret_7d'])
    return df, emb_cols

# -------------------------
# 2) Dimensionality reduction & scalers
# -------------------------
def fit_pca_scaler(X_emb_matrix, n_components=32):
    pca = PCA(n_components=n_components, random_state=42)
    Xp = pca.fit_transform(X_emb_matrix)
    scaler = StandardScaler()
    Xps = scaler.fit_transform(Xp)
    return pca, scaler

# -------------------------
# 3) Train LightGBM models for each target
# -------------------------
def train_lgb_models(df, target_cols=['ret_1d','ret_3d','ret_7d'],
                     pca=None, scaler=None,
                     emb_prefix='dec_emb_mean_',
                     top_n_pca=32):
    """
    df: daily features (index=date), contains dec_emb_mean_0..D and meta fields
    returns dict of trained models and preprocessing artifacts
    """
    # build X matrix: transform decayed mean embeddings by PCA
    emb_cols = [c for c in df.columns if c.startswith(emb_prefix)]
    X_emb = df[emb_cols].values
    if pca is None or scaler is None:
        pca, scaler = fit_pca_scaler(X_emb, n_components=min(top_n_pca, X_emb.shape[1]))
    Xp = pca.transform(X_emb)
    Xps = scaler.transform(Xp)
    # additional features
    X_meta = df[['articles_count','decayed_count']].fillna(0).values
    X = np.hstack([Xps, X_meta])
    feature_names = [f'pca_{i}' for i in range(Xps.shape[1])] + ['articles_count','decayed_count']
    models = {}
    for target in target_cols:
        y = df[target].values
        # simple time-aware training: use all data but later you should do rolling CV
        lgb_train = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.05, num_leaves=64, random_state=42)
        lgb_train.fit(X, y, early_stopping_rounds=50, eval_set=[(X,y)], verbose=False)
        models[target] = {'model': lgb_train, 'pca': pca, 'scaler': scaler, 'feature_names': feature_names}
    return models

# -------------------------
# 4) Similarity-weighted historical reaction baseline
# Store per-article: embedding, published_date, article-level future returns (the returns for that article, aligned to its date)
# For fast lookup, you will save embeddings_matrix_article (N x D) and article_targets dicts
# -------------------------
class SimilarityReactor:
    def __init__(self, article_embeddings, article_targets, article_dates):
        # article_embeddings: np.array (N,D)
        # article_targets: dict with keys 'ret_1d','ret_3d','ret_7d' -> arrays (N,)
        self.emb = article_embeddings
        self.targets = article_targets
        self.dates = article_dates
        # L2 normalize embeddings for cosine
        norms = np.linalg.norm(self.emb, axis=1, keepdims=True) + 1e-9
        self.emb_normed = self.emb / norms

    def predict(self, new_emb, topk=50, target='ret_3d', min_sim_threshold=0.1):
        vec = np.array(new_emb).reshape(1,-1)
        vec = vec / (np.linalg.norm(vec) + 1e-9)
        sims = (self.emb_normed @ vec.T).ravel()  # cosine similarities
        idx = np.argsort(-sims)[:topk]
        vals = self.targets[target][idx]
        w = sims[idx]
        mask = ~np.isnan(vals)
        if mask.sum() == 0:
            return np.nan, 0.0
        w = w[mask]
        vals = vals[mask]
        if w.sum() == 0 or w.mean() < min_sim_threshold:
            # low confidence
            return np.mean(vals), float(w.sum())
        pred = float(np.dot(w, vals) / w.sum())
        return pred, float(w.sum())

# -------------------------
# 5) Single-article prediction function (combines modules)
# -------------------------
def predict_from_single_article(article_row, history_daily_df, sim_reactor, models, pca, scaler, weight_sim=0.4):
    """
    article_row: dict-like with embedding, country, category, published_at
    history_daily_df: the daily_df up to today- used to compute decayed aggregates including this article
    sim_reactor: SimilarityReactor instance
    models: dict for targets trained earlier
    pca/scaler: from training
    weight_sim: mixture weight of similarity baseline vs ML model
    Returns: dict with predictions for ret_1d, ret_3d, ret_7d and confidences
    """
    # 1) Build daily features for "today" including this new article appended
    # We'll append this article into the articles for the published day and recompute decayed mean embedding for that day
    # For simplicity, use the decayed mean embedding on the published day + existing daily aggregates
    # Extract existing decayed embedding vector for latest date (most recent row)
    latest_date = history_daily_df.index.max()
    # Compute decayed-mean embedding for the new day by simple formula: combine existing decayed mean with new emb with weight 1 (approx)
    # Real compute: recompute decayed mean using same half-life logic; here we approximate by treating new article weight = 1 at day 0.
    existing_row = history_daily_df.loc[latest_date]
    # Build embedding vector for new day's decayed mean:
    # get previous decayed mean embedding (assuming columns dec_emb_mean_*)
    emb_cols = [c for c in history_daily_df.columns if c.startswith('dec_emb_mean_')]
    prev_mean = existing_row[emb_cols].values.astype(float)
    prev_decayed_count = float(existing_row.get('decayed_count', 0.0))
    new_emb = np.array(article_row['embedding']).astype(float)
    # approximate new decayed_count = prev_decayed_count + 1
    new_count = prev_decayed_count + 1.0
    new_mean = (prev_mean * prev_decayed_count + new_emb) / (new_count + 1e-9)
    # transform with pca + scaler
    new_pca = pca.transform(new_mean.reshape(1,-1))
    new_pca_s = scaler.transform(new_pca)
    X_meta = np.array([[existing_row.get('articles_count',0)+1, new_count]])
    X = np.hstack([new_pca_s, X_meta])
    preds = {}
    confs = {}
    for target, mdl_info in models.items():
        model = mdl_info['model']
        pred_model = float(model.predict(X)[0])
        # similarity prediction for the single article
        sim_pred, sim_mass = sim_reactor.predict(new_emb, topk=50, target=target)
        # combine with weights
        if np.isnan(sim_pred):
            final_pred = pred_model
            conf = 0.5
        else:
            final_pred = weight_sim * sim_pred + (1-weight_sim) * pred_model
            conf = min(1.0, sim_mass / 50.0 + 0.1)  # crude conf: similarity mass scaled
        preds[target] = final_pred
        confs[target] = conf
    return {'preds': preds, 'confs': confs, 'model_pred': pred_model, 'sim_pred': sim_pred}

# -------------------------
# 6) Training driver + rolling backtest example
# -------------------------
def rolling_train_test_pipeline(news_df, prices_df, emb_dim=None):
    # Prepare daily features
    daily_df, emb_cols = prepare_daily_features(news_df, prices_df, emb_dim=emb_dim)
    # We'll do a rolling expand-train test
    dates = daily_df.index.sort_values()
    # Keep arrays of historical article-level embeddings and article-level target values
    # Build article-level target arrays:
    # For each article (in news_df), assign the price deltas of its published date
    news_df = ensure_datetime(news_df, 'published_at')
    news_df['date'] = day_floor(news_df['published_at'])
    merged = news_df.merge(daily_df[['ret_1d','ret_3d','ret_7d']], left_on='date', right_index=True, how='left')
    article_embeddings = np.vstack(merged['embedding'].values)
    article_targets = {
        'ret_1d': merged['ret_1d'].values,
        'ret_3d': merged['ret_3d'].values,
        'ret_7d': merged['ret_7d'].values
    }
    article_dates = merged['date'].values
    # Fit pca/scaler on all data (or only training - prefer within rolling loop, simplified here)
    pca, scaler = fit_pca_scaler(daily_df[[c for c in daily_df.columns if c.startswith('dec_emb_mean_')]].values, n_components=min(32, article_embeddings.shape[1]))
    # Train global models on all data (example). In production do rolling retraining.
    models = train_lgb_models(daily_df, pca=pca, scaler=scaler)
    sim_reactor = SimilarityReactor(article_embeddings, article_targets, article_dates)
    # Example evaluation on held-out tail: last 365 days
    train_cut = dates[-365] if len(dates)>365 else dates[int(len(dates)*0.7)]
    train_df = daily_df.loc[:train_cut]
    test_df = daily_df.loc[train_cut + timedelta(days=1):]
    # Refit models using train_df only for realistic eval
    models_train = train_lgb_models(train_df, pca=pca, scaler=scaler)
    # Evaluate
    # build X_train/test
    def build_X_from_daily(daily_sub):
        emb_cols = [c for c in daily_sub.columns if c.startswith('dec_emb_mean_')]
        X_emb = daily_sub[emb_cols].values
        Xp = pca.transform(X_emb)
        Xps = scaler.transform(Xp)
        X_meta = daily_sub[['articles_count','decayed_count']].fillna(0).values
        return np.hstack([Xps, X_meta])
    X_test = build_X_from_daily(test_df)
    results = {}
    for target in ['ret_1d','ret_3d','ret_7d']:
        y_test = test_df[target].values
        y_pred = models_train[target]['model'].predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        results[target] = {'mae': mae}
    # return artifacts
    return {'models': models_train, 'sim_reactor': sim_reactor, 'pca': pca, 'scaler': scaler, 'daily_df': daily_df, 'results': results}

# -------------------------
# 7) Save / load convenience
# -------------------------
def save_artifacts(filename_prefix, artifacts):
    joblib.dump(artifacts['models'], filename_prefix + '_models.pkl')
    joblib.dump(artifacts['sim_reactor'], filename_prefix + '_sim.pkl')
    joblib.dump(artifacts['pca'], filename_prefix + '_pca.pkl')
    joblib.dump(artifacts['scaler'], filename_prefix + '_scaler.pkl')
    artifacts['daily_df'].to_parquet(filename_prefix + '_daily.parquet')
