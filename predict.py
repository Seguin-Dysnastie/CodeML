import csv
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

#! -------------------------------------------------------------------------------------------------------------- !#
#! --------------------------------------- Generate price_diff dictionary --------------------------------------- !#
#! -------------------------------------------------------------------------------------------------------------- !#

with open('train_gas_price.csv', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    rows = list(reader)

prices = {}
for row in rows:
    date = row['date']
    if not row.get('price_usd_per_mmbtu'):
        continue
    try:
        prices[date] = float(row['price_usd_per_mmbtu'])
    except Exception:
        continue

# Convert prices to Series and build 3-day resampled log returns for training
price_series = pd.Series(prices)
price_series.index = pd.to_datetime(price_series.index)
three_day_prices = price_series.resample('3D').last()
three_day_log_returns = np.log(three_day_prices).diff().dropna().reset_index()
three_day_log_returns.columns = ["period_end", "delta_log_price"]

#! -------------------------------------------------------------------------------------------------------------- !#
#! ---------------------------------------- Generate articles dictionary ---------------------------------------- !#
#! -------------------------------------------------------------------------------------------------------------- !#

articles = []
with open("full_train_article.csv", mode="r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        # Ensure required fields exist
        articles.append({
            "date": row.get("date", ""),
            "title": row.get("title", ""),
            "domain": row.get("domain", ""),
            "country": row.get("country", "")
        })

articles_df = pd.DataFrame(articles)
articles_df["date"] = pd.to_datetime(articles_df["date"], errors="coerce")
articles_df = articles_df.dropna(subset=["date"]).reset_index(drop=True)

# Aggregate 3-day windows for each log return
period_texts = []
for _, row in three_day_log_returns.iterrows():
    period_end = row["period_end"]
    period_start = period_end - pd.Timedelta(days=2)
    mask = (articles_df["date"] >= period_start) & (articles_df["date"] <= period_end)
    text = " ".join((articles_df.loc[mask, "title"] + ". " +
                     articles_df.loc[mask, "domain"] + ". " +
                     articles_df.loc[mask, "country"]).tolist())
    period_texts.append(text)

three_day_log_returns["text"] = period_texts
three_day_log_returns = three_day_log_returns[three_day_log_returns["text"].str.strip() != ""]

if three_day_log_returns.empty:
    raise ValueError("No training periods with text available after 3-day aggregation.")

#! -------------------------------------------------------------------------------------------------------------- !#
#! ------------------------------------- Generate embedding and train model ------------------------------------- !#
#! -------------------------------------------------------------------------------------------------------------- !#

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
X = model.encode(three_day_log_returns["text"].tolist(), show_progress_bar=True)
y = three_day_log_returns["delta_log_price"].values

mean_delta = float(np.nanmean(y))
trend = float(np.nanmean(y[-5:])) if len(y) >= 5 else mean_delta
print("Mean 3-day delta:", mean_delta, "Trend:", trend)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
reg = Ridge(alpha=1.0)
reg.fit(X_train, y_train)

y_pred = reg.predict(X_test)
print("3-day model performance:")
print("R²:", round(r2_score(y_test, y_pred), 3))
print("MAE:", round(mean_absolute_error(y_test, y_pred), 6), "\n")

#! -------------------------------------------------------------------------------------------------------------- !#
#! -------------------------------------------- Generate predictions -------------------------------------------- !#
#! -------------------------------------------------------------------------------------------------------------- !#

test_template = pd.read_csv("test-template.csv")
test_template["id"] = pd.to_datetime(test_template["id"], errors="coerce")
test_template = test_template.dropna(subset=["id"]).sort_values("id").reset_index(drop=True)

# Starting price: use 2019-12-31 if present else the last available price in prices
if "2019-12-31" in prices:
    previous_price = float(prices["2019-12-31"])
else:
    # fallback: last known price
    last_price_date = max(prices.keys())
    previous_price = float(prices[last_price_date])

predicted_rows = []

# Build 3-day periods that cover the test range (end inclusive)
periods = pd.date_range(start=test_template["id"].min(), end=test_template["id"].max(), freq='3D')

for period_end in periods:
    period_start = period_end - pd.Timedelta(days=2)

    # Use only previous 3-day articles to predict this period (no future leak)
    # We choose window_end = period_start (i.e. last day available before this period)
    window_start = period_start - pd.Timedelta(days=2)
    window_end = period_start

    mask = (articles_df["date"] >= window_start) & (articles_df["date"] <= window_end)
    text = " ".join((articles_df.loc[mask, "title"] + ". " +
                     articles_df.loc[mask, "domain"] + ". " +
                     articles_df.loc[mask, "country"]).tolist())

    if text.strip():
        embed = model.encode([text])
        log_delta = float(reg.predict(embed)[0])
        # mild trend correction and expanded allowed range
        log_delta = log_delta + 0.2 * trend
        log_delta = float(np.clip(log_delta, -0.15, 0.25))
    else:
        # fallback: mean_delta + small trend push
        log_delta = float(np.clip(mean_delta + 0.2 * trend, -0.15, 0.25))

    # Smooth daily progression inside the 3-day period (exponential weighting)
    days_in_period = list(pd.date_range(period_start, period_end))
    for j, single_date in enumerate(days_in_period):
        # only produce predictions for dates that exist in test_template
        if single_date < test_template["id"].min() or single_date > test_template["id"].max():
            continue
        if single_date not in test_template["id"].values:
            continue
        # exponential weight growing across the 3-day block
        weight = np.exp(j / 1.5) / np.exp((len(days_in_period)-1) / 1.5)
        daily_log_delta = log_delta * weight / max(1, len(days_in_period))
        predicted_price = previous_price * np.exp(daily_log_delta)

        predicted_rows.append({
            "id": single_date.strftime("%Y-%m-%d"),
            "price_usd_per_mmbtu": round(float(predicted_price), 4)
        })

        previous_price = predicted_price  # sequentially use predicted price

# Create DataFrame of predictions
predicted_df = pd.DataFrame(predicted_rows)
if not predicted_df.empty:
    predicted_df["id"] = pd.to_datetime(predicted_df["id"])

# Merge with test_template to preserve original rows order and ensure coverage
final_df = test_template[["id"]].merge(predicted_df, on="id", how="left")

# Ensure the target column exists
if "price_usd_per_mmbtu" not in final_df.columns:
    final_df["price_usd_per_mmbtu"] = np.nan

# Forward-fill then back-fill any remaining missing predictions
final_df["price_usd_per_mmbtu"] = final_df["price_usd_per_mmbtu"].ffill().bfill()

# Round and save
final_df["price_usd_per_mmbtu"] = final_df["price_usd_per_mmbtu"].round(4)
final_df.to_csv("test-template.csv", index=False)

print(f"✅ Predicted prices computed for all {len(final_df)} dates in test-template.csv.")
print("First 5 predictions:")
print(final_df.head(5).to_string(index=False))
print("Last 5 predictions:")
print(final_df.tail(5).to_string(index=False))
