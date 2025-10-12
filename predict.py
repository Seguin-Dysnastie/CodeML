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
    if not row['price_usd_per_mmbtu']:
        continue
    prices[date] = float(row['price_usd_per_mmbtu'])

# Convert prices to DataFrame and weekly aggregation
price_series = pd.Series(prices)
price_series.index = pd.to_datetime(price_series.index)
weekly_prices = price_series.resample('W-FRI').last()  # weekly on Friday
weekly_log_returns = np.log(weekly_prices).diff().dropna().reset_index()
weekly_log_returns.columns = ["week_end", "delta_log_price"]

#! -------------------------------------------------------------------------------------------------------------- !#
#! ---------------------------------------- Generate articles dictionary ---------------------------------------- !#
#! -------------------------------------------------------------------------------------------------------------- !#

articles = []
with open("full_train_article.csv", mode="r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        articles.append({
            "date": row["date"],
            "title": row["title"],
            "domain": row["domain"],
            "country": row["country"]
        })

articles_df = pd.DataFrame(articles)
articles_df["date"] = pd.to_datetime(articles_df["date"])

# Aggregate articles from previous 7 days for each weekly return
weekly_texts = []
for _, row in weekly_log_returns.iterrows():
    week_end = row["week_end"]
    week_start = week_end - pd.Timedelta(days=6)
    mask = (articles_df["date"] >= week_start) & (articles_df["date"] <= week_end)
    text = " ".join((articles_df.loc[mask, "title"] + ". " +
                     articles_df.loc[mask, "domain"] + ". " +
                     articles_df.loc[mask, "country"]).tolist())
    weekly_texts.append(text)

weekly_log_returns["text"] = weekly_texts
weekly_log_returns = weekly_log_returns[weekly_log_returns["text"].str.strip() != ""]  # drop empty weeks

#! -------------------------------------------------------------------------------------------------------------- !#
#! ------------------------------------- Generate embedding and train model ------------------------------------- !#
#! -------------------------------------------------------------------------------------------------------------- !#

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
X = model.encode(weekly_log_returns["text"].tolist(), show_progress_bar=True)
y = weekly_log_returns["delta_log_price"].values

mean_delta = y.mean()
print("Mean weekly delta:", mean_delta)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
reg = Ridge(alpha=1.0)
reg.fit(X_train, y_train)

y_pred = reg.predict(X_test)
print("Weekly model performance:")
print("R²:", round(r2_score(y_test, y_pred), 3))
print("MAE:", round(mean_absolute_error(y_test, y_pred), 6), "\n")

#! -------------------------------------------------------------------------------------------------------------- !#
#! -------------------------------------------- Generate predictions -------------------------------------------- !#
#! -------------------------------------------------------------------------------------------------------------- !#

# Load test-template
test_template = pd.read_csv("test-template.csv")
test_template["id"] = pd.to_datetime(test_template["id"])
test_template = test_template.sort_values("id").reset_index(drop=True)

previous_price = float(prices["2019-12-31"])
predicted_rows = []

# Build weekly ranges covering all test-template dates
weeks = pd.date_range(start=test_template["id"].min(), end=test_template["id"].max(), freq='W-FRI')

for week_end in weeks:
    week_start = week_end - pd.Timedelta(days=6)
    mask = (articles_df["date"] >= week_start) & (articles_df["date"] <= week_end)
    week_text = " ".join((articles_df.loc[mask, "title"] + ". " +
                          articles_df.loc[mask, "domain"] + ". " +
                          articles_df.loc[mask, "country"]).tolist())

    if week_text.strip():
        week_embed = model.encode([week_text])
        weekly_log_delta = reg.predict(week_embed)[0]
        # Bias correction
        weekly_log_delta -= mean_delta
        # Clip extreme changes to ±10% per week
        weekly_log_delta = np.clip(weekly_log_delta, -0.05, 0.15)
    else:
        weekly_log_delta = 0.0

    # Apply weekly delta evenly to days in test_template
    for single_date in pd.date_range(week_start, week_end):
        if single_date < test_template["id"].min() or single_date > test_template["id"].max():
            continue
        if single_date not in test_template["id"].values:
            continue
        predicted_price = previous_price * np.exp(weekly_log_delta / 7)  # distribute delta
        predicted_rows.append({
            "id": single_date.strftime("%Y-%m-%d"),
            "price_usd_per_mmbtu": round(float(predicted_price), 4)
        })
        previous_price = predicted_price

# Ensure all dates in test_template are covered
predicted_df = pd.DataFrame(predicted_rows)
predicted_df["id"] = pd.to_datetime(predicted_df["id"])

# If any dates are missing, fill them with previous known price
full_df = test_template[["id"]].merge(predicted_df, on="id", how="left").fillna(method="ffill")

full_df.to_csv("test-template.csv", index=False)
print(f"✅ Predicted prices computed for all {len(full_df)} dates in test-template.csv.")
