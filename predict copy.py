import csv
from datetime import timedelta
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

# -------------------------------------------------------------------------------------------------------------
# Load price_diff and get last known price
# -------------------------------------------------------------------------------------------------------------
with open('train_gas_price.csv', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    rows = list(reader)

price_diff = {}
prices = {}
for i in range(len(rows) - 1):
    date = rows[i]['date']
    if not rows[i]['price_usd_per_mmbtu'] or not rows[i + 1]['price_usd_per_mmbtu']:
        continue
    price_today = float(rows[i]['price_usd_per_mmbtu'])
    price_next = float(rows[i + 1]['price_usd_per_mmbtu'])
    price_diff[date] = (price_next - price_today)/price_today
    prices[date] = price_today

# Get last known price in 2018 to start predictions
last_known_date = max([d for d in prices.keys() if d <= "2018-12-31"])
last_known_price = prices[last_known_date]

# -------------------------------------------------------------------------------------------------------------
# Load articles
# -------------------------------------------------------------------------------------------------------------
articles = []
with open("train_articles.csv", mode="r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        articles.append({
            "date": row["date"],
            "title": row["title"],
            "domain": row["domain"],
            "country": row["country"]
        })

articles_by_date = {}
for article in articles:
    date = article["date"]
    if date not in articles_by_date:
        articles_by_date[date] = []
    articles_by_date[date].append(article)

# -------------------------------------------------------------------------------------------------------------
# Prepare dataset for training (same as before)
# -------------------------------------------------------------------------------------------------------------
START_DATE = "2017-01-01"
END_DATE   = "2019-12-31"

df = pd.DataFrame(articles)
df["date"] = pd.to_datetime(df["date"])
df["text"] = df["title"] + ". " + df["domain"] + ". " + df["country"]

mask = (df["date"] >= START_DATE) & (df["date"] <= END_DATE)
df = df.loc[mask].copy()

daily_news = (
    df.groupby("date")["text"]
    .apply(lambda x: " ".join(x))
    .reset_index()
)

price_df = pd.DataFrame(list(price_diff.items()), columns=["date", "delta_price"])
price_df["date"] = pd.to_datetime(price_df["date"])

merged = pd.merge(daily_news, price_df, on="date", how="inner")
if merged.empty:
    raise ValueError("No overlapping dates between articles and price_diff within the selected range.")

# -------------------------------------------------------------------------------------------------------------
# Generate embeddings and train
# -------------------------------------------------------------------------------------------------------------
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

X = model.encode(merged["text"].tolist(), show_progress_bar=True)
y = merged["delta_price"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
reg = Ridge(alpha=1.0)
reg.fit(X_train, y_train)

y_pred = reg.predict(X_test)
print("Model performance:")
print("R²:", round(r2_score(y_test, y_pred), 3))
print("MAE:", round(mean_absolute_error(y_test, y_pred), 3), "\n")



# -------------------------------------------------------------------------------------------------------------
# Load test-template.csv and full historical prices
# -------------------------------------------------------------------------------------------------------------
test_template = pd.read_csv("test-template.csv")
full_prices = pd.read_csv("train_gas_prices_full.csv")

# Convert dates to datetime
test_template["id"] = pd.to_datetime(test_template["id"])
full_prices["date"] = pd.to_datetime(full_prices["date"])

# Sort full price data
full_prices = full_prices.sort_values("date").reset_index(drop=True)
real_dates = full_prices["date"].tolist()
real_prices = full_prices["price_usd_per_mmbtu"].tolist()

# Helper: find closest real price to any target date
def get_closest_real_price(target_date):
    deltas = [abs((d - target_date).days) for d in real_dates]
    idx = int(np.argmin(deltas))
    return real_prices[idx]

# -------------------------------------------------------------------------------------------------------------
# Generate predictions for every date in test-template.csv
# -------------------------------------------------------------------------------------------------------------
updated_rows = []

for _, row in test_template.iterrows():
    date_obj = row["id"]
    date_str = date_obj.strftime("%Y-%m-%d")

    # Always find the closest real price from historical data
    prev_price = get_closest_real_price(date_obj)

    # Predict delta if articles exist for that day
    if date_str in articles_by_date:
        day_articles = articles_by_date[date_str]
        day_text = " ".join([a["title"] + ". " + a["domain"] + ". " + a["country"] for a in day_articles])
        day_embed = model.encode([day_text])
        predicted_delta = reg.predict(day_embed)[0]
        predicted_price = prev_price * (1 + predicted_delta)
    else:
        # No articles → keep price equal to closest historical price
        predicted_price = prev_price

    updated_rows.append({
        "id": date_str,
        "price_usd_per_mmbtu": round(float(predicted_price), 4)
    })

# -------------------------------------------------------------------------------------------------------------
# Overwrite test-template.csv with predictions
# -------------------------------------------------------------------------------------------------------------
updated_df = pd.DataFrame(updated_rows)
updated_df.to_csv("test-template.csv", index=False)
print(f"✅ All {len(updated_df)} dates now have predicted prices in test-template.csv.")
