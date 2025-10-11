import csv
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

#! ############################################################################################################ !#
#! ---------------------------------------- Generate price change dict ---------------------------------------- !#
#! ############################################################################################################ !#
with open('train_gas_price.csv', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    rows = list(reader)

price_diff = {}
for i in range(len(rows) - 1):
    date = rows[i]['date']
    if not rows[i]['price_usd_per_mmbtu'] or not rows[i + 1]['price_usd_per_mmbtu']:
        continue
    price_today = float(rows[i]['price_usd_per_mmbtu'])
    price_next = float(rows[i + 1]['price_usd_per_mmbtu'])
    price_diff[date] = (price_next - price_today)/price_today


#! ############################################################################################################ !#
#! ------------------------------------------ Generate articles dict ------------------------------------------ !#
#! ############################################################################################################ !#

articles = []

# Read CSV and convert to list of dicts
with open("train_articles.csv", mode="r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        # Each row is already a dict with keys matching CSV headers
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



#! ############################################################################################################ !#
#! ----------------------------------------------- Prepare data ----------------------------------------------- !#
#! ############################################################################################################ !#

# Date range for training
START_DATE = "2018-01-01"
END_DATE   = "2018-12-31"

# Convert articles list to DataFrame
df = pd.DataFrame(articles)
df["date"] = pd.to_datetime(df["date"])
df["text"] = df["title"] + ". " + df["domain"] + ". " + df["country"]

# Filter by date range
mask = (df["date"] >= START_DATE) & (df["date"] <= END_DATE)
df = df.loc[mask].copy()

# Aggregate all news per date
daily_news = (
    df.groupby("date")["text"]
    .apply(lambda x: " ".join(x))
    .reset_index()
)

# Convert price_diff dict → DataFrame
price_df = pd.DataFrame(list(price_diff.items()), columns=["date", "delta_price"])
price_df["date"] = pd.to_datetime(price_df["date"])

# Merge news and price data (only keep dates present in both)
merged = pd.merge(daily_news, price_df, on="date", how="inner")

if merged.empty:
    raise ValueError("No overlapping dates between articles and price_diff within the selected range.")

print("Merged dataset:")
print(merged, "\n")

#! ############################################################################################################ !#
#! -------------------------------------------- Generate embedding -------------------------------------------- !#
#! ############################################################################################################ !#

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

X = model.encode(merged["text"].tolist(), show_progress_bar=True)
y = merged["delta_price"].values

#! ############################################################################################################ !#
#! ----------------------------------------------- Train model ------------------------------------------------ !#
#! ############################################################################################################ !#

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
reg = Ridge(alpha=1.0)
reg.fit(X_train, y_train)

# Evaluate
y_pred = reg.predict(X_test)
print("Model performance:")
print("R²:", round(r2_score(y_test, y_pred), 3))
print("MAE:", round(mean_absolute_error(y_test, y_pred), 3), "\n")


#! ############################################################################################################ !#
#! ------------------------------------------- Predict price change ------------------------------------------- !#
#! ############################################################################################################ !#

# Get articles for a specific date
target_date = "2019-02-28"
if target_date in articles_by_date:
    new_articles = articles_by_date[target_date]
else:
    new_articles = []

new_df = pd.DataFrame(new_articles)
new_text = " ".join(new_df["title"] + ". " + new_df["domain"] + ". " + new_df["country"])
new_embed = model.encode([new_text])
predicted_change = reg.predict(new_embed)[0]

print(f"Predicted ΔPrice for {new_df['date'][0]}: {predicted_change:.2f}%")
print(f"Actual ΔPrice for {target_date}: {price_diff.get(target_date, 'N/A')}%")
