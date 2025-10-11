

# must find https for 10 company news info
# with columns ['date', 'title', 'Country', 'Source']
# cnn.com
# bloomberg.com
# cnbc.com
# reuters.com
# nytimes.com
# theguardian.com
# wsj.com
# forbes.com
# usnews.com

# from 2017 to 2019

# from those 10 companies then find the news per day
# then find for all those each days find the change in price of gas for 1day, 3 days, 7 days (average for the 3days, 7days)
# then associate the news with the 3 prices change then save to csv file
# so in the csv we have ['date', 'title', 'Country', 'Source', 'price_change_1day', 'price_change_3day', 'price_change_7day']


import requests
import csv
from datetime import datetime, timedelta

# Config
URL = "https://api.gdeltproject.org/api/v2/doc/doc"
DOMAINS = [
    "cnn.com", "bloomberg.com", "cnbc.com", "reuters.com",
    "nytimes.com", "theguardian.com", "wsj.com", "forbes.com", "usnews.com"
]
START_DATE = datetime(2017, 1, 1)
END_DATE = datetime(2019, 12, 31)
MAX_ARTICLES_PER_DAY = 20
OUTPUT_FILE = "news_gas_price.csv"
GAS_PRICE_FILE = "train_henry_hub_natural_gas_spot_price_daily 1.csv"

# Load gas price data
gas_prices = {}
with open(GAS_PRICE_FILE, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        date_str = row["date"]
        row_num = row["price_usd_per_mmbtu"]
        if (row_num == ''):
            price = '0'
        else :
            price = float(row_num)
        gas_prices[date_str] = price

def get_price_change(date_str, days):
    date = datetime.strptime(date_str, "%Y-%m-%d")
    future_date = date + timedelta(days=days)
    future_str = future_date.strftime("%Y-%m-%d")
    if date_str in gas_prices and future_str in gas_prices:
        return gas_prices[future_str] - gas_prices[date_str]
    return None

def get_avg_price_change(date_str, days):
    date = datetime.strptime(date_str, "%Y-%m-%d")
    changes = []
    for i in range(1, days+1):
        future_date = date + timedelta(days=i)
        future_str = future_date.strftime("%Y-%m-%d")
        if date_str in gas_prices and future_str in gas_prices:
            changes.append(gas_prices[future_str] - gas_prices[date_str])
    if changes:
        return sum(changes) / len(changes)
    return None

with open(OUTPUT_FILE, mode="w", newline="", encoding="utf-8") as f:
    fieldnames = ["date", "title", "Country", "Source", "price_change_1day", "price_change_3day", "price_change_7day"]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()

    current_date = START_DATE
    while current_date <= END_DATE:
        start_str = current_date.strftime("%Y%m%d000000")
        end_str = current_date.strftime("%Y%m%d235959")
        for domain in DOMAINS:
            params = {
                "query": f"domain:{domain}",
                "mode": "ArtList",
                "format": "json",          # json for easier parsing
                "maxrecords": 250,          # fetch more so we can pick top 20
                "startdatetime": start_str,
                "enddatetime": end_str,
                "sort": "Mentions"          # sort by most mentions / popular
            }
            try:
                response = requests.get(URL, params=params, timeout=30)
                print(response.url)
                if response.status_code != 200:
                    print(f"Failed to fetch data for {domain} on {current_date.strftime('%Y-%m-%d')}")
                    continue
                data = response.json()
                articles = data.get("articles", [])
                # take top MAX_ARTICLES_PER_DAY
                for article in articles[:MAX_ARTICLES_PER_DAY]:
                    date_str = article.get("seendate", "")[:8]
                    date_fmt = datetime.strptime(date_str, "%Y%m%d").strftime("%Y-%m-%d")
                    title = article.get("title", "")
                    country = article.get("country", "")
                    source = domain
                    price_1d = get_price_change(date_fmt, 1)
                    price_3d = get_avg_price_change(date_fmt, 3)
                    price_7d = get_avg_price_change(date_fmt, 7)
                    writer.writerow({
                        "date": date_fmt,
                        "title": title,
                        "Country": country,
                        "Source": source,
                        "price_change_1day": price_1d,
                        "price_change_3day": price_3d,
                        "price_change_7day": price_7d
                    })
            except Exception as e:
                print(f"Error fetching {domain} on {current_date.strftime('%Y-%m-%d')}: {e}")
                continue
        current_date += timedelta(days=1)
