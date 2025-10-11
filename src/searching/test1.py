import requests
import csv
from datetime import datetime, timedelta

#-------------------------------------------------------------------------
#CONFIG
#-------------------------------------------------------------------------
URL = "https://api.gdeltproject.org/api/v2/doc/doc"
DOMAIN = "cbc.ca"  # you can change this to another domain if needed
START_DATE = datetime(2017, 1, 1)
END_DATE = datetime(2019, 12, 31)
MAX_RECORDS = 250  # per query (GDELT limit)
OUTPUT_FILE = "train_articles.csv"

#-------------------------------------------------------------------------
#Initialize CSV
#-------------------------------------------------------------------------
with open(OUTPUT_FILE, mode="w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["date", "title", "domain", "country"])
    writer.writeheader()

    current_date = START_DATE
    total_count = 0

    # ---------------------------------------------------------------------
    # Loop through each day
    # ---------------------------------------------------------------------
    while current_date <= END_DATE:
        start_str = current_date.strftime("%Y%m%d000000")
        end_str = current_date.strftime("%Y%m%d235959")

        params = {
            "query": f"domain:{DOMAIN}",
            "mode": "ArtList",
            "format": "json",
            "maxrecords": MAX_RECORDS,
            "startdatetime": start_str,
            "enddatetime": end_str
        }

        print(f"📰 {current_date.strftime('%Y-%m-%d')} ...", end=" ", flush=True)
        try:
            response = requests.get(URL, params=params, timeout=30)
            if response.status_code != 200:
                print(f"❌ HTTP {response.status_code}")
                current_date += timedelta(days=1)
                continue
            else:
                data = response.json()
                articles = data.get("articles", [])
                count = 0
                for article in articles:
                    date_str = article.get("seendate", "")[:8]
                    date_fmt = datetime.strptime(date_str, "%Y%m%d").strftime("%Y-%m-%d")
                    title = article.get("title", "")
                    country = article.get("country", "")
                    writer.writerow({
                        "date": date_fmt,
                        "title": title,
                        "domain": DOMAIN,
                        "country": country
                    })
                    count += 1
                    total_count += 1
                print(f"✅ {count} articles")
        except Exception as e:
            print(f"❌ Error: {e}")

