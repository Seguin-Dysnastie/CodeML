from concurrent.futures import ThreadPoolExecutor
import requests
import csv
from datetime import datetime, timedelta

URL = "https://api.gdeltproject.org/api/v2/doc/doc"
START_DATE = datetime(2017, 1, 1)
END_DATE = datetime(2017, 12, 31)
MAX_RECORDS = 20
OUTPUT_FILE = "test_articles.csv"
DOMAINS = ["bloomberg.com", "nytimes.com", "theguardian.com", "forbes.com"]

def fetch_domain_for_day(domain, date):
    start_str = date.strftime("%Y%m%d000000")
    end_str = date.strftime("%Y%m%d235959")
    params = {
        "query": f"domain:{domain}",
        "sort": "Mentions",
        "mode": "ArtList",
        "format": "json",
        "maxrecords": MAX_RECORDS,
        "startdatetime": start_str,
        "enddatetime": end_str
    }
    try:
        resp = requests.get(URL, params=params, timeout=30)
        if resp.status_code != 200:
            return []
        data = resp.json()
        return data.get("articles", [])
    except:
        return []

with open(OUTPUT_FILE, mode="w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["date", "title", "domain", "country"])
    writer.writeheader()

    current_date = START_DATE
    total_count = 0

    while current_date <= END_DATE:
        with ThreadPoolExecutor(max_workers=len(DOMAINS)) as executor:
            futures = [executor.submit(fetch_domain_for_day, d, current_date) for d in DOMAINS]
            results = [f.result() for f in futures]

        day_count = 0
        for articles in results:
            for item in articles:
                title = item.get("title", "").strip()
                if not title:
                    continue
                raw_date = item.get("seendate")
                formatted_date = datetime.strptime(raw_date, "%Y%m%dT%H%M%SZ").strftime("%Y-%m-%d") if raw_date else ""
                writer.writerow({
                    "date": formatted_date,
                    "title": title,
                    "domain": item.get("domain"),
                    "country": item.get("sourcecountry")
                })
                day_count += 1

        total_count += day_count
        print(f"{current_date.strftime('%Y-%m-%d')} ✅ {day_count} articles saved. Total so far: {total_count}")
        current_date += timedelta(days=1)
