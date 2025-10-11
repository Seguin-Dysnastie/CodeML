import requests
import csv
from datetime import datetime, timedelta

# -------------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------------
URL = "https://api.gdeltproject.org/api/v2/doc/doc"
START_DATE = datetime(2020, 12, 25)
END_DATE = datetime(2020, 12, 31)
MAX_RECORDS = 20  # per query (GDELT limit)
OUTPUT_FILE = "test_articles.csv"

# -------------------------------------------------------------------------
# Initialize CSV
# -------------------------------------------------------------------------
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

        for domain in ("bloomberg.com", "nytimes.com", "theguardian.com", "forbes.com"):

            params = {
                "query": f"domain:{domain}",
                "sort": "Mentions",
                "mode": "ArtList",
                "format": "json",
                "maxrecords": MAX_RECORDS,
                "startdatetime": start_str,
                "enddatetime": end_str
            }

            print(f"{current_date.strftime('%Y-%m-%d')} - {domain} ...", end=" ", flush=True)
            try:
                response = requests.get(URL, params=params, timeout=30)
                if response.status_code != 200:
                    print(f"❌ HTTP {response.status_code}")
                    current_date += timedelta(days=1)
                    continue

                try:
                    data = response.json()
                except Exception:
                    print("⚠️ No JSON content.")
                    current_date += timedelta(days=1)
                    continue

                articles = data.get("articles", [])
                count = 0

                for item in articles:
                    title = item.get("title", "").strip()
                    if not title:
                        continue  # skip empty titles

                    raw_date = item.get("seendate")
                    formatted_date = None
                    if raw_date:
                        try:
                            formatted_date = datetime.strptime(
                                raw_date, "%Y%m%dT%H%M%SZ"
                            ).strftime("%Y-%m-%d")
                        except ValueError:
                            formatted_date = raw_date

                    writer.writerow({
                        "date": formatted_date,
                        "title": title,
                        "domain": item.get("domain"),
                        "country": item.get("sourcecountry")
                    })
                    count += 1

                total_count += count
                print(f"✅ {count} articles saved. Total so far: {total_count}")

            except Exception as e:
                print(f"⚠️ Error: {e}")

        current_date += timedelta(days=1)

print(f"\n🎯 Done! Total {total_count} articles written to {OUTPUT_FILE}")
