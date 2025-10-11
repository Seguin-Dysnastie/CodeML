import csv

# File path
FILE_PATH = "train_articles.csv"

# Initialize the list
articles = []

# Read CSV and convert to list of dicts
with open(FILE_PATH, mode="r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        # Each row is already a dict with keys matching CSV headers
        articles.append({
            "date": row["date"],
            "title": row["title"],
            "domain": row["domain"],
            "country": row["country"]
        })

# Optional: preview
print(f"Total articles loaded: {len(articles)}")
print(articles[:5])
