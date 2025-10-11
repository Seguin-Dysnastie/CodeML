import csv

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

print(price_diff["2019-02-12"])