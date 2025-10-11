import csv
from datetime import datetime

input_file = 'train_gas_prices_full.csv'
output_file = 'gas_prices_with_returns1_3_7.csv'

# Read and sort data by date
data = []
with open(input_file, newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        row_num = row['price_usd_per_mmbtu']
        price = float(row_num) if row_num and row_num != '0' else 0.0
        data.append({
            'date': row['date'],
            'price': price
        })
data.sort(key=lambda x: datetime.strptime(x['date'], '%Y-%m-%d'))

# Calculate returns
results = []
for i in range(len(data)):
    price_t = data[i]['price']
    date_t = data[i]['date']

    # 1-day return
    if i + 1 < len(data) and price_t != 0:
        r1 = (data[i + 1]['price'] - price_t) / price_t
    else:
        r1 = None

    # 3-day average return
    r3_list = []
    for j in range(1, 4):
        if i + j < len(data) and data[i + j - 1]['price'] != 0:
            r = (data[i + j]['price'] - data[i + j - 1]['price']) / data[i + j - 1]['price']
            r3_list.append(r)
    r3 = sum(r3_list) / len(r3_list) if r3_list else None

    # 7-day average return
    r7_list = []
    for j in range(1, 8):
        if i + j < len(data) and data[i + j - 1]['price'] != 0:
            r = (data[i + j]['price'] - data[i + j - 1]['price']) / data[i + j - 1]['price']
            r7_list.append(r)
    r7 = sum(r7_list) / len(r7_list) if r7_list else None

    results.append({
        'date': date_t,
        'price_usd_per_mmbtu': price_t,
        'return_1day': round(r1, 6) if r1 is not None else '',
        'return_3days': round(r3, 6) if r3 is not None else '',
        'return_7days': round(r7, 6) if r7 is not None else ''
    })

# Write to new CSV
with open(output_file, mode='w', newline='', encoding='utf-8') as f:
    fieldnames = ['date', 'price_usd_per_mmbtu', 'return_1day', 'return_3days', 'return_7days']
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)