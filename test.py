import pandas as pd

import matplotlib.pyplot as plt
# Load the CSV file
df = pd.read_csv('test-template.csv')
# Only use the first 200 points
df = df.head(1000)
print(df['price_usd_per_mmbtu'].to_list())
# Assuming the column with prices is named 'price'
plt.figure(figsize=(10, 6))
plt.plot(df['price_usd_per_mmbtu'], marker='o', markersize=1)
plt.title('Prices from test-template.csv')
plt.xlabel('Index')
plt.ylabel('Price')
plt.grid(True)
plt.show()