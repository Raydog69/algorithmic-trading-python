from datetime import datetime
import pandas as pd

data = {'Date': ['2024-05-28', '2024-05-29', '2024-05-30', '2024-05-31'],
        'Value': [10, 20, 30, 40]}
df = pd.DataFrame(data)

# Convert the 'Date' column to datetime
df['Date'] = pd.to_datetime(df['Date'])

today = datetime(2024, 5, 30, 12, 30, 2)
print(today)
print(today.date())

# Find the row where 'Date' matches 'today'
today_row = df[df['Date'].dt.date == today.date()]
print(today_row)