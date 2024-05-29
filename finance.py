from datetime import datetime
import matplotlib.pyplot as plt
import requests
import pandas as pd

def fetch_stock_data(api_key, ticker, start_date, end_date, timeframe):
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/{timeframe}/{start_date}/{end_date}"
    params = {
        'apiKey': api_key
    }
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code}")
        return None

def plot_graph(data):
    timestamps = [datetime.fromtimestamp(result['t'] / 1000) for result in data['results']]
    closing_prices = [result['c'] for result in data['results']]
    plt.plot(timestamps, closing_prices)
    plt.show()

# Parameters
api_key = 'B6aVjCLff5E3KTflUXWVlD7sV3W128hd'
ticker = 'AAPL'
start_date = '2023-01-09'
end_date = '2023-01-12'
timeframe = '10/minute'
# Fetch data
data = fetch_stock_data(api_key, ticker, start_date, end_date, timeframe)

hqm_columns = [
                'Ticker',
                'Company Name',
                'Price', 
                'Number of Shares to Buy', 
                'Two-Day Price Return', 
                'Two-Day Return Percentile',
                'One-Day Price Return',
                'One-Day Return Percentile',
                '12-Hour Price Return',
                '12-Hour Return Percentile',
                '4-Hour Price Return',
                '4-Hour Return Percentile',
                'HQM Score'
                ]

hqm_dataframe = pd.DataFrame(columns = hqm_columns)

# Handle data
if data:
    length = len(data['result'])
    new_series =pd.Series([ticker, 
                data['companyName'],   
                float(data['result'][length]['c']),
                'N/A',
                float(data['result'][length:]['c']/data['result'][0]['c']),
                'N/A',
                float(data['result'][length]['c']/data['result'][length/2]['c']),
                'N/A',
                float(data['result'][length]['c']/data['result'][length/4]['c']),
                'N/A',
                float(data['result'][length]['c']/data['result'][length/12]['c']),
                'N/A',
                'N/A'
                ], 
                index = hqm_columns)
    new_row_df = new_series.to_frame().T
    hqm_dataframe = pd.concat([hqm_dataframe, new_row_df], ignore_index=True)

    print(data)
    plot_graph(data)

    