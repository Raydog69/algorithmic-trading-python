from datetime import datetime
import matplotlib.pyplot as plt
import requests
import pandas as pd
import math
from scipy import stats

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
end_date = '2023-03-09'
timeframe = '1/hour'
# Fetch data
data = fetch_stock_data(api_key, ticker, start_date, end_date, timeframe)

time_periods = [
            'Three-Month',
            'One-Month',
            '30-Day',
            '5-Day'
            ]

ticker_columns = [
                'Ticker',
                'Price', 
                'Number of Shares to Buy', 
                f'{time_periods[0]} Price Return', 
                f'{time_periods[0]} Return Percentile',
                f'{time_periods[1]} Price Return',
                f'{time_periods[1]} Return Percentile',
                f'{time_periods[2]} Price Return',
                f'{time_periods[2]} Return Percentile',
                f'{time_periods[3]} Price Return',
                f'{time_periods[3]} Return Percentile',
                'HQM Score'
                ]

ticker_dataframe = pd.DataFrame(columns = ticker_columns)

hqm_columns = [
                'Ticker',
                'Price', 
                'Number of Shares to Buy', 
                f'{time_periods[0]} Price Return', 
                f'{time_periods[0]} Return Percentile',
                f'{time_periods[1]} Price Return',
                f'{time_periods[1]} Return Percentile',
                f'{time_periods[2]} Price Return',
                f'{time_periods[2]} Return Percentile',
                f'{time_periods[3]} Price Return',
                f'{time_periods[3]} Return Percentile',
                'HQM Score'
                ]

hqm_dataframe = pd.DataFrame(columns = hqm_columns)

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

IEX_CLOUD_API_TOKEN = 'sk_541d349ea9ee429782bf1e5562bf74fb'
stocks = pd.read_csv('starter_files/sp_500_stocks.csv')
symbol_groups = list(chunks(stocks['Ticker'], 100))
symbol_strings = []
for i in range(0, len(symbol_groups)):
    symbol_strings.append(','.join(symbol_groups[i]))

# Handle data
if data:
    length = len(data['results']) - 1
    new_series1 =pd.Series([ticker, 
                float(data['results'][length]['c']),
                'N/A',
                float(data['results'][length]['c']/data['results'][0]['c'] -1),
                'N/A',
                float(data['results'][length]['c']/data['results'][math.floor(length/3)]['c'] - 1),
                'N/A',
                float(data['results'][length]['c']/data['results'][math.floor(length/3)]['c'] - 1),
                'N/A',
                float(data['results'][length]['c']/data['results'][math.floor(length/18)]['c'] - 1),
                'N/A',
                'N/A'
                ], 
                index = hqm_columns)
    new_row_df = new_series1.to_frame().T
    ticker_dataframe = pd.concat([ticker_dataframe, new_row_df], ignore_index=True)

    for symbol_string in symbol_strings:
    #     print(symbol_strings)
        batch_api_call_url = f'https://cloud.iexapis.com/stable/stock/market/batch/?types=stats,quote&symbols={symbol_string}&token={IEX_CLOUD_API_TOKEN}'
        rel_data = requests.get(batch_api_call_url).json()
        for symbol in symbol_string.split(','):
            try:
                # print(rel_data[symbol]['stats'])
                new_series2 =pd.Series([symbol, 
                    float(rel_data[symbol]['quote']['latestPrice']),
                    'N/A',
                    float(rel_data[symbol]['stats']['month3ChangePercent']),
                    'N/A',
                    float(rel_data[symbol]['stats']['month1ChangePercent']),
                    'N/A',
                    float(rel_data[symbol]['stats']['day30ChangePercent']),
                    'N/A',
                    float(rel_data[symbol]['stats']['day5ChangePercent']),
                    'N/A',
                    'N/A'
                    ], 
                    index = hqm_columns)
                new_row_df = new_series2.to_frame().T
                hqm_dataframe = pd.concat([hqm_dataframe, new_row_df], ignore_index=True)
            except:
                pass

    for row in ticker_dataframe.index:
        for time_period in time_periods:
            price_returns = hqm_dataframe[f'{time_period} Price Return']
            valid_price_returns = [x for x in price_returns if pd.notna(x)]        
            price_return = ticker_dataframe.loc[row, f'{time_period} Price Return']
            if pd.notna(price_return):
                ticker_dataframe.loc[row, f'{time_period} Return Percentile'] = stats.percentileofscore(valid_price_returns, price_return)/100
    from statistics import mean

    for row in ticker_dataframe.index:
        momentum_percentiles = []
        for time_period in time_periods:
            momentum_percentiles.append((ticker_dataframe.loc[row, f'{time_period} Return Percentile']))
        ticker_dataframe.loc[row, 'HQM Score'] = mean(momentum_percentiles)

    position_size = float(100000) / len(ticker_dataframe.index)
    for i in range(len(ticker_dataframe['Ticker'])):
        ticker_dataframe.loc[i, 'Number of Shares to Buy'] = math.floor(position_size / ticker_dataframe['Price'][i])

print(f'HQM for {ticker_dataframe.loc[0, 'Ticker']} is {ticker_dataframe.loc[0, 'HQM Score']}. And the trend for the last 5 days is {ticker_dataframe.loc[0, f'{time_periods[3]} Price Return']}')
