from datetime import datetime
import matplotlib.pyplot as plt
import requests
import pandas as pd
import math
from scipy import stats
import numpy as np
import pandas as pd
import os
import random
import copy
import matplotlib.pyplot as plt
import pandas
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit 
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib
from sklearn import tree

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
start_date = '2008-01-09'
end_date = '2023-03-09'
timeframe = '1/Day'
# Fetch data
data = fetch_stock_data(api_key, ticker, start_date, end_date, timeframe)

def handle_data():
    data = []
    for filename in filenames:
        df = pd.read_csv(os.path.join('',filename), sep=',')
        label, _, _ = filename.split(sep='.')
        df['Label'] = label
        df['Date'] = pd.to_datetime(df['Date'])

        data.append(df)

    TechIndicator = copy.deepcopy(data)


    # Relative Strength Index
    # Avg(PriceUp)/(Avg(PriceUP)+Avg(PriceDown)*100
    # Where: PriceUp(t)=1*(Price(t)-Price(t-1)){Price(t)- Price(t-1)>0};
    #        PriceDown(t)=-1*(Price(t)-Price(t-1)){Price(t)- Price(t-1)<0};

    def rsi(values):
        up = values[values>0].mean()
        down = -1*values[values<0].mean()
        return 100 * up / (up + down)

    # Add Momentum_1D column for all 15 stocks.
    # Momentum_1D = P(t) - P(t-1)
    for stock in range(len(TechIndicator)):
        TechIndicator[stock]['Momentum_1D'] = (TechIndicator[stock]['Close']-TechIndicator[stock]['Close'].shift(1)).fillna(0)
        TechIndicator[stock]['RSI_14D'] = TechIndicator[stock]['Momentum_1D'].rolling(center=False, window=14).apply(rsi).fillna(0)

    ### Calculation of Volume (Plain)m
    for stock in range(len(TechIndicator)):
        TechIndicator[stock]['Volume_plain'] = TechIndicator[stock]['Volume'].fillna(0)
    TechIndicator[0].tail()

    def bbands(price, length=30, numsd=2):
        """ returns average, upper band, and lower band"""
        #ave = pd.stats.moments.rolling_mean(price,length)
        ave = price.rolling(window = length, center = False).mean()
        #sd = pd.stats.moments.rolling_std(price,length)
        sd = price.rolling(window = length, center = False).std()
        upband = ave + (sd*numsd)
        dnband = ave - (sd*numsd)
        return np.round(ave,3), np.round(upband,3), np.round(dnband,3)

    for stock in range(len(TechIndicator)):
        TechIndicator[stock]['BB_Middle_Band'], TechIndicator[stock]['BB_Upper_Band'], TechIndicator[stock]['BB_Lower_Band'] = bbands(TechIndicator[stock]['Close'], length=20, numsd=1)
        TechIndicator[stock]['BB_Middle_Band'] = TechIndicator[stock]['BB_Middle_Band'].fillna(0)
        TechIndicator[stock]['BB_Upper_Band'] = TechIndicator[stock]['BB_Upper_Band'].fillna(0)
        TechIndicator[stock]['BB_Lower_Band'] = TechIndicator[stock]['BB_Lower_Band'].fillna(0)

    def aroon(df, tf=25):
        aroonup = []
        aroondown = []
        x = tf
        while x< len(df['Date']):
            aroon_up = ((df['High'][x-tf:x].tolist().index(max(df['High'][x-tf:x])))/float(tf))*100
            aroon_down = ((df['Low'][x-tf:x].tolist().index(min(df['Low'][x-tf:x])))/float(tf))*100
            aroonup.append(aroon_up)
            aroondown.append(aroon_down)
            x+=1
        return aroonup, aroondown

    for stock in range(len(TechIndicator)):
        listofzeros = [0] * 25
        up, down = aroon(TechIndicator[stock])
        aroon_list = [x - y for x, y in zip(up,down)]
        if len(aroon_list)==0:
            aroon_list = [0] * TechIndicator[stock].shape[0]
            TechIndicator[stock]['Aroon_Oscillator'] = aroon_list
        else:
            TechIndicator[stock]['Aroon_Oscillator'] = listofzeros+aroon_list

    for stock in range(len(TechIndicator)):
        TechIndicator[stock]["PVT"] = (TechIndicator[stock]['Momentum_1D']/ TechIndicator[stock]['Close'].shift(1))*TechIndicator[stock]['Volume']
        TechIndicator[stock]["PVT"] = TechIndicator[stock]["PVT"]-TechIndicator[stock]["PVT"].shift(1)
        TechIndicator[stock]["PVT"] = TechIndicator[stock]["PVT"].fillna(0)

    def abands(df):
        #df['AB_Middle_Band'] = pd.rolling_mean(df['Close'], 20)
        df['AB_Middle_Band'] = df['Close'].rolling(window = 20, center=False).mean()
        # High * ( 1 + 4 * (High - Low) / (High + Low))
        df['aupband'] = df['High'] * (1 + 4 * (df['High']-df['Low'])/(df['High']+df['Low']))
        df['AB_Upper_Band'] = df['aupband'].rolling(window=20, center=False).mean()
        # Low *(1 - 4 * (High - Low)/ (High + Low))
        df['adownband'] = df['Low'] * (1 - 4 * (df['High']-df['Low'])/(df['High']+df['Low']))
        df['AB_Lower_Band'] = df['adownband'].rolling(window=20, center=False).mean()

    for stock in range(len(TechIndicator)):
        abands(TechIndicator[stock])
        TechIndicator[stock] = TechIndicator[stock].fillna(0)
    TechIndicator[0].tail()

    columns2Drop = ['Momentum_1D', 'aupband', 'adownband']
    for stock in range(len(TechIndicator)):
        TechIndicator[stock] = TechIndicator[stock].drop(labels = columns2Drop, axis=1)
    TechIndicator[0].head()

    def STOK(df, n):
        df['STOK'] = ((df['Close'] - df['Low'].rolling(window=n, center=False).mean()) / (df['High'].rolling(window=n, center=False).max() - df['Low'].rolling(window=n, center=False).min())) * 100
        df['STOD'] = df['STOK'].rolling(window = 3, center=False).mean()

    for stock in range(len(TechIndicator)):
        STOK(TechIndicator[stock], 4)
        TechIndicator[stock] = TechIndicator[stock].fillna(0)

    def CMFlow(df, tf):
        CHMF = []
        MFMs = []
        MFVs = []
        x = tf
        
        while x < len(df['Date']):
            PeriodVolume = 0
            volRange = df['Volume'][x-tf:x]
            for eachVol in volRange:
                PeriodVolume += eachVol
            
            MFM = ((df['Close'][x] - df['Low'][x]) - (df['High'][x] - df['Close'][x])) / (df['High'][x] - df['Low'][x])
            MFV = MFM*PeriodVolume
            
            MFMs.append(MFM)
            MFVs.append(MFV)
            x+=1
        
        y = tf
        while y < len(MFVs):
            PeriodVolume = 0
            volRange = df['Volume'][x-tf:x]
            for eachVol in volRange:
                PeriodVolume += eachVol
            consider = MFVs[y-tf:y]
            tfsMFV = 0
            
            for eachMFV in consider:
                tfsMFV += eachMFV
            
            tfsCMF = tfsMFV/PeriodVolume
            CHMF.append(tfsCMF)
            y+=1
        return CHMF

    for stock in range(len(TechIndicator)):
        listofzeros = [0] * 40
        CHMF = CMFlow(TechIndicator[stock], 20)
        if len(CHMF)==0:
            CHMF = [0] * TechIndicator[stock].shape[0]
            TechIndicator[stock]['Chaikin_MF'] = CHMF
        else:
            TechIndicator[stock]['Chaikin_MF'] = listofzeros+CHMF

    def psar(df, iaf = 0.02, maxaf = 0.2):
        length = len(df)
        dates = (df['Date'])
        high = (df['High'])
        low = (df['Low'])
        close = (df['Close'])
        psar = df['Close'][0:len(df['Close'])]
        psarbull = [None] * length
        psarbear = [None] * length
        bull = True
        af = iaf
        ep = df['Low'][0]
        hp = df['High'][0]
        lp = df['Low'][0]
        for i in range(2,length):
            if bull:
                psar[i] = psar[i - 1] + af * (hp - psar[i - 1])
            else:
                psar[i] = psar[i - 1] + af * (lp - psar[i - 1])
            reverse = False
            if bull:
                if df['Low'][i] < psar[i]:
                    bull = False
                    reverse = True
                    psar[i] = hp
                    lp = df['Low'][i]
                    af = iaf
            else:
                if df['High'][i] > psar[i]:
                    bull = True
                    reverse = True
                    psar[i] = lp
                    hp = df['High'][i]
                    af = iaf
            if not reverse:
                if bull:
                    if df['High'][i] > hp:
                        hp = df['High'][i]
                        af = min(af + iaf, maxaf)
                    if df['Low'][i - 1] < psar[i]:
                        psar[i] = df['Low'][i - 1]
                    if df['Low'][i - 2] < psar[i]:
                        psar[i] = df['Low'][i - 2]
                else:
                    if df['Low'][i] < lp:
                        lp = df['Low'][i]
                        af = min(af + iaf, maxaf)
                    if df['High'][i - 1] > psar[i]:
                        psar[i] = df['High'][i - 1]
                    if df['High'][i - 2] > psar[i]:
                        psar[i] = df['High'][i - 2]
            if bull:
                psarbull[i] = psar[i]
            else:
                psarbear[i] = psar[i]
        #return {"dates":dates, "high":high, "low":low, "close":close, "psar":psar, "psarbear":psarbear, "psarbull":psarbull}
        #return psar, psarbear, psarbull
        df['psar'] = psar
        #df['psarbear'] = psarbear
        #df['psarbull'] = psarbull

    for stock in range(len(TechIndicator)):
        psar(TechIndicator[stock])

    # ROC = [(Close - Close n periods ago) / (Close n periods ago)] * 100
    for stock in range(len(TechIndicator)):
        TechIndicator[stock]['ROC'] = ((TechIndicator[stock]['Close'] - TechIndicator[stock]['Close'].shift(12))/(TechIndicator[stock]['Close'].shift(12)))*100
        TechIndicator[stock] = TechIndicator[stock].fillna(0)

    for stock in range(len(TechIndicator)):
        TechIndicator[stock]['VWAP'] = np.cumsum(TechIndicator[stock]['Volume'] * (TechIndicator[stock]['High'] + TechIndicator[stock]['Low'])/2) / np.cumsum(TechIndicator[stock]['Volume'])
        TechIndicator[stock] = TechIndicator[stock].fillna(0)

    for stock in range(len(TechIndicator)):
        TechIndicator[stock]['Momentum'] = TechIndicator[stock]['Close'] - TechIndicator[stock]['Close'].shift(4)
        TechIndicator[stock] = TechIndicator[stock].fillna(0)

    def CCI(df, n, constant):
        TP = (df['High'] + df['Low'] + df['Close']) / 3
        CCI = pd.Series((TP - TP.rolling(window=n, center=False).mean()) / (constant * TP.rolling(window=n, center=False).std())) #, name = 'CCI_' + str(n))
        return CCI

    for stock in range(len(TechIndicator)):
        TechIndicator[stock]['CCI'] = CCI(TechIndicator[stock], 20, 0.015)
        TechIndicator[stock] = TechIndicator[stock].fillna(0)

    for stock in range(len(TechIndicator)):
        new = (TechIndicator[stock]['Volume'] * (~TechIndicator[stock]['Close'].diff().le(0) * 2 -1)).cumsum()
        TechIndicator[stock]['OBV'] = new

    #Keltner Channel  
    def KELCH(df, n):  
        KelChM = pd.Series(((df['High'] + df['Low'] + df['Close']) / 3).rolling(window =n, center=False).mean(), name = 'KelChM_' + str(n))  
        KelChU = pd.Series(((4 * df['High'] - 2 * df['Low'] + df['Close']) / 3).rolling(window =n, center=False).mean(), name = 'KelChU_' + str(n))  
        KelChD = pd.Series(((-2 * df['High'] + 4 * df['Low'] + df['Close']) / 3).rolling(window =n, center=False).mean(), name = 'KelChD_' + str(n))    
        return KelChM, KelChD, KelChU

    for stock in range(len(TechIndicator)):
        KelchM, KelchD, KelchU = KELCH(TechIndicator[stock], 14)
        TechIndicator[stock]['Kelch_Upper'] = KelchU
        TechIndicator[stock]['Kelch_Middle'] = KelchM
        TechIndicator[stock]['Kelch_Down'] = KelchD
        TechIndicator[stock] = TechIndicator[stock].fillna(0)

    for stock in range(len(TechIndicator)):
        TechIndicator[stock]['EMA'] = TechIndicator[stock]['Close'].ewm(span=3,min_periods=0,adjust=True,ignore_na=False).mean()
        
        TechIndicator[stock] = TechIndicator[stock].fillna(0)

    for stock in range(len(TechIndicator)):
        TechIndicator[stock]['TEMA'] = (3 * TechIndicator[stock]['EMA'] - 3 * TechIndicator[stock]['EMA'] * TechIndicator[stock]['EMA']) + (TechIndicator[stock]['EMA']*TechIndicator[stock]['EMA']*TechIndicator[stock]['EMA'])

    for stock in range(len(TechIndicator)):
        TechIndicator[stock]['HL'] = TechIndicator[stock]['High'] - TechIndicator[stock]['Low']
        TechIndicator[stock]['absHC'] = abs(TechIndicator[stock]['High'] - TechIndicator[stock]['Close'].shift(1))
        TechIndicator[stock]['absLC'] = abs(TechIndicator[stock]['Low'] - TechIndicator[stock]['Close'].shift(1))
        TechIndicator[stock]['TR'] = TechIndicator[stock][['HL','absHC','absLC']].max(axis=1)
        TechIndicator[stock]['ATR'] = TechIndicator[stock]['TR'].rolling(window=14).mean()
        TechIndicator[stock]['NATR'] = (TechIndicator[stock]['ATR'] / TechIndicator[stock]['Close']) *100
        TechIndicator[stock] = TechIndicator[stock].fillna(0)

    def DMI(df, period):
        df['UpMove'] = df['High'] - df['High'].shift(1)
        df['DownMove'] = df['Low'].shift(1) - df['Low']
        df['Zero'] = 0

        df['PlusDM'] = np.where((df['UpMove'] > df['DownMove']) & (df['UpMove'] > df['Zero']), df['UpMove'], 0)
        df['MinusDM'] = np.where((df['UpMove'] < df['DownMove']) & (df['DownMove'] > df['Zero']), df['DownMove'], 0)

        df['plusDI'] = 100 * (df['PlusDM']/df['ATR']).ewm(span=period,min_periods=0,adjust=True,ignore_na=False).mean()
        df['minusDI'] = 100 * (df['MinusDM']/df['ATR']).ewm(span=period,min_periods=0,adjust=True,ignore_na=False).mean()

        df['ADX'] = 100 * (abs((df['plusDI'] - df['minusDI'])/(df['plusDI'] + df['minusDI']))).ewm(span=period,min_periods=0,adjust=True,ignore_na=False).mean()

    for stock in range(len(TechIndicator)):
        DMI(TechIndicator[stock], 14)
        TechIndicator[stock] = TechIndicator[stock].fillna(0)

    columns2Drop = ['UpMove', 'DownMove', 'ATR', 'PlusDM', 'MinusDM', 'Zero', 'EMA', 'HL', 'absHC', 'absLC', 'TR']
    for stock in range(len(TechIndicator)):
        TechIndicator[stock] = TechIndicator[stock].drop(labels = columns2Drop, axis=1)

    for stock in range(len(TechIndicator)):
        TechIndicator[stock]['26_ema'] = TechIndicator[stock]['Close'].ewm(span=26,min_periods=0,adjust=True,ignore_na=False).mean()
        TechIndicator[stock]['12_ema'] = TechIndicator[stock]['Close'].ewm(span=12,min_periods=0,adjust=True,ignore_na=False).mean()
        TechIndicator[stock]['MACD'] = TechIndicator[stock]['12_ema'] - TechIndicator[stock]['26_ema']
        TechIndicator[stock] = TechIndicator[stock].fillna(0)

    def MFI(df):
        # typical price
        df['tp'] = (df['High']+df['Low']+df['Close'])/3
        #raw money flow
        df['rmf'] = df['tp'] * df['Volume']
        
        # positive and negative money flow
        df['pmf'] = np.where(df['tp'] > df['tp'].shift(1), df['tp'], 0)
        df['nmf'] = np.where(df['tp'] < df['tp'].shift(1), df['tp'], 0)

        # money flow ratio
        df['mfr'] = df['pmf'].rolling(window=14,center=False).sum()/df['nmf'].rolling(window=14,center=False).sum()
        df['Money_Flow_Index'] = 100 - 100 / (1 + df['mfr'])

    for stock in range(len(TechIndicator)):
        MFI(TechIndicator[stock])
        TechIndicator[stock] = TechIndicator[stock].fillna(0)

    def ichimoku(df):
        # Turning Line
        period9_high = df['High'].rolling(window=9,center=False).max()
        period9_low = df['Low'].rolling(window=9,center=False).min()
        df['turning_line'] = (period9_high + period9_low) / 2
        
        # Standard Line
        period26_high = df['High'].rolling(window=26,center=False).max()
        period26_low = df['Low'].rolling(window=26,center=False).min()
        df['standard_line'] = (period26_high + period26_low) / 2
        
        # Leading Span 1
        df['ichimoku_span1'] = ((df['turning_line'] + df['standard_line']) / 2).shift(26)
        
        # Leading Span 2
        period52_high = df['High'].rolling(window=52,center=False).max()
        period52_low = df['Low'].rolling(window=52,center=False).min()
        df['ichimoku_span2'] = ((period52_high + period52_low) / 2).shift(26)
        
        # The most current closing price plotted 22 time periods behind (optional)
        df['chikou_span'] = df['Close'].shift(-22) # 22 according to investopedia

    for stock in range(len(TechIndicator)):
        ichimoku(TechIndicator[stock])
        TechIndicator[stock] = TechIndicator[stock].fillna(0)

    def WillR(df):
        highest_high = df['High'].rolling(window=14,center=False).max()
        lowest_low = df['Low'].rolling(window=14,center=False).min()
        df['WillR'] = (-100) * ((highest_high - df['Close']) / (highest_high - lowest_low))

    for stock in range(len(TechIndicator)):
        WillR(TechIndicator[stock])
        TechIndicator[stock] = TechIndicator[stock].fillna(0)

    def MINMAX(df):
        df['MIN_Volume'] = df['Volume'].rolling(window=14,center=False).min()
        df['MAX_Volume'] = df['Volume'].rolling(window=14,center=False).max()

    for stock in range(len(TechIndicator)):
        MINMAX(TechIndicator[stock])
        TechIndicator[stock] = TechIndicator[stock].fillna(0)

    def KAMA(price, n=10, pow1=2, pow2=30):
        ''' kama indicator '''    
        ''' accepts pandas dataframe of prices '''

        absDiffx = abs(price - price.shift(1) )  

        ER_num = abs( price - price.shift(n) )
        ER_den = absDiffx.rolling(window=n,center=False).sum()
        ER = ER_num / ER_den

        sc = ( ER*(2.0/(pow1+1)-2.0/(pow2+1.0))+2/(pow2+1.0) ) ** 2.0


        answer = np.zeros(sc.size)
        N = len(answer)
        first_value = True

        for i in range(N):
            if sc[i] != sc[i]:
                answer[i] = np.nan
            else:
                if first_value:
                    answer[i] = price[i]
                    first_value = False
                else:
                    answer[i] = answer[i-1] + sc[i] * (price[i] - answer[i-1])
        return answer

    for stock in range(len(TechIndicator)):
        TechIndicator[stock]['KAMA'] = KAMA(TechIndicator[stock]['Close'])
        TechIndicator[stock] = TechIndicator[stock].fillna(0)

    columns2Drop = ['26_ema', '12_ema','tp','rmf','pmf','nmf','mfr']
    for stock in range(len(TechIndicator)):
        TechIndicator[stock] = TechIndicator[stock].drop(labels = columns2Drop, axis=1)




# Handle data
if data:
    print(data)
    