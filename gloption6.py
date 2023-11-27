import yfinance as yf
import numpy as np
from tabulate import tabulate
import pandas as pd
import matplotlib.pyplot as plt
symbol = 'spy'


#Quick notes. This program needs to be re written so each day is processed seperately. Also a stop profit needs to be implemented such as -5 precent from the highest price we could have sold for.

def ema_greater_than_knn(ema, knn_ma):
    if ema * 1 > knn_ma:
        return 1
    else:
        return 0


def calculate_ema(price_values, ema_len):
    ema = np.zeros(len(price_values))
    ema[ema_len-1] = np.mean(price_values[:ema_len])
    multiplier = 2 / (ema_len + 1)
    
    for i in range(ema_len, len(price_values)):
        ema[i] = (price_values[i] - ema[i-1]) * multiplier + ema[i-1]

    return ema


def calculate_knn_ma(price_values, ma_len):
    knn_ma = [np.mean(price_values[i-ma_len:i]) for i in range(ma_len, len(price_values))]
    knn_ma = [0]*ma_len + knn_ma
    return knn_ma


def calculate_knn_prediction(price_values, ma_len, num_closest_values=3, smoothing_period=50):
    def mean_of_k_closest(value, target, num_closest):
        closest_values = []
        for i in range(len(value)):
            distances = [abs(target[i] - v) for v in closest_values]
            if len(distances) < num_closest or min(distances) < min(distances):
                closest_values.append(value[i])
            if len(distances) >= num_closest:
                max_dist_index = distances.index(max(distances))
                if distances[max_dist_index] > min(distances):
                    closest_values[max_dist_index] = value[i]
        return sum(closest_values) / len(closest_values)

    knn_ma = [mean_of_k_closest(price_values[i-ma_len:i], price_values[i-ma_len:i], num_closest_values)
              for i in range(ma_len, len(price_values))]

    if len(knn_ma) < smoothing_period:
        return []

    knn_smoothed = np.convolve(knn_ma, np.ones(smoothing_period) / smoothing_period, mode='valid')

    def knn_prediction(price, knn_ma, knn_smoothed):
        pos_count = 0
        neg_count = 0
        min_distance = 1e10
        nearest_index = 0
        
        # Check if there are enough elements in knn_ma and knn_smoothed
        if len(knn_ma) < 2 or len(knn_smoothed) < 2:
            return 0  # Return 0 for neutral if there aren't enough elements
        
        for j in range(1, min(10, len(knn_ma))):
            distance = np.sqrt((knn_ma[j] - price) ** 2)
            if distance < min_distance:
                min_distance = distance
                nearest_index = j
                
                # Check if there are enough elements to compare
                if nearest_index >= 1:
                    if knn_smoothed[nearest_index] > knn_smoothed[nearest_index - 1]:
                        pos_count += 1
                    if knn_smoothed[nearest_index] < knn_smoothed[nearest_index - 1]:
                        neg_count += 1
        
        return 1 if pos_count > neg_count else -1

    knn_predictions = [knn_prediction(price_values[i], knn_ma[i - smoothing_period:i], knn_smoothed[i - smoothing_period:i])
                       for i in range(smoothing_period, len(price_values))]
    return knn_predictions

sma_len = 20 

def calculate_sma(price_values, sma_len):
    sma = [np.mean(price_values[i-sma_len:i]) for i in range(sma_len, len(price_values))]
    sma = [0]*sma_len + sma
    return sma


def calcMACD(data, short_period=12, long_period=26, signal_period=9):
    short_ema = calcEMA(data, short_period)
    long_ema = calcEMA(data, long_period)

    macd = [short - long for short, long in zip(short_ema, long_ema)]

    signal = calcEMA(macd, signal_period)

    return macd, signal

def calcEMA(data, period):
    multiplier = 2 / (period + 1)
    ema = [data[0]]

    for i in range(1, len(data)):
        ema_val = (data[i] - ema[-1]) * multiplier + ema[-1]
        ema.append(ema_val)

    return ema    

def macdCross(macd, signal):
    if macd > signal:
        return 1
    else:
        return 0


def stoch(data, period=50):
    stoch = [np.nan]*period
    for i in range(period, len(data)):
        high = max(data[i-period:i])
        low = min(data[i-period:i])
        stoch.append((data[i] - low) / (high - low))
    return stoch

def calcVWAP(data, volume, period=50):
    vwap = [np.nan]*period
    for i in range(period, len(data)):
        vwap.append(sum(data[i-period:i]*volume[i-period:i]) / sum(volume[i-period:i]))
    return vwap



#Ticker Detailss
historical_data = []
tradeOpen = False
buyPrice = 0
sellPrice = 0
buyTime = None
sellTime = None

buyPriceArray = []
sellPriceArray = []
buyTimeArray = []
sellTimeArray = []
profitArray = []
positive = []
negative = []
profit_by_year = {}
capitalArray = []
percentageArray = []

#I think making a protfolio consisting of HIBL, MSTR, OILU, KOLD
symbol = 'soxl'
ticker = yf.Ticker(symbol) #hibl seems to be the best thus far
start_date = "2023-10-07"
end_date = "2023-11-15"
interval = "1d"
#data = ticker.history(start=start_date, end=end_date, interval='1d') #could try doing hourly with confirmation on daily or weekly
data = ticker.history(start=start_date, interval='1h')
historical_data.append(data)





for i in range(len(historical_data)):
    ma_len = 5
    ema_len_5 = 5
    historical_data[i]['EMA_5'] = calculate_knn_ma(historical_data[i]['Close'], ema_len_5)
    historical_data[i]['KNN_MA'] = calculate_ema(historical_data[i]['Close'], ma_len)
    historical_data[i]['SMA'] = calculate_sma(historical_data[i]['Close'], sma_len)
    historical_data[i]['MACD'], historical_data[i]['Signal'] = calcMACD(historical_data[i]['Close'])
    historical_data[i]['STOCH'] = stoch(historical_data[i]['Close'])
    #historical_data[i]['VWAP'] = calcVWAP(historical_data[i]['Close'], historical_data[i]['Volume'])



    table = []
    capital = 1000

    for index, row in historical_data[i].iterrows():
        
        date = index
        open_price = row['Open']
        close_price = row['Close']
        volume = row['Volume']
        ema = row['EMA_5']
        knn_ma = row['KNN_MA']
        sma = row['SMA']
        MACD = row['MACD']
        Signal = row['Signal']
        MACDConverge = macdCross(MACD, Signal)
        stoch_momentum = row['STOCH']
        #vwap = row['VWAP']

   




        if ema != None and knn_ma != None and sma != None and MACD != None and Signal != None:# and vwap != None:
            KnnEmaX = ema_greater_than_knn(ema, knn_ma)
            TrendConfirmation = ema_greater_than_knn(ema, sma)
            MACDConverge = macdCross(MACD, Signal)
            

        else:
            KnnEmaX = None
            TrendConfirmation = None
            MACDConverge = None
            stoch_momentum = None
            vwap = None


        if (tradeOpen == False and (TrendConfirmation == 1) and (MACDConverge == 0) ) :
            buyPrice = close_price
            buyTime = date
            tradeOpen = True
            shares = capital / buyPrice

            print("Buy at: ", buyPrice, "on: ", buyTime, "Shares: ", shares)
            
            
        elif ((KnnEmaX == 0) and (tradeOpen == True)) or (tradeOpen == True and close_price > open_price and close_price > buyPrice and close_price > previous_close and close_price > previous_open ):
            sellPrice = close_price
            sellTime = date
            tradeOpen = False
            print("Sell at: ", sellPrice, "on: ", sellTime)
            profit = sellPrice - buyPrice
            print("Profit: ", profit)





            buyPriceArray.append(buyPrice)
            sellPriceArray.append(sellPrice)
            buyTimeArray.append(buyTime)
            sellTimeArray.append(sellTime)
            profitArray.append(profit)

            capital = shares * sellPrice
            capitalArray.append(capital)

            percentage = (sellPrice - buyPrice) / buyPrice * 100
            percentageArray.append(percentage)
            

            if profit > 0:
                positive.append(profit)
            else:
                negative.append(profit)


            # Record profit by year
            year = index.year
            if year not in profit_by_year:
                profit_by_year[year] = []
            profit_by_year[year].append(profit)

        previous_close = close_price
        previous_open = open_price


        
        table.append([date, open_price, close_price, volume, ema, knn_ma, KnnEmaX, TrendConfirmation, tradeOpen])

header = ['Date', 'Open', 'Close', 'Volume', 'EMA_5', 'KNN_MA', 'KnnEmaX', 'TrendConfirmation', 'TradeOpen']
output = tabulate(table, headers=header, tablefmt='orgtbl')

print("\n")
headers = ["Buy Price", "Buy Time", "Sell Price", "Sell Time", "Profit", 'Percent', "Capital"]
data = list(zip(buyPriceArray, buyTimeArray, sellPriceArray, sellTimeArray, profitArray, percentageArray, capitalArray))
output += "\n\n" + tabulate(data, headers=headers)
output += "\nTotal Profit: " + str(sum(profitArray))
output += "\nTotal Trades: " + str(len(profitArray))
output += "\nPositive Trades: " + str(len(positive))
output += "\nNegative Trades: " + str(len(negative))
output += "\nAverage Percentage: " + str(sum(percentageArray)/len(percentageArray))
output += "\nSuccess Rate: " + str(len(positive)/len(profitArray)*100) + "%\n"
output += str(capital) + "\n"
for year in profit_by_year:
    output += "Year " + str(year) + " Profit: " + str(sum(profit_by_year[year])) + "\n"


if tradeOpen == True:
    output += "Trade Open " + str(buyPrice) + " " + str(buyTime) + "\n"


with open("outputCheck.txt", "w") as f:
    f.write(output)

print("Output saved to output.txt")


print(output)