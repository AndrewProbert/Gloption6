import yfinance as yf
import numpy as np
from tabulate import tabulate
import pandas as pd
import matplotlib.pyplot as plt

#create a simple moving average function
def sma(data, period):
    return data.rolling(window=period).mean()

def ema(data, period):
    return data.ewm(span=period, adjust=False).mean()

def dema(data, period):
    ema = data.ewm(span=period, adjust=False).mean()
    dema = 2 * ema - ema.ewm(span=period, adjust=False).mean()
    return dema

def macd(data, period_long, period_short, period_signal):
    ema_long = sma(data, period_long)
    ema_short = sma(data, period_short)
    macd = ema_short - ema_long
    signal = sma(macd, period_signal)
    return macd, signal

def cci(data, period):
    typical_price = (data['High'] + data['Low'] + data['Close']) / 3
    cci = (typical_price - typical_price.rolling(period).mean()) / (0.015 * typical_price.rolling(period).std())
    return cci


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
symbol = 'spy'
ticker = yf.Ticker(symbol) #hibl seems to be the best thus far
start_date = "2000-11-30"
end_date = "2023-11-27"
interval = "1d"
#data = ticker.history(start=start_date, end=end_date, interval='1d') #could try doing hourly with confirmation on daily or weekly
data = ticker.history(start=start_date, interval='1d')
historical_data.append(data)


tradeCheck = False

highPrice = 0
lowPrice = 9999999999999
shares = 0

for i in range(len(historical_data)):
    historical_data[i]['DEMA_100'] = dema(historical_data[i]['Close'], 100)
 

    historical_data[i]['MACD'], historical_data[i]['Signal'] = macd(historical_data[i]['Close'], 26, 12, 9)
    historical_data[i]['MACDSHORT'], historical_data[i]['SignalShort'] = macd(historical_data[i]['Close'], 20, 10, 6)


  


    table = []
    capital = 1000

    for index, row in historical_data[i].iterrows():
        
        date = index
        open_price = row['Open']
        close_price = row['Close']
        volume = row['Volume']
        dema = row['DEMA_100']
  
        macd = row['MACD']
        signal = row['Signal']

        macdshort = row['MACDSHORT']
        signalshort = row['SignalShort']



       

   


        if (tradeOpen == False  and macd > signal and macd < 0 and macdshort > signalshort and macdshort < 0  )   :
            buyPrice = close_price
            buyTime = date
            tradeOpen = True
            shares = capital / buyPrice

            print("Buy at: ", buyPrice, "on: ", buyTime, "Shares: ", shares)
            
        
        elif (tradeOpen == True and ((macdshort > signalshort and macd > macdshort  ) )):
            sellPrice = close_price
            sellTime = date
            tradeOpen = False
            print("Sell at: ", sellPrice, "on: ", sellTime, "Capital: ", capital, "High Price", highPrice, "Low Price", lowPrice)
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

            highPrice = 0
            lowPrice = 9999999999999
            

            if profit > 0:
                positive.append(profit)
            else:
                negative.append(profit)


            # Record profit by year
            year = index.year
            if year not in profit_by_year:
                profit_by_year[year] = []
            profit_by_year[year].append(profit)
        
        elif (tradeOpen == True):
            highPrice = max(highPrice, close_price)
            lowPrice = min(lowPrice, close_price)



        previous_close = close_price
        previous_open = open_price
        previous_volume = volume

        
        table.append([date, open_price, close_price, volume, tradeOpen])

header = ['Date', 'Open', 'Close', 'Volume', 'TradeOpen']
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


print(sum(profitArray))
print(capital)
import matplotlib.pyplot as plt

# ... your existing code ...

# Plotting the capital over time
plt.plot(buyTimeArray, capitalArray)
plt.xlabel('Buy Time')
plt.ylabel('Capital')
plt.title('Capital over Time')
plt.show()
