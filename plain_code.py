import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from datetime import date, timedelta
import yfinance as fin
from sklearn.ensemble import RandomForestRegressor
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from sklearn.metrics import mean_absolute_percentage_error

def get_data(stock_symbol, data_period='5y', data_interval='1d'):
    stock = fin.Ticker(stock_symbol)
    data = stock.history(period=data_period, interval=data_interval)

    data = data.reset_index()
    data = data[['Date','Open']]
    data = data.rename(columns={'Date':'ds','Open': 'y'})
    data = data.set_index(['ds'])
    data = data.fillna(data['y'].mean())
    return data

def get_stock_fundamentals(stock_symbol):
    stock = fin.Ticker(stock_symbol)
    stock_info = stock.info

    industry = stock_info['industry']
    sector = stock_info['sector']
    longName = stock_info['longName']
    fte = stock_info['fullTimeEmployees']

    total_revenue = stock_info['totalRevenue']
    total_debt = stock_info['totalDebt']

    currentPrice = stock_info['currentPrice']
    fiveYearAvgDividendYield = stock_info['fiveYearAvgDividendYield']
    lastDividendValue = stock_info['lastDividendValue']
    
    fiftyDayAverage = stock_info['fiftyDayAverage']
    twoHundredDayAverage = stock_info['twoHundredDayAverage']

    recommendationKey = stock_info['recommendationKey']
    numberOfAnalystOpinions = stock_info['numberOfAnalystOpinions']

    return [industry, sector, longName, fte, total_revenue, total_debt, currentPrice, fiveYearAvgDividendYield,
            lastDividendValue, fiftyDayAverage, twoHundredDayAverage, recommendationKey, numberOfAnalystOpinions]


def train_test_split(data):
    
    train = data[:-250]
    test = data[-250:]
    return data, train, test

def forecasting(data, train, test):
    forecaster = ForecasterAutoreg(
                regressor = RandomForestRegressor(random_state=10),
                lags      = 30
             )
    forecaster.fit(y=train.y)
    predictions = forecaster.predict(steps=len(test))
    predictions.index=test.index

    train.plot(label='train', figsize=(20,7))
    test.plot(label='test', figsize=(20,7))
    predictions.plot(label='predictions', figsize=(20,7))
    plt.legend()
    plt.show()

    mape = mean_absolute_percentage_error(
                y_true = test,
                y_pred = predictions
            )
    
    accuracy = 100 - (mape * 100)

    forecaster.fit(y=data.y)
    final_predictions = forecaster.predict(steps=90)
    date_range = pd.date_range(data.index[-1] + timedelta(days=1), data.index[-1] + timedelta(days=90),freq='d')
    final_predictions.index = date_range

    return accuracy, final_predictions


if __name__ == "__main__":
    data = get_data('ITC.NS')
    data_info = get_stock_fundamentals('ITC.NS')
    data, train, test = train_test_split(data)
    accuracy, forecast = forecasting(data, train, test)

    print(f'Stock : ITC.NS \nStock Fundamentals:\n{data_info}\nHistorical Accuracy: {accuracy}\nForecast:\n{forecast}')


    