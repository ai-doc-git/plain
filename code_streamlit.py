#####     PLAIN - A Stock Market analysis tool for beginners     #####
# This is just for Educational purposes


# Import python packages
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib import style
plt.style.use('classic')

import time
from datetime import date, timedelta

import streamlit as st # for web interface and deployment

from PIL import Image
import yfinance as fin # for getting stock information

from autots import AutoTS # for forecasting

from sklearn.metrics import mean_absolute_percentage_error # for evaluation
from scipy.stats import kendalltau # for checking the trend



################################# Function Definitions #################################

def get_data(stock_symbol, data_period='5y', data_interval='1d'):
    """
    Function to get daily level data for the given stock ticker
    """
    stock = fin.Ticker(stock_symbol)
    data = stock.history(period=data_period, interval=data_interval)

    data = data.reset_index()
    data = data[['Date','Open']]
    data = data.rename(columns={'Date':'ds','Open': 'y'})
    data = data.set_index(['ds'])
    data = data.fillna(data['y'].mean())
    return data

def get_stock_fundamentals(stock_symbol):
    """
    Function to get the details for any given stock
    """
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
    revenueGrowth = stock_info['revenueGrowth']
    lastDividendValue = stock_info['lastDividendValue']
    dividendYield = stock_info['dividendYield']
    
    fiftyDayAverage = stock_info['fiftyDayAverage']
    twoHundredDayAverage = stock_info['twoHundredDayAverage']
    sharesOutstanding = stock_info['sharesOutstanding']

    recommendationKey = stock_info['recommendationKey']
    numberOfAnalystOpinions = stock_info['numberOfAnalystOpinions']

    return industry, sector, longName, fte, total_revenue, total_debt, currentPrice, fiveYearAvgDividendYield, revenueGrowth, dividendYield, \
            lastDividendValue, fiftyDayAverage, twoHundredDayAverage, recommendationKey, numberOfAnalystOpinions, sharesOutstanding


def train_test_split(data):
    """
    Function to split the data into train and test
    """
    train = data[:-250]
    test = data[-250:]
    return data, train, test


def forecasting(data, train, test):
    """
    Function to forecast 3 months ahead for any given stock 
    """
    model = AutoTS(
    forecast_length=len(test),
    frequency='infer',
    ensemble=None,
    max_generations=0,
    num_validations=0,
    model_list='motifs',
    )
    
    model = model.fit(train.y)
    model.export_template('model.csv', models='best', max_per_model_class=1)
    predictions = model.predict()

    mape = mean_absolute_percentage_error(
                y_true = test,
                y_pred = predictions.forecast
            )
    
    accuracy = 100 - (mape * 100)

    model = AutoTS(
    forecast_length=90,
    frequency='infer',
    ensemble=None,
    max_generations=0,
    num_validations=0,
    model_list='motifs',
    )
    model = model.import_template('model.csv', method='only')
    model = model.fit(data.y)
    final_predictions = model.predict()

    return accuracy, final_predictions.forecast

def calculate_kendall_tau(data):
    """
    Function to check the trend of any given stock in last year
    """
    index_list = [item for item in range(1,len(data)+1)]
    data['Index'] = index_list
    data = data[-270:]

    corr, _ = kendalltau(data['Index'], data['y'])

    if corr > 0.6:
        return "Strong Positive Trend"
    elif corr > 0 and corr <= 0.6:
        return "Weak Positive Trend"
    elif corr < 0 and corr >= -0.6:
        return "Weak Negative Trend"
    else:
        return "Strong Negative Trend"

def market_cap_finder(sharesOutstanding, currentPrice):
    """
    Function to find the market cap category of a company
    """
    market_cap = (sharesOutstanding * currentPrice)/10000000
    if market_cap >=20000:
        return "Large Cap"
    elif market_cap < 20000 and market_cap >= 5000:
        return "Mid Cap"
    else:
        return "Small Cap"

def bullish_or_bearish(currentPrice, twoHundredDayMovingAvg):
    """
    Function to check for the current trend of the stock
    """
    if currentPrice > twoHundredDayMovingAvg:
        return 'BULLISH TREND'
    else:
        return 'BEARISH TREND'

def stock_price_growth(stock_symbol, data_period='5y', data_interval='1d'):
    """
    Function to calculate the growth of a stock price over 1y, 3y and 5y
    """
    stock = fin.Ticker(stock_symbol)
    data = stock.history(period=data_period, interval=data_interval)

    data = data.reset_index()
    data = data[['Date','Open']]
    data = data.rename(columns={'Date':'ds','Open': 'y'})
    data = data.set_index(['ds'])
    data = data.fillna(data['y'].mean())

    if data_period == '5y':
        period = 5
    elif data_period == '3y':
        period = 3
    else:
        period = 1

    end_value = data[-1:].values[0][0]
    start_value = data[:1].values[0][0]
    stock_price_growth = ((end_value - start_value)/start_value) * 100
    return stock_price_growth

def final_verdict(market_cap):
    """
    Function to recommend a stock based on it's fundamental and technical aspects
    v1 - It will be enhanced in the future with better KPIs
    """
    if market_cap in ['Large Cap', 'Mid Cap']:
        return 'Long Term'
    else:
        return 'Short Term'

########################################################################################


################################# Streamlit setup for web interface and Function calls #################################

st.set_page_config(page_title='PLAIN', page_icon='🤑',layout="wide")

m = st.markdown("""
<style>
div.stButton > button:first-child {
    background-color: rgb(236, 247, 237);
    color: rgb(60, 116, 64);
    height: 3em;
    width: 31.5em; 
}   
</style>""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

col2.markdown("<h1 style='text-align: center; color: rgb(60, 116, 64);'> PLAIN </h1>", unsafe_allow_html=True)
col2.markdown("<p style='text-align: center; color: rgb(60, 116, 64);'>Stock Market Analysis Tool for Beginners</p>", unsafe_allow_html=True)
title = col2.text_input('Enter the stock symbol', 'TCS.NS',)

if col2.button('Analyze'):

    data = get_data(title)
    industry, sector, longName, fte, total_revenue, total_debt, currentPrice, fiveYearAvgDividendYield, revenueGrowth, dividendYield, \
        lastDividendValue, fiftyDayAverage, twoHundredDayAverage, recommendationKey, numberOfAnalystOpinions, \
        sharesOutstanding = get_stock_fundamentals(title)

    data, train, test = train_test_split(data)
    accuracy, forecast = forecasting(data, train, test)

    print(forecast)
    data[-150:].y.plot(label='actual', figsize=(7,5), color='black')
    forecast.y.plot(label='forecast', figsize=(7,5), color='green')
    plt.legend(loc='best')
    plt.grid()
    plt.savefig('forecast.jpg')
    
    image = Image.open('forecast.jpg')
    col2.image(image)

    col1.markdown("<p style='text-align: center; color: rgb(0, 0, 0);'> Historical Forecast </p>", unsafe_allow_html=True)

    col1.success(f"Historical Trend (1 year): {calculate_kendall_tau(data)}")
    col1.success(f"Forecast horizon: 3 months")
    col1.success(f"Historical forecast accuracy: {round(accuracy,2)} %")

    col3.markdown("<p style='text-align: center; color: rgb(0, 0, 0);'> Stock Fundamentals </p>", unsafe_allow_html=True)

    col3.success(f"Stock Name: {longName}")
    col3.success(f"Current Price: Rs. {currentPrice}")
    col3.success(f"Market Cap: {market_cap_finder(sharesOutstanding, currentPrice)}")

    # col3.markdown("<p style='text-align: center; color: rgb(0, 0, 0);'> Technical Information </p>", unsafe_allow_html=True)

    col3.success(f"Industry: {industry}")
    col3.success(f"Sector: {sector}")
    # col3.success(f"Full-time Employee: {fte}")

    col3.success(f"50 day average: Rs. {fiftyDayAverage}")
    col3.success(f"200 day average: Rs. {twoHundredDayAverage}")

    col1.markdown("<p style='text-align: center; color: rgb(0, 0, 0);'> Technical Information </p>", unsafe_allow_html=True)
    
    col3.success(f"Current Stock Trend: {bullish_or_bearish(currentPrice, twoHundredDayAverage)}")
    col1.success(f"Revenue: Rs. {round(total_revenue/10000000, 2)} Cr.")
    col1.success(f"Debt: Rs. {round(total_debt/10000000, 2)} Cr.")
    col1.success(f"Revenue Growth (YoY): {round(revenueGrowth*100,1)} %")

    col1.success(f"Stock price growth (1y): {round(stock_price_growth(title, data_period='1y', data_interval='1d'), 2)} %")
    col1.success(f"Stock price growth (3y): {round(stock_price_growth(title, data_period='3y', data_interval='1d'), 2)} %")
    col1.success(f"Stock price growth (5y): {round(stock_price_growth(title, data_period='5y', data_interval='1d'), 2)} %")

    col3.markdown("<p style='text-align: center; color: rgb(0, 0, 0);'> Dividend Information </p>", unsafe_allow_html=True)
    
    col3.success(f"Last Dividend Value: Rs. {lastDividendValue}")
    col3.success(f"Dividend Yield: {round(dividendYield*100,2)}")
    col3.success(f"5 year avg dividend: {fiveYearAvgDividendYield} %")

    col2.markdown("<p style='text-align: center; color: rgb(0, 0, 0);'> -------------------- </p>", unsafe_allow_html=True)
    col2.markdown("<p style='text-align: center; color: rgb(0, 0, 0);'> Final Recommendation </p>", unsafe_allow_html=True)
    col2.markdown("<p style='text-align: center; color: rgb(0, 0, 0);'>  </p>", unsafe_allow_html=True)
    
    col1.success(f"Analyst recommendation: {recommendationKey}")
    col1.success(f"No. of Analyst: {numberOfAnalystOpinions}")
    col2.success(f"Invest for : {final_verdict(market_cap_finder(sharesOutstanding, currentPrice))}")


################################# The END? #################################