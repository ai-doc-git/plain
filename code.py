import os
import numpy as np
import pandas as pd
import time

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By

from scipy.stats import kendalltau

def set_chrome_webdriver():
    download_path = os.getcwd()
    full_path = download_path + "/Invest_or_Not_Project"
    print(full_path)
    chrome_options = webdriver.ChromeOptions()
    prefs = {'download.default_directory' : full_path}
    chrome_options.add_experimental_option('prefs', prefs)
    driver = webdriver.Chrome(chrome_options=chrome_options)
    return full_path, driver

def get_historical_data(full_path):
    os.chdir(full_path)
    csv_file = [item for item in os.listdir() if "csv" in item]
    df = pd.read_csv(csv_file[0])
    os.remove(csv_file[0])
    return df

def scrape_website(driver, stock_symbol, full_path):
    print("Loading website .....")
    driver.get("https://finance.yahoo.com/")

    time.sleep(1)
    driver.find_element("name","yfin-usr-qry").send_keys(stock_symbol)
    time.sleep(1)
    driver.find_element("id","header-desktop-search-button").send_keys(Keys.ENTER)

    time.sleep(1)
    historic_data_page_url = driver.find_elements(By.XPATH, '//*[@id="quote-nav"]/ul/li[5]/a')[0].get_attribute('href')
    driver.get(historic_data_page_url)

    time.sleep(5)
    driver.find_elements(By.XPATH, '//*[@id="myLightboxContainer"]/section/button[1]')[0].send_keys(Keys.ENTER)

    # Get 5Y data
    driver.find_elements(By.XPATH, '//*[@id="Col1-1-HistoricalDataTable-Proxy"]/section/div[1]/div[1]/div[1]/div/div/div')[0].send_keys(Keys.ENTER)
    driver.find_elements(By.XPATH, '//*[@id="dropdown-menu"]/div/ul[2]/li[3]/button')[0].send_keys(Keys.ENTER)
    driver.find_elements(By.XPATH, '//*[@id="Col1-1-HistoricalDataTable-Proxy"]/section/div[1]/div[1]/button')[0].send_keys(Keys.ENTER)
    time.sleep(1)
    download_page_url = driver.find_elements(By.XPATH, '//*[@id="Col1-1-HistoricalDataTable-Proxy"]/section/div[1]/div[2]/span[2]/a')[0].get_attribute('href')
    driver.get(download_page_url)
    time.sleep(1)

    data_5y = get_historical_data(full_path)

    # Get 1Y data
    driver.find_elements(By.XPATH, '//*[@id="Col1-1-HistoricalDataTable-Proxy"]/section/div[1]/div[1]/div[1]/div/div/div')[0].send_keys(Keys.ENTER)
    driver.find_elements(By.XPATH, '//*[@id="dropdown-menu"]/div/ul[2]/li[2]/button')[0].send_keys(Keys.ENTER)
    driver.find_elements(By.XPATH, '//*[@id="Col1-1-HistoricalDataTable-Proxy"]/section/div[1]/div[1]/button')[0].send_keys(Keys.ENTER)
    download_page_url = driver.find_elements(By.XPATH, '//*[@id="Col1-1-HistoricalDataTable-Proxy"]/section/div[1]/div[2]/span[2]/a')[0].get_attribute('href')
    driver.get(download_page_url)
    time.sleep(1)

    data_1y = get_historical_data(full_path)

    return data_1y, data_5y

def calculate_kendall_tau():
    df_1y = pd.read_csv('data_1y.csv')
    index_list = [item for item in range(1,len(df_1y)+1)]
    df_1y['Index'] = index_list

    df_5y = pd.read_csv('data_5y.csv')
    index_list = [item for item in range(1,len(df_5y)+1)]
    df_5y['Index'] = index_list

    corr_1y, _ = kendalltau(df_1y['Index'], df_1y['Open'])
    corr_5y, _ = kendalltau(df_5y['Index'], df_5y['Open'])

    return corr_1y, corr_5y



if __name__ == "__main__":
    download_path, driver = set_chrome_webdriver()
    # stock_symbol = input("Please enter a stock symbol:")
    data_1y, data_5y = scrape_website(driver, "ITC.NS", download_path)
    data_1y.to_csv("data_1y.csv",index=False)
    data_5y.to_csv("data_5y.csv",index=False)

    calculate_kendall_tau()


    