import bt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
plt.style.use('ggplot')
import streamlit as st
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBRegressor


import requests
import os
import sys
import subprocess

# import talib


if not os.path.isdir("/tmp/ta-lib"):

    # Download ta-lib to disk
    with open("/tmp/ta-lib-0.4.0-src.tar.gz", "wb") as file:
        response = requests.get(
            "http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz"
        )
        file.write(response.content)
    # get our current dir, to configure it back again. Just house keeping
    default_cwd = os.getcwd()
    os.chdir("/tmp")
    # untar
    os.system("tar -zxvf ta-lib-0.4.0-src.tar.gz")
    os.chdir("/tmp/ta-lib")
    os.system("ls -la /app/equity/")
    # build
    os.system("./configure --prefix=/home/appuser")
    os.system("make")
    # install
    os.system("make install")
    # back to the cwd
    os.chdir(default_cwd)
    sys.stdout.flush()

# add the library to our current environment
from ctypes import *

lib = CDLL("/home/appuser/lib/libta_lib.so.0.0.0")
# import library
try:
    import talib
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--global-option=build_ext", "--global-option=-L/home/appuser/lib/", "--global-option=-I/home/appuser/include/", "ta-lib"])
finally:
    import talib




def main():


    st.title("Simple Stock Backtesting")
    st.sidebar.markdown("# Simple Stock Backtesting for Any Asset", unsafe_allow_html=True)
    select = st.sidebar.selectbox("Menu", ['About', 'EMA Crossover', 'MACD Crossover', 'On-Balance-Volume Crossover',
                                           'Make A Prediction'])

    if select == 'EMA Crossover':
        ema_interface()

    if select == 'MACD Crossover':
        macd_interface()
    if select == 'About':
        about()
    if select == 'Make A Prediction':
        make_pred()
    if select == 'On-Balance-Volume Crossover':
        obv()





def ema_interface():
    st.title("EMA Crossover Strategy.")

    st.markdown("#### Type in a ticker, and select the date range for which you would like to perform this test."
                " Suggested EMA windows are 5/20, 10/40, 20/70 etc.", unsafe_allow_html=True)
    st.write("\n")
    st.markdown("This strategy is defined by utilizing two separate EMA Windows, short and long. When the short window "
                "cross the long window going up, this is a 'buy' signal. When the short window cross the long window "
                "going "
                "down, "
                " this is a 'Sell' signal.", unsafe_allow_html=True)

    st.write("\n")

    st.markdown("**REMEMBER**: Crypto tickers use '-USD' notation (Ex. BTC-USD, DOGE-USD, ETH-USD etc.)")

    ticker = st.text_input("Ticker")
    ticker.capitalize()

    start = st.date_input("Start Date", value=(datetime.today() - timedelta(5 * 365)))
    end = st.date_input("End Date")

    short_window = st.number_input("Short EMA (Only Applies for EMA Crossover Strategy)", value=10)
    long_window = st.number_input("Long EMA (Only Applies for EMA Crossover Strategy", value=40)

    if st.button("Generate"):
        try:
            if short_window > 1 and long_window > 1:
                    df, bt_test = ema_crossover(ticker, short_period=short_window, long_period=long_window, start=start,
                                            end = end)

                    bt_results = bt.run(bt_test)
                    bt_results.prices.columns = ['Equity Progression']
                    st.markdown(" #### Equity Progression for " + ticker + " between " + str(start) + " and " +
                                str(end),
                                unsafe_allow_html=True)
                    st.line_chart(bt_results.prices)

                    st.markdown("#### Buy/Sell Signals for " + ticker)

                    fig, ax = plt.subplots()
                    fig.set_figheight(12)
                    fig.set_figwidth(20)
                    ax.plot(df[['Price', 'EMA_short', 'EMA_long']])
                    ax.legend(['Price', 'EMA_short', 'EMA_long'])
                    ax.set_title("Crossover Signals for " + ticker)
                    ax.set_xlabel("Date")
                    ax.set_ylabel("Price")

                    ax2 = ax.twinx()
                    ax2.plot(df['Signal'], color='navy')
                    ax2.set_ylabel("Signal")
                    ax.legend(['Price', 'EMA_short', 'EMA_long'], loc='upper left')
                    ax2.legend(['Buy/Sell Signal'], loc='lower right')

                    st.pyplot(fig)

            else:
                st.error("Please ensure you've chosen EMA periods greater than 2. Ideal values for short windows " +
                         "are 5, 10, 15 etc. Ideal values for long windows are 40, 50, 60 and so on.")

        except:
            st.error("Please Ensure All Entries Are Filled Correctly, and that you've selected valid date ranges.")

def macd_interface():

    st.title("MACD/EMA Crossover Strategy.")

    st.markdown("#### Type in a ticker, and select the date range for which you would like to perform this test."
                " Suggested EMA values for this test are smaller value, such as 5 or 9.")

    st.write("\n")

    st.markdown("This strategy is defined by observing the behavior of the MACD line in relation to a short EMA line. "
                "When the MACD line crosses the EMA line going up, this is a 'Buy' signal. When the MACD line crosses "
                "the "
                "EMA line going down, this is a 'Sell' signal.")

    st.write("\n")

    st.markdown("**REMEMBER**: Crypto tickers use '-USD' notation (Ex. BTC-USD, DOGE-USD, ETH-USD etc.)")


    ticker = st.text_input("Ticker")
    ticker.capitalize()

    start = st.date_input("Start Date", value=(datetime.today() - timedelta(5 * 365)))
    end = st.date_input("End Date")

    ema_window = st.number_input("Choose an EMA Window:", value=9)
    try:
        if st.button("Generate"):
            if ticker == '':
                st.error("Did you type in a ticker?")
            if ema_window > 1:
                df = get_stock_data(ticker, start, end)

                macd, signal, hist = talib.MACD(df.Close)
                df['MACD'] = macd
                df['signal'] = signal

                a = MACD(df)
                df['Buy_Signal_Price'] = a[1]
                df['Sell_Signal_Price'] = a[0]

                st.line_chart(df['Adj Close'])

                # MACD strategy visualization:


                st.write("MACD Crossover Lines with " + str(ema_window) + " EMA for " + ticker)


                st.line_chart(df[['MACD', 'signal']])

                # Buy/Sell signals here

                st.write("Buy/Sell signals for " + ticker)

                fig2, ax = plt.subplots()
                fig2.set_figheight(6)
                fig2.set_figwidth(12)

                ax.scatter(df.index, df.Sell_Signal_Price, color='green', marker='^', label='Buy')
                ax.scatter(df.index, df.Buy_Signal_Price, color='red', marker='v', label='Sell')
                ax.plot(df['Close'], label='Close Price ($)', alpha=.35)
                ax.set_title("MACD Buy/Sell Strategy for " + ticker)
                ax.set_xlabel("Date")
                ax.set_ylabel("Price ($)")
                ax.legend()

                st.pyplot(fig2)
            else:
                st.error("Please ensure you've selected a valid date range and an EMA Window greater than 1")
    except ValueError:
        st.error("Please ensure you've chosen a valid date range and an EMA window greater than 1.")




def ema_crossover(ticker, short_period=10, long_period=40, name='EMA_Crossover', start='2020-05-01', end=datetime.today()):
    # Get price data
    price_data = get_close(ticker, start=start, end=end)
    price_data.columns = ['Close']

    st.write("Daily Chart for " + ticker)
    st.line_chart(price_data)

    # Calculate short term EMA and long term EMA
    EMA_short = talib.EMA(price_data['Close'],
                                timeperiod=short_period).to_frame()
    EMA_long = talib.EMA(price_data['Close'],
                                timeperiod=long_period).to_frame()

    # Create a copy of EMA_long to construct a signal dataframe
    signal = EMA_long.copy()
    signal[EMA_long.isnull()] = 0 # Set signal to 0 for the points that do not have data

    signal[EMA_short > EMA_long] = 1
    signal[EMA_short < EMA_long] = -1
    signal.columns=['Close']

    combined_df = bt.merge(signal, price_data, EMA_short, EMA_long)
    combined_df.columns = ['Signal', 'Price', 'EMA_short', 'EMA_long']

    stock_close = pd.DataFrame(price_data['Close'])

    bt_strategy = bt.Strategy('EMA_crossover',
                            [
                                bt.algos.WeighTarget(signal),
                                bt.algos.Rebalance()
                            ])

    return combined_df, bt.Backtest(bt_strategy, stock_close)



def get_stock_data(ticker, start, end):
    stock = yf.download(ticker, start=start, end=end)
    # stock = pd.DataFrame(stock['Adj Close'])
    return stock

def get_close(ticker, start, end):
    stock = yf.download(ticker, start=start, end=end)
    stock = pd.DataFrame(stock['Adj Close'])
    return stock


def MACD(signal):
    Buy = []
    Sell = []
    flag = -1

    for i in range(0, len(signal)):
        if signal['MACD'][i] > signal['signal'][i]:
            Sell.append(np.nan)
            if flag != 1:
                Buy.append(signal.Close[i])
                flag = 1
            else:
                Buy.append(np.nan)
        elif signal['MACD'][i] < signal['signal'][i]:
            Buy.append(np.nan)
            if flag != 0:
                Sell.append(signal.Close[i])
                flag = 0
            else:
                Sell.append(np.nan)

        else:
            Buy.append(np.nan)
            Sell.append(np.nan)

    return (Buy, Sell)



def about():

    st.image('https://images.unsplash.com/photo-1612696874005-d015469bc660?ixid=MnwxMjA3fDB8'
             'MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&ixlib=rb-1.2.1&auto=format&fit=crop&w=641&q=80', width=300)

    st.markdown("#### App by Adrien Clay", unsafe_allow_html=True)
    st.markdown("I wrote this app with the intention of simplifying the interface with which I do my own backtesting. \n"
                "This back testing was previously all done out of Jupyter Notebooks, and seemed like it would make a decent"
                " app for non-serious stock backtesting that gives the user an easy-to-use interface, rather than being housed"
                " exclusively in a script.")
    st.markdown("Thanks for checking it out!")


    st.write()
    st.write()
    st.markdown("**Disclaimer: None of the information taken from this app should be considered financial advise.**")


def make_pred():
    st.title("Make a Weak Prediction using Machine Learning")

    st.markdown("#### Type in a ticker, and select the date range for which you would like to perform this test."
                " The larger the date range, the better this machine learning algorithm is likely to perform. "
                "I recommend only doing between 5 and 30 day predictions.")

    st.write("\n")

    st.markdown("**REMEMBER**: Crypto tickers use '-USD' notation (Ex. BTC-USD, DOGE-USD, ETH-USD etc.)")
    ticker = st.text_input("Ticker")
    ticker.capitalize()

    start = st.date_input("Start Date", value=(datetime.today() - timedelta(4 * 365)))
    end = st.date_input("End Date")

    future_days = st.number_input("How many days in the future do you want to look? (We recommend 30 or less.."
                                  " Things get really wonky after that.)", value=5)

    if st.button("Generate"):
        generate_pred(ticker, start, end, int(future_days))


def generate_pred(ticker, start, end, future_days):
    df = yf.download(ticker, start=start, end=end, interval='1d')
    df.dropna(inplace=True)
    df[str(future_days) + '_Day_Price_Forecast'] = df[['Close']].shift(-future_days)
    X = np.array(df[['Close']])
    X = X[:df.shape[0] - future_days]
    y = np.array(df[str(future_days) + '_Day_Price_Forecast'])
    y = y[:-future_days]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    xgbr = XGBRegressor()
    xgbr.fit(X_train, y_train)

    confidence = xgbr.score(X_test, y_test)
    xgb_preds = xgbr.predict(X_test)
    future_preds = xgbr.predict(X[-future_days:])
    future_preds = pd.DataFrame(future_preds, index=range(len(df), len(df) + future_days))
    df.reset_index(inplace=True)

    st.write("Actual Price Movement vs. Predicted Price Movement for " + ticker)

    fig, ax = plt.subplots()
    fig.set_figheight(6)
    fig.set_figwidth(12)
    ax.plot(xgb_preds, color='navy', label='Prediction')
    ax.plot(y_test, color='darkred', label='Actual')
    ax.set_xlabel("Timeframe (Days)")
    ax.set_ylabel("Closing Price ($)")

    ax.set_title("Actual Prices vs. Prediction for " + ticker + " using XGBoost Regressor")
    ax.legend()

    st.pyplot(fig)

    st.markdown("#### " + str(future_days) + " day prediction for " + ticker)

    fig, ax = plt.subplots()
    fig.set_figheight(6)
    fig.set_figwidth(12)
    ax.plot(df.Close.iloc[-200:], label='Closing Price')
    ax.plot(future_preds, label=str(future_days) + ' Day Prediction')
    ax.scatter(x=df.index[-1], y=df.Close.iloc[-1], color='green', s=70, label='Last Price: ' + str(round(df.Close.iloc[-1], 4)))

    ax.scatter(x=future_preds.sort_values(0).index[0], y=future_preds.sort_values(0).iloc[0],
               label='Lowest Price: ' + str(round(future_preds.sort_values(0).iloc[0][0], 4)), color='red', s=70)
    ax.set_xlabel("Timeframe (Days)")
    ax.set_ylabel("Closing Price ($)")
    ax.legend()
    ax.set_title(str(future_days) + " Day Prediction for " + ticker + " ----- Confidence: " + str(
        round(confidence, 3) * 100) + "%")

    st.pyplot(fig)


def obv():
    st.title("On Balance Volume Crossover")

    st.markdown("#### Type in a ticker, and select the date range for which you would like to perform this test."
                " Suggested EMA windows are 5/20, 10/40, 20/70 etc.", unsafe_allow_html=True)
    st.write("\n")
    st.markdown("This strategy is defined by utilizing the 'On Balance Volume' for the chosen asset. OBV is a momentum"
                " indicator that uses volume flow to predict changes in stock price. The strategy entails calculating "
                "the OBV, and subsequently the exponential moving average of the OBV. When the OBV crosses above it's "
                "EMA, this is a buy singal. When it crosses below, this is a sell signal. We used an EMA window of "
                "20.")

    st.write("\n")

    st.markdown("**REMEMBER**: Crypto tickers use '-USD' notation (Ex. BTC-USD, DOGE-USD, ETH-USD etc.)")

    ticker = st.text_input("Ticker")
    ticker.capitalize()

    start = st.date_input("Start Date", value=(datetime.today() - timedelta(5 * 365)))
    end = st.date_input("End Date")

    EMA_LENGTH = st.number_input("Short EMA (Only Applies for EMA Crossover Strategy)", value=20)

    if st.button("Generate"):
        if EMA_LENGTH > 1 and start < end:
            df = get_obv(ticker, start, end, EMA_LENGTH)

            st.markdown("#### Closing Price for " + ticker)
            st.line_chart(df['Adj Close'])

            st.markdown("### OBV and OBV EMA for " + ticker + " with " + str(EMA_LENGTH) + " day EMA window.")

            st.line_chart(df[['OBV', 'OBV_EMA']])


            st.markdown("### Buy/Sell signals for " + ticker + " based on OBV Crossover:")

            fig, ax = plt.subplots()

            fig.set_figheight(6)
            fig.set_figwidth(16)
            ax.scatter(df.index, df['Buy_Signal_Price'], color='green', label='Buy Signal', marker='^', alpha=.7)
            ax.scatter(df.index, df['Sell_Signal_Price'], color='red', label='Sell Signal', marker='v', alpha=.7)
            ax.plot(df['Close'], label='Close Price', alpha=0.35)
            ax.set_title("Buy/Sell Signals for " + ticker)
            ax.set_xlabel("Date", fontsize=18)
            ax.set_ylabel("Closing Price ($)", fontsize=18)
            ax.legend()

            st.pyplot(fig)
        else:
            st.error("Please ensure you're using valid start/end dates and that your EMA window is greater than 1.")




def get_obv(ticker, start, end, EMA_Length):
    df = yf.download(ticker, start=start, end=end)

    OBV = []
    OBV.append(0)
    for i in range(1, len(df.Close)):
        if df.Close[i] > df.Close[i - 1]:  # If the closing price is above the prior close price
            OBV.append(OBV[-1] + df.Volume[i])  # then: Current OBV = Previous OBV + Current Volume
        elif df.Close[i] < df.Close[i - 1]:
            OBV.append(OBV[-1] - df.Volume[i])
        else:
            OBV.append(OBV[-1])

    # Store OBV and OBV_EMA
    df['OBV'] = OBV
    df['OBV_EMA'] = df['OBV'].ewm(com=EMA_Length).mean()

    x = obv_buy_sell(df, 'OBV', 'OBV_EMA')
    df['Buy_Signal_Price'] = x[0]
    df['Sell_Signal_Price'] = x[1]

    return df



def obv_buy_sell(signal, col1, col2):
    sigPriceBuy = []
    sigPriceSell = []
    flag = -1  # A flag for the trend upward/downward
    # Loop through the length of the data set
    for i in range(0, len(signal)):
        # if OBV > OBV_EMA  and flag != 1 then buy else sell
        if signal[col1][i] > signal[col2][i] and flag != 1:
            sigPriceBuy.append(signal['Close'][i])
            sigPriceSell.append(np.nan)
            flag = 1
        # else  if OBV < OBV_EMA  and flag != 0 then sell else buy
        elif signal[col1][i] < signal[col2][i] and flag != 0:
            sigPriceSell.append(signal['Close'][i])
            sigPriceBuy.append(np.nan)
            flag = 0
        # else   OBV == OBV_EMA  so append NaN
        else:
            sigPriceBuy.append(np.nan)
            sigPriceSell.append(np.nan)

    return (sigPriceBuy, sigPriceSell)


if __name__ == "__main__":
    main()

