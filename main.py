import bt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
plt.style.use('ggplot')
import streamlit as st

import requests
import os
import sys
import subprocess


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
    select = st.sidebar.selectbox("Menu", ['About', 'EMA Crossover', 'MACD Crossover'])

    if select == 'EMA Crossover':
        ema_interface()

    if select == 'MACD Crossover':
        macd_interface()
    if select == 'About':
        about()


    # if st.button("Generate"):
    #     try:
    #         if strategy == 'EMA Crossover':
    #             if short_window > 0 and long_window > 0:
    #                 df, bt_test = ema_crossover(ticker, short_period=short_window, long_period=long_window, start=start,
    #                                             end = end)
    #                 bt_results = bt.run(bt_test)
    #                 bt_results.prices.columns = ['Equity Progression']
    #                 st.markdown(" #### Equity Progression for " + ticker + " between " + str(start) + " and " +
    #                             str(end),
    #                             unsafe_allow_html=True)
    #                 st.line_chart(bt_results.prices)
    #
    #                 st.markdown("#### Buy/Sell Signals for " + ticker)
    #
    #                 fig, ax = plt.subplots()
    #                 fig.set_figheight(12)
    #                 fig.set_figwidth(20)
    #                 ax.plot(df[['Price', 'EMA_short', 'EMA_long']])
    #                 ax.legend(['Price', 'EMA_short', 'EMA_long'])
    #                 ax.set_title("Crossover Signals for " + ticker)
    #                 ax.set_xlabel("Date")
    #                 ax.set_ylabel("Price")
    #
    #                 ax2 = ax.twinx()
    #                 ax2.plot(df['Signal'], color='navy')
    #                 ax2.set_ylabel("Signal")
    #                 ax.legend(['Price', 'EMA_short', 'EMA_long'], loc='upper left')
    #                 ax2.legend(['Buy/Sell Signal'], loc='lower right')
    #
    #                 st.pyplot(fig)
    #
    #             else:
    #                 st.error("Please ensure you've chosen EMA periods greater than 2. Ideal values for short windows " +
    #                          "are 5, 10, 15 etc. Ideal values for long windows are 40, 50, 60 and so on.")
    #
    #         if strategy == 'MACD':
    #             df = get_stock_data(ticker, start, end)
    #
    #             macd, signal, hist = talib.MACD(df.Close)
    #             df['MACD'] = macd
    #             df['signal'] = signal
    #
    #             a = MACD(df)
    #             df['Buy_Signal_Price'] = a[1]
    #             df['Sell_Signal_Price'] = a[0]
    #
    #             st.line_chart(df['Adj Close'])
    #
    #             # MACD strategy visualization:
    #
    #
    #             st.write("MACD Crossover Lines with 9 EMA for " + ticker)
    #
    #             st.line_chart(df[['MACD', 'signal']])
                #
                # fig1, ax = plt.subplots()
                #
                # fig1.set_figheight(6)
                # fig1.set_figwidth(12)
                # ax.plot(df['MACD'], label='MACD', color='red')
                # ax.plot(df['signal'], label='Signal', color='navy')
                # ax.set_xlabel("Date")
                # ax.set_title("MACD with 9 EMA Signal Line for " + ticker)
                # ax.legend()
                #
                # st.pyplot(fig1)





                # Buy/Sell signals here
                #
                # st.write("Buy/Sell signals for " + ticker)
                #
                # fig2, ax = plt.subplots()
                # fig2.set_figheight(6)
                # fig2.set_figwidth(12)
                #
                # ax.scatter(df.index, df.Sell_Signal_Price, color='green', marker='^', label='Buy')
                # ax.scatter(df.index, df.Buy_Signal_Price, color='red', marker='v', label='Sell')
                # ax.plot(df['Close'], label='Close Price ($)', alpha=.35)
                # ax.set_title("MACD Buy/Sell Strategy for " + ticker)
                # ax.set_xlabel("Date")
                # ax.set_ylabel("Price ($)")
                # ax.legend()
                #
                # st.pyplot(fig2)




        #     else:
        #         st.error("Please ensure you've selected a valid date range for a valid ticker.")
        # except ValueError:
        #     st.error("Please Ensure All Entries Are Filled Correctly.")




def ema_interface():
    st.title("EMA Crossover Strategy.")

    st.markdown("#### Type in a ticker, and select the date range for which you would like to perform this test."
                " Suggested EMA windows are 5/20, 10/40, 20/70 etc.", unsafe_allow_html=True)
    ticker = st.text_input("Ticker")
    ticker.capitalize()

    start = st.date_input("Start Date", value=(datetime.today() - timedelta(5 * 365)))
    end = st.date_input("End Date")

    short_window = st.number_input("Short EMA (Only Applies for EMA Crossover Strategy)")
    long_window = st.number_input("Long EMA (Only Applies for EMA Crossover Strategy")

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
                " Suggested EMA values for this test are smaller value, such as 5 or 9 .(We used 9)")
    ticker = st.text_input("Ticker")
    ticker.capitalize()

    start = st.date_input("Start Date", value=(datetime.today() - timedelta(5 * 365)))
    end = st.date_input("End Date")

    ema_window = st.number_input("Choose an EMA Window:")
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


if __name__ == "__main__":
    main()

