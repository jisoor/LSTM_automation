import pandas as pd
import yfinance as yf
import datetime
import pickle
import time
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt


# 총 83개 ( 29 + 31  +  23 )

asset_class = [('^AORD', 'ALL ORDINARIES'), ('^BFX', 'BEL 20'), ('^FCHI', 'CAC 40'), ('^BUK100P', 'Cboe UK 100'), ('^GDAXI','DAX PERFORMANCE-INDEX'),
             ('^DJI','Dow Jones Industrial Average'),('^STOXX50E', 'ESTX 50 PR.EUR'),('^N100', 'Euronext 100 Index'),('^KLSE','FTSE Bursa Malaysia KLCI'),
             ('^FTSE', 'FTSE 100'),('^HSI','HANG SENG INDEX'),('^BVSP','IBOVESPA'), ('^MXX','IPC MEXICO'),
             ('^JKSE', 'Jakarta Composite Index'),('^KS11','KOSPI Composite Index'),('^MERV','MERVAL'),('^IXIC','NASDAQ Composite'),
             ('^N225', 'Nikkei 225'),('^XAX','NYSE AMEX COMPOSITE INDEX'),('^NYA','NYSE COMPOSITE (DJ)'),('^RUT','Russell 2000'),('^GSPC','S&P 500'),
             ('^BSESN', 'S&P BSE SENSEX'), ('399001.SZ', 'Shenzhen Component'),('000001.SS', 'SSE Composite Index'),('^STI','STI Index'),
             ('^TA125.TA', 'TA-125'),('^TWII','TSEC weighted index'),('^VIX','Vix'),                                            # 여기까지 world_indicies
('BZ=F','BRENT_OIL' ),('CC=F','COCOA'),('KC=F', 'Coffee'),('HG=F','COPPER'),('ZC=F','CORN'),('CT=F','COTTON'),
            ('CL=F','CRUDE_OIL' ),('YM=F', 'DOW'),('GF=F','FEEDER_CATTLE'),('GC=F','GOLD'),('HE=F','LEAN_HOGS'),('LE=F','LIVE_CATTLE'),
            ('LBS=F','LUMBER'),('NQ=F', 'NASDAQ'),('NG=F','NATURAL_GAS'),('ZO=F','OAT'),('PA=F','PALLADIUM' ),('PL=F','PLATINUM'),('ZR=F','ROUGH_RICE'),
            ('RTY=F', 'RUSSEL2000'),('SI=F','SILVER'),('ZS=F','SOYBEAN'),('ZM=F','SOYBEAN_MEAL'),('ZL=F','SOYBEAN_OIL'),
            ('ES=F', 'SPX'),('SB=F','SUGAR'),('ZT=F','US2YT'),('ZF=F', 'US5YT'),( 'ZN=F', 'US10YT'), ('ZB=F','US30YT' ),('KE=F','WHEAT'),  # 여기까지 futures
('EURUSD=X', 'EUR-USD'), ('JPY=X', 'USD-JPY'), ('GBPUSD=X', 'GBP-USD'), ('AUDUSD=X', 'AUD-USD'), ('NZDUSD=X', 'NZD-USD'),
            ('EURJPY=X', 'EUR-JPY'), ('GBPJPY=X', 'GBP-JPY'), ('EURGBP=X', 'EUR-GBP'), ('EURCAD=X', 'EUR-CAD'), ('EURSEK=X', 'EUR-SEK'),
            ('EURCHF=X', 'EUR-CHF'), ('EURHUF=X', 'EUR-HUF'), ('CNY=X', 'USD-CNY'), ('HKD=X', 'USD-HKD'), ('SGD=X', 'USD-SGD'),
            ('INR=X', 'USD-INR'), ('MXN=X', 'USD-MXN'), ('PHP=X', 'USD-PHP'), ('IDR=X', 'USD-IDR'), ('THB=X', 'USD-THB'),
            ('MYR=X', 'USD-MYR'), ('ZAR=X', 'USD-ZAR'), ('RUB=X', 'USD-RUB')]                                                       # 여기까지 currencies
# Nan값으로 채워진 데이터프레임 깡통 만들기.
# 컬럼은 4개 =>  '예측치', '실제치', '잔차', '예측할 날짜'
# 인덱스는 => '예측을 실행하는 날짜'(즉, 오늘을 의미함)
for ticker, name in asset_class:

    df_columns = ['예측을 실행하는 날짜', '예측치', '실제치', '잔차', '예측할 날짜']
    df = pd.DataFrame(columns=df_columns)
    df = df.set_index('예측을 실행하는 날짜')
    df.columns.name = name
    print(df)
    print(df.columns.name)
    df.to_csv('./close_price/predict_df/{}_predict_actual_df.csv'.format(name), index=True)
