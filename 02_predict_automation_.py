import pandas as pd
import yfinance as yf
import datetime
import pickle
import time
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt


# print(len(world_indices))  # 30
# print(len(futures))  # 31
# print(len(currencies_lists))  # 23 총 84개

# 아직 vix랑 currencies는 추가 안됨
asset_class = [('^AORD', 'ALL ORDINARIES'), ('^BFX', 'BEL 20'), ('^FCHI', 'CAC 40'), ('^BUK100P', 'Cboe UK 100'), ('^GDAXI','DAX PERFORMANCE-INDEX'),
             ('^DJI','Dow Jones Industrial Average'),('^STOXX50E', 'ESTX 50 PR.EUR'),('^N100', 'Euronext 100 Index'),('^KLSE','FTSE Bursa Malaysia KLCI'),
             ('^FTSE', 'FTSE 100'),('^HSI','HANG SENG INDEX'),('^BVSP','IBOVESPA'), ('IMOEX.ME','MOEX Russia Index'),('^MXX','IPC MEXICO'),
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


# 종목 입력시 , 종목과 이름으로 묶인 튜플을 가져와서 진행시키게 함
# 종가만 가져와서 예측케 하기.

# ticker_list = input('예측을 원하는 종목을 모두 입력하시오.')
ticker_list = ['KE=F','^AORD',  'SI=F']
a = input('예측을 원하는 날짜을 입력하시오 YYYY-MM-DD (오늘/어저께/그저께) ')


selected_assets = []
for asset in asset_class:
    if asset[0] in ticker_list:
        selected_asset = asset
        selected_assets.append(selected_asset)
print(selected_assets)   # 튜플로 가져오기 [('^IXIC', 'NASDAQ Composite'), ('SI=F', 'SILVER'), ('KE=F', 'WHEAT')]


for ticker, name in selected_assets:
    # Nan값으로 채워진 데이터프레임 깡통 만들기.
    # 인덱스의 이름이 '예측을 실행하는 날짜' 이고 인덱스는 '예측치' '실제치' '잔차' '예측할 날짜' 네 개임
    # 인덱스가 20개가 되면 자동으로 플롯이 만들어지고 (잔차에 대한 절댓값)
    df_columns = ['예측을 실행하는 날짜', '예측치', '실제치', '잔차', '예측할 날짜']
    df = pd.DataFrame(columns=df_columns)
    df = df.set_index('예측을 실행하는 날짜')
    print(df)

    # updated 데이터 먼저 가져오기(맨처음에 마지막60개 저장한 그 csv)
    updated_Close = pd.read_csv('./close_price/updated/{}_{}_updated.csv'.format(name, 'Adj Close'))
    updated_Close['Date'] = pd.to_datetime(updated_Close['Date'])
    updated_Close.set_index('Date', inplace=True)

    # 새로운 데이터 불러오고 전처리
    Today = datetime.date.today()
    last_date_from_previous_df = pd.to_datetime(updated_Close.index[-1]).date()  # 01.25일
    one_day = datetime.timedelta(days=1)
    new_data = yf.download(ticker, start=last_date_from_previous_df - one_day, end=a)  #updated데이터의 마지막 날짜부터, 가장최신 데이터까지
    print(new_data)
    print(type(a))  # str
    new_data.drop(['Open', 'High', 'Low', 'Close', 'Volume'], axis=1, inplace=True)

    # concat
    updated_Close = pd.concat([updated_Close, new_data])
    # 업데이트 된 데이터 저장(다음번 예측에 쓰임)
    # updated_Close.to_csv('./close_price/updated/{}_Adj Close_updated.csv'.format(name), index=True)
    print(updated_Close.head())

    # 마지막 30개 예측
    with open('./close_price/minmaxscaler/{}_Adj Close_minmaxscaler.pickle'.format(name), 'rb') as f:
        minmaxscaler = pickle.load(f)
    last30_df = updated_Close[-30:]
    scaled_last30_df = minmaxscaler.transform(last30_df)
    model = load_model('./close_price/models/{}_Adj Close_model.h5'.format(name))
    tmr_predict = model.predict(scaled_last30_df.reshape(1, 30, 1))
    tmr_predicted_value = minmaxscaler.inverse_transform(tmr_predict)
    # print('내일예측값$ %2f '%tmr_predicted_value[0][0])
    print('종가의 내일 예측값_{}'.format(tmr_predicted_value))

    # df '예측치' 추가
    # last_close_price_date = new_data.index[-1]# 마지막 종가의 날짜.
    # last_close_price_date = str(last_close_price_date).split()[0]
    # print(last_close_price_date)
    # df.loc[last_close_price_date] = [tmr_predicted_value, np.nan, np.nan, np.nan]
    # print(df)


    Today = str(Today)
    df.loc[Today] = [tmr_predicted_value, np.nan, np.nan, np.nan]
    # if df.shape[0] > 1:
    #     previous_row = df.shape[0]-2    # 현재 예측치 까지 추가한 데이터프레임에서 현재 행 바로 이전행의 넘버값.
    #     # df 예측할 날짜 추가
    #     df.iloc[previous_row][3] = df.index[previous_row]
    #     # df 실제치 추가
    #     df.iloc[previous_row][1] = new_data.iloc[-2][0]
    #     print(df.iloc[previous_row][1])
    #     # df 잔차추가  => 실제치에 값이 들어간 순간 잔차를 구해서 추가한다.
    #     if df['실제치'] != np.nan:
    #         df.iloc[previous_row][2] = df.iloc[previous_row][1] - df.iloc[previous_row][0]
    #     if df.shape[0] >= 20:
    #         plt.plot(df[['잔차']])
    #         # 플롯저장
    #     # if new_data[-1]
    #     print(df)

    print(df)
    # 새로 만들어진 데이터 프레임 저장
    df.to_csv('./close_price/predict_df/{}_predict_actual_df.csv', index=True)


# 인2덱스의 이름이 '예측하는 날짜' 이고 인덱스는 '예측치' '실제치' '잔차' 세개 임
# 인덱스가 20개가 되면 자동으로 플롯이 만들어지고 (잔차에 대한 절댓값)

data = { '종목': [name],
         'High' : ['high'] }
predict_4_df = pd.DataFrame(data)
predict_4_df.set_index('종목', inplace=True)
predict_4_df.columns.name = '예측치'

# 각각의 column을 예측해주는 과정 및 모든 예측결과를 데이터프레임으로 save_predict_tmr 폴더에 저장해준다,


