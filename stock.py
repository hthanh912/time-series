from tkinter import Tk, LEFT, RIGHT, BOTTOM, X, Y, BOTH, RAISED, filedialog, Text, END, Label
from tkinter.ttk import Frame, Button, Style, Notebook
import tkinter as tk                     
from PIL import Image, ImageTk
from pandastable import Table, TableModel
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import matplotlib.dates as mpl_dates
import os
from datetime import datetime, timedelta, date, time
import talib as tb
import mplfinance as mpf
from mplfinance.original_flavor import candlestick2_ochl
from ta.trend import ADXIndicator
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 

from sklearn.preprocessing import MinMaxScaler
from sklearn.externals.joblib import dump, load

from keras import backend as K
import tensorflow as tf
from keras.models import Model, Sequential, model_from_json
from keras.layers import Dense, Dropout, LSTM, Input, Activation, concatenate, Reshape, Flatten
from keras import optimizers
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


def load_file():
    global df
    global filename
    global cal_indi_button
    global preprocess_button
    global train_button
    global img_loss

    path = filedialog.askopenfilename()
    if (path!= ""):
        df_indi = pd.DataFrame()
        table_indi = Table(frame21, dataframe=df_indi, showtoolbar=True, showstatusbar=True)
        table_indi.show()
        table_indi.redraw()
        txt_X_train.delete('1.0', END)
        txt_Indi_train.delete('1.0', END)
        txt_Y_train.delete('1.0', END)

        img_load = Image.open('placeholder.png')
        render = ImageTk.PhotoImage(img_load)
        img_loss.configure(image=render)
        img_loss.imgage = render

        df = load_data(path)
        filename = os.path.splitext(os.path.basename(path))[0]
        label.configure(text="Loaded data " + os.path.basename(path)+ " " + str(df.shape) )   
        table_data = Table(frame1, dataframe=df, showtoolbar=True, showstatusbar=True)
        tabControl.select(0)
        table_data.show()
        table_data.redraw()
        cal_indi_button.configure(state = 'enable')
        preprocess_button.configure(state = 'disable')
        train_button.configure(state = 'disable')
        predict_button.configure(state = 'disable')


def load_data(path):
    df = pd.read_csv(path)
    df = df.rename(columns={"Lần cuối": "Close", "Mở": "Open", "Cao": "High", "Thấp": "Low", "KL": "Volume", "% Thay đổi": "Change", "Ngày": "Date"})
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
    df = df.set_index(pd.DatetimeIndex(df['Date']))
    df.index.name = 'Date'
    df = df.drop(['Date'], axis=1)
    df = df.sort_index(axis=0, ascending=True)
    df['High'] = [float(x.replace(',', '')) for x in df['High']]
    df['Low'] = [float(x.replace(',', '')) for x in df['Low']]
    df['Close'] = [float(x.replace(',', '')) for x in df['Close']]
    df['Open'] = [float(x.replace(',', '')) for x in df['Open']]
    df['Volume'] = df['Volume'].replace({'K': '*1e3', 'M': '*1e6', '-':'0'}, regex=True).map(pd.eval).astype(float)
    df['Change'] = [float(x.strip('%'))/100 for x in df['Change']]
    df.index = df.index.format(formatter=lambda x: x.strftime('%Y-%m-%d'))
    return df


def create_data_X(X, time_step):
	data = []
	for i in range(X.shape[1]):
		dataset = np.insert(np.array(X.iloc[:,i]),[0]*(time_step),0)   
		dataX = []
		for i in range(len(dataset)-(time_step)+1):
			a = dataset[i:(i+(time_step))]
			dataX.append(a)
		data.append(dataX)
	data = np.array(data)

	X = []
	for i in range(time_step, data.shape[1]):
		t = []
		for j in range(data.shape[0]):
			t.append(data[j][i])
		X.append(t)
	data_X = np.array(X)
 #return data train & last data
	return data_X[:-1], data_X[-1]

def create_data_Y(Y, time_step):
    Y = np.array([Y[:,0][i + time_step].copy() for i in range(len(Y) - time_step)])
    data_Y = np.expand_dims(Y, -1)
    return data_Y

def cal_indi_func():
    global df
    global df_indi
    global cal_indi_button
    df_indi = cal_indi(df)
    label.configure(text="Calculated 7 indicators")   
    table_indi = Table(frame21, dataframe=df_indi, showtoolbar=True, showstatusbar=True)
    tabControl.select(1)
    table_indi.show()
    table_indi.redraw()
    cal_indi_button.configure(state = 'disable')
    preprocess_button.configure(state = 'enable')


def cal_indi(df):
    df['Moving Average'], df['Upper Band'], df['Lower Band'] = bollinger_bands(df['Close'])
    df['Kijun_sen'], df['Tenkan_sen'], df['Lead 1'], df['Lead 2'] = Ichimoku(df['High'], df['Low'], df['Close'])
    df['RSI'] = RSI(df['Close'])
    df['macd'], df['signal'] = MACD(df['Close'])
    df['SAR'] = SAR(df['High'],df['Low'])
    df['K'], df['D'] = Stochastic(df['High'],df['Low'],df['Close'])
    df['PDM'], df['NDM'], df['ADX'] = ADX(df['High'],df['Low'],df['Close'])
    return df

def preprocessing():
    global df_indi
    global data_X
    global data_Y
    global last_X
    global X_train
    global Indi_train
    global Y_train
    global x_test
    global indi_test
    global data_Indi
    global last_Indi
    global filename
    global preprocess_button

    X = df_indi[['Close','Volume','Open','High','Low','Change']][77:]
    Indi = df_indi[['Moving Average', 'Upper Band', 'Lower Band', 'Kijun_sen', 'Tenkan_sen', 'Lead 1', 'Lead 2', 'RSI', 'macd', 'signal','SAR', 'K', 'D','PDM','NDM','ADX']][77:]
    Y = np.array(df_indi[['Close']][77:])

    time_step = 10
    data_X, last_X = create_data_X(X, time_step)
    data_Indi, last_Indi = create_data_X(Indi, time_step)
    data_Y = create_data_Y(Y, time_step)

    #Split train, test
    n = round(len(data_X)*0.8)
    X_train = data_X[:n]
    Indi_train = data_Indi[:n]
    Y_train = data_Y[:n]

    x_test = data_X[n:]
    y_test = data_Y[n:]
    indi_test = data_Indi[n:]


    #Scale features of train data
    scaler_X11 = MinMaxScaler(feature_range=(-1, 1))   
    X_train[:,-1] = scaler_X11.fit_transform(X_train[:,-1])

    scaler_X01 = MinMaxScaler(feature_range=(0, 1))   
    X_train_shape = X_train[:,0:5].shape
    temp = X_train[:,0:5].reshape(X_train[:,0:5].shape[0],-1)
    temp = scaler_X01.fit_transform(temp)
    temp = temp.reshape(X_train_shape)
    X_train[:,0:5] = temp

    scaler_Indi01 = MinMaxScaler(feature_range=(0, 1))   
    Indi_train_shape = Indi_train[:,:-2].shape
    temp = Indi_train[:,:-2].reshape(Indi_train[:,:-2].shape[0],-1)
    temp = scaler_Indi01.fit_transform(temp)
    temp = temp.reshape(Indi_train_shape)
    Indi_train[:,:-2] = temp

    scaler_Indi11 = MinMaxScaler(feature_range=(-1, 1))   
    Indi_train_shape = Indi_train[:, -2:].shape
    temp = Indi_train[:, -2:].reshape(Indi_train[:, -2:].shape[0],-1)
    temp = scaler_Indi11.fit_transform(temp)
    temp = temp.reshape(Indi_train_shape)
    Indi_train[:, -2:] = temp

    scaler_Y = MinMaxScaler(feature_range=(0, 1))   
    Y_train = scaler_Y.fit_transform(Y_train)

    #Create folder to save model weights and scalers
    try:
        os.makedirs('model/'+filename)
    except OSError:
        print ("Creation of the directory failed")
    else:
        print ("Successfully created the directory")
    #Save scalers
    dump(scaler_X11, 'model/'+ filename+ '/scaler_X11.bin', compress=True)
    dump(scaler_X01, 'model/' + filename+ '/scaler_X01.bin', compress=True)
    dump(scaler_Indi01, 'model/'+ filename+ '/scaler_Indi01.bin', compress=True)
    dump(scaler_Indi11, 'model/' + filename+ '/scaler_Indi11.bin', compress=True)   
    dump(scaler_Y, 'model/' + filename+ '/scaler_Y.bin', compress=True)

    #Scale features of test data
    x_test[:,-1] = scaler_X11.transform(x_test[:,-1])

    x_test_shape = x_test[:,0:5].shape
    temp = x_test[:,0:5].reshape(x_test[:,0:5].shape[0],-1)
    temp = scaler_X01.transform(temp)
    temp = temp.reshape(x_test_shape)
    x_test[:,0:5] = temp

    indi_test_shape = indi_test[:,:-2].shape
    temp = indi_test[:,:-2].reshape(indi_test[:,:-2].shape[0],-1)
    temp = scaler_Indi01.transform(temp)
    temp = temp.reshape(indi_test_shape)
    indi_test[:,:-2] = temp

    indi_test_shape = indi_test[:, -2:].shape
    temp = indi_test[:, -2:].reshape(indi_test[:, -2:].shape[0],-1)
    temp = scaler_Indi11.transform(temp)
    temp = temp.reshape(indi_test_shape)
    indi_test[:, -2:] = temp

    txt_X_train.insert(END,"X_train:\n\n " + "Shape = " +str(X_train.shape) +"\n\n"+ str(X_train))
    txt_Indi_train.insert(END,"Indicator_train:\n\n "+ "Shape = " +str(Indi_train.shape) +"\n\n"+ str(Indi_train))
    txt_Y_train.insert(END,"Y_train:\n\n " + "Shape = " +str(Y_train.shape) +"\n\n"+ str(Y_train))
    
    tabControl.select(2)
    label.configure(text = 'Preprocessed Data')
    preprocess_button.configure(state = 'disable')
    train_button.configure(state = 'enable')


def load_model():
    # load model from json
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    return model
    
def load_model_weights(model, filename):
    # load weights into model
    model.load_weights('model/'+ filename+ '/weights.h5')
    print("loaded model")
    return model

def load_scaler(filename):
    #load scalers
    scaler_X11 = load('model/'+ filename+ '/scaler_X11.bin')
    scaler_X01 = load('model/'+ filename+ '/scaler_X01.bin')
    scaler_Indi01 = load('model/'+ filename+ '/scaler_Indi01.bin')
    scaler_Indi11 = load('model/'+ filename+ '/scaler_Indi11.bin')
    scaler_Y = load('model/'+ filename+ '/scaler_Y.bin')
    return scaler_X11, scaler_X01, scaler_Indi11, scaler_Indi01 , scaler_Y

def train_model():
    global loaded_model
    global filename
    global img_loss
    global tabControl
    global train_button
    global X_train
    global Indi_train
    global Y_train

    loaded_model = load_model()

    if os.path.isfile('model/'+ filename+ '/weights.h5'):
        loaded_model = load_model_weights(loaded_model, filename)
        label.configure(text = 'Loaded Model')

    else:
        print("train model")
        loaded_model.compile(loss='mean_squared_error', optimizer='adam')
        history = loaded_model.fit([X_train, Indi_train], Y_train, epochs=250 , batch_size=1, verbose=1, shuffle=False)
        loaded_model.save_weights('model/'+ filename+ '/weights.h5')
        label.configure(text = 'Trained Model')  
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.suptitle('Loss')
        ax.plot(history.history['loss'])
        ax.set_ylabel('Loss')
        ax.set_xlabel('Epoch')
        ax.legend(['Train'], loc='upper right')
        fig.savefig('model/'+ filename+ '/loss.png')

    tabControl.select(3)
    label.configure(text="Model Trained")
    img_load = Image.open('model/'+ filename+ '/loss.png')
    render = ImageTk.PhotoImage(img_load)
    img_loss.configure(image=render)
    img_loss.imgage = render
    train_button.configure(state = 'disable')
    predict_button.configure(state = 'enable')


def run_predict():
    global loaded_model
    global data_Y
    global df
    scaler_X11, scaler_X01, scaler_Indi11, scaler_Indi01 , scaler_Y = load_scaler(filename)

    global last_X
    global last_Indi

    #transform predict data
    last_X_temp = last_X.copy()

    last_X_temp[-1] = scaler_X11.transform(last_X_temp[-1].reshape(1,-1))
    temp = last_X_temp[0:5].reshape(1,-1)
    temp = scaler_X01.transform(temp)
    temp = temp.reshape(1,5,10)
    last_X_temp[:5] = temp
    last_X_temp = last_X_temp.reshape(1,6,10)

    last_Indi_temp = last_Indi.copy()
    temp = last_Indi_temp[:-2].reshape(1,-1)
    temp = scaler_Indi01.transform(temp)
    temp = temp.reshape(1,14,10)
    last_Indi_temp[:-2] = temp

    temp = last_Indi_temp[-2:].reshape(1,-1)
    temp = scaler_Indi11.transform(temp)
    temp = temp.reshape(1,2,10)
    last_Indi_temp[-2:] = temp
    last_Indi_temp = last_Indi.reshape(1,16,10)

    pre_train = predict(loaded_model, scaler_Y, X_train, Indi_train)
    pre_test = predict(loaded_model, scaler_Y, x_test, indi_test)
    pre_last = predict(loaded_model, scaler_Y, last_X_temp, last_Indi_temp)

    predictions = np.concatenate([pre_train, pre_test,pre_last, pre_last])

    #plot predictions 
    x = df[87:].index[-1]
    x = pd.to_datetime(x) + timedelta(7)
    print(x.strftime("%Y/%m/%d"))

    fig, ax = plt.subplots(figsize=(15, 7))
    fig.suptitle(filename+' Stock Prediction', fontsize=14)
    ax.plot(df[87:].index, data_Y.reshape(data_Y.shape[0],-1), label='Real')
    ax.set_xticks(df[87:].index[::4])
    ax.set_xticklabels(df[87:].index[::4], rotation=45)
    ax.plot(predictions, label='Predictions')
    ax.plot(len(predictions)-1,predictions[-1][0],'go', label='Next day prediction')
    ax.axvline(x=len(pre_train)+1,color='k', linestyle='--')
    ax.legend(loc='upper left')
    ax.set_xlabel('Time',fontsize = 16)
    ax.set_ylabel('Stock',fontsize = 16)
    ax.annotate(x.strftime("%Y/%m/%d") + "\n"+str(predictions[-1][0]) , # this is the text
                    (len(predictions)-1,predictions[-1]), # this is the point to label
                    textcoords="offset points", # how to position the text
                    xytext=(10,0), # distance from text to points (x,y)
                    fontsize=10,
                    ha='left')
    fig.savefig('model/'+ filename+ '/predictions.png')
    fig.show()

def predict(model, scaler_Y, prices, indi):
  pre = []
  for i in range(len(prices)):
    x = model.predict([prices[i].reshape(1,6,10), indi[i].reshape(1,16,10)])
    x = scaler_Y.inverse_transform(x)
    pre.append(x)
  pre = np.array(pre)
  pre = pre.reshape(pre.shape[0],1)
  return pre

def bollinger_bands(prices):   
    dma = np.empty(len(prices))
    dma = prices.rolling(window=20).mean()
    # set .std(ddof=0) for population std instead of sample
    dstd = prices.rolling(window=20).std() 
    
    upper = dma + (dstd * 2)
    lower = dma - (dstd * 2)
    
    return dma, upper, lower

def RSI(prices, n=14):
    deltas = np.diff(prices)
    seed = deltas[:n+1]
    up = seed[seed>=0].sum()/n
    down = -seed[seed<0].sum()/n
    rs = up/down
    rsi = np.full(len(prices), np.NAN)
    rsi[n-1] = 100. - 100./(1.+rs)

    for i in range(n, len(prices)):
        delta = deltas[i-1] # the diff is 1 shorter

        if delta>0:
            upval = delta
            downval = 0.
        else:
            upval = 0.
            downval = -delta

        up = (up*(n-1) + upval)/n
        down = (down*(n-1) + downval)/n

        rs = up/down
        rsi[i] = 100. - 100./(1.+rs)

    return rsi

def MACD(px):
    #ema26 = pd.ewm(px, span=26).mean()
    ema26 = px.ewm(span=26,adjust=False).mean()
    ema12 = px.ewm(span=12,adjust=False).mean()
    macd = (ema12 - ema26)
    signal = macd.ewm(span=9,adjust=False).mean()

    return macd, signal

def Ichimoku(high,low,close):
    #Kijun Sen
    period26_high = high.rolling(window= 26).max()
    period26_low = low.rolling(window= 26).min()
    kijun_sen = (period26_high + period26_low) / 2

    #Tenkan Sen
    period9_high = high.rolling(window= 9).max()
    period9_low = low.rolling(window= 9).min()
    tenkan_sen = (period9_high + period9_low) / 2

    #Chikou Span
    chikou_span = close.shift(-26)

    #SenkouSpan A
    #senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)


    #SenkouSpan B
    period52_high = high.rolling(window= 52).max()
    period52_low = low.rolling(window= 52).min()
    #senkou_span_b = ((period52_high + period52_low) / 2).shift(26)
    senkou_span_b = ((period52_high + period52_low) / 2)

    return kijun_sen, tenkan_sen, senkou_span_a, senkou_span_b


#### function to calculate indicators
def SAR(high,low,acceleration = 0.02,maximum = 0.2):
    SAR = tb.SAR(high,low,acceleration,maximum)
    return SAR

def Stochastic (high,low,close):
    Low14 = low.rolling(window = 14).min()
    High14 = high.rolling(window = 14).max()
    K = 100*((close - Low14) / (High14 - Low14))
    D = K.rolling(window = 3).mean()
    return K, D

def ADX(high,low,close):
    adxI = ADXIndicator(high,low,close,14,False)
    PDM = adxI.adx_pos()
    NDM = adxI.adx_neg()
    ADX = adxI.adx()
    return PDM,NDM,ADX

#### function to plot indicators  
def bollgraph():
    global df
    fig = plt.figure(figsize=(18, 20))
    fig.suptitle('Bollinger Band', fontsize=40)
    ax = fig.add_axes([0.1, 0.3, 0.8, 0.4])
    ax2 = fig.add_axes([0.1, 0.1, 0.8, 0.2])
    ax.set_xticks(range(0, len(df.index), 20))
    ax.set_xticklabels(df.index[::23])
    mpf.original_flavor.candlestick2_ochl(ax, df['Open'], df['Close'], df['High'], df['Low'], width=0.6,
                          colorup='r', colordown='g', alpha=0.75)
    ax.plot(df['Moving Average'],label='Moving Average')
    ax.plot(df['Upper Band'], label='Upper Band')
    ax.plot(df['Lower Band'], label='Lower Band')
    mpf.original_flavor.volume_overlay(ax2, df['Open'], df['Close'], df['Volume'], colorup='r', colordown='g', width=0.5,
                       alpha=0.8)
    ax2.set_xticks(range(0, len(df.index), 20))
    ax2.set_xticklabels(df.index[::23])
    ax.legend()
    plt.show()

def rsigraph():
    df['RSI'] = RSI(df['Close'])
    fig = plt.figure(figsize=(18, 20))
    fig.suptitle('RSI', fontsize=40)
    ax = fig.add_axes([0.1, 0.3, 0.8, 0.4])
    ax2 = fig.add_axes([0.1, 0.1, 0.8, 0.2])
    ax.set_xticks(range(0, len(df.index), 20))
    ax.set_xticklabels(df.index[::23])
    mpf.original_flavor.candlestick2_ochl(ax, df['Open'], df['Close'], df['High'], df['Low'], width=0.6, colorup='r',colordown='g', alpha=0.75)
    ax2.set_xticks(range(0, len(df.index), 20))
    ax2.set_xticklabels(df.index[::23])
    ax2.plot(df['RSI'], label='RSI')
    ax.legend()
    ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()

def macdgraph():
    df['macd'], df['signal'] = MACD(df['Close'])
    fig = plt.figure(figsize=(18, 20))
    fig.suptitle('MACD', fontsize=40)
    ax = fig.add_axes([0.1, 0.3, 0.8, 0.4])
    ax2 = fig.add_axes([0.1, 0.1, 0.8, 0.2])
    ax.set_xticks(range(0, len(df.index), 20))
    ax.set_xticklabels(df.index[::23])
    mpf.original_flavor.candlestick2_ochl(ax, df['Open'], df['Close'], df['High'], df['Low'], width=0.6, colorup='r',colordown='g', alpha=0.75)
    ax2.set_xticks(range(0, len(df.index), 20))
    ax2.set_xticklabels(df.index[::23])
    ax2.plot(df['macd'], label='MACD')
    ax2.plot(df['signal'], label='Signal')
    ax.legend()
    ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()

def ichigraph():
    df['Kijun_sen'], df['Tenkan_sen'],df['Senkou_span_A'], df['Senkou_span_B'] = Ichimoku(df['High'], df['Low'], df['Close'])
    fig = plt.figure(figsize=(18, 20))
    fig.suptitle('Ichimoku', fontsize=40)
    ax = fig.add_axes([0.1, 0.3, 0.8, 0.4])
    ax2 = fig.add_axes([0.1, 0.1, 0.8, 0.2])
    ax.set_xticks(range(0, len(df.index), 20))
    ax.set_xticklabels(df.index[::23])
    mpf.original_flavor.candlestick2_ochl(ax, df['Open'], df['Close'], df['High'], df['Low'], width=0.6, colorup='r', colordown='g', alpha=0.75)
    ax.plot(df['Kijun_sen'], label='Kijun sen')
    ax.plot(df['Tenkan_sen'], label='Tenkan sen')
    ax.plot(df['Senkou_span_A'], label='Senkou Span A')
    ax.plot(df['Senkou_span_B'], label='Senkou Span B')
    mpf.original_flavor.volume_overlay(ax2, df['Open'], df['Close'], df['Volume'], colorup='r', colordown='g',width=0.5,alpha=0.8)
    ax2.set_xticks(range(0, len(df.index), 20))
    ax2.set_xticklabels(df.index[::23])
    ax.legend()
    plt.show()

def sargraph():
    df['SAR'] = SAR(df['High'], df['Low'], 0.02, 0.2)
    fig = plt.figure(figsize=(18, 20))
    fig.suptitle('SAR', fontsize=40)
    ax = fig.add_axes([0.1, 0.3, 0.8, 0.4])
    ax2 = fig.add_axes([0.1, 0.1, 0.8, 0.2])
    ax.set_xticks(range(0, len(df.index), 20))
    ax.set_xticklabels(df.index[::23])
    mpf.original_flavor.candlestick2_ochl(ax, df['Open'], df['Close'], df['High'], df['Low'], width=0.6, colorup='r', colordown='g', alpha=0.75)
    ax2.set_xticks(range(0, len(df.index), 20))
    ax2.set_xticklabels(df.index[::23])
    ax2.plot(df['SAR'], label='SAR')
    ax.legend()
    ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()

def stochasgraph():
    df['K'], df['D'] = Stochastic(df['High'], df['Low'], df['Close'])
    fig = plt.figure(figsize=(18, 20))
    fig.suptitle('Stochastic', fontsize=40)
    ax = fig.add_axes([0.1, 0.3, 0.8, 0.4])
    ax2 = fig.add_axes([0.1, 0.1, 0.8, 0.2])
    ax.set_xticks(range(0, len(df.index), 20))
    ax.set_xticklabels(df.index[::23])
    mpf.original_flavor.candlestick2_ochl(ax, df['Open'], df['Close'], df['High'], df['Low'], width=0.6, colorup='r',colordown='g', alpha=0.75)
    ax2.set_xticks(range(0, len(df.index), 20))
    ax2.set_xticklabels(df.index[::23])
    ax2.plot(df['K'], label='%K')
    ax2.plot(df['D'], label='%D')
    ax.legend()
    ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()

def adxgraph():
    df['PDM'], df['NDM'], df['ADX'] = ADX(df['High'], df['Low'], df['Close'])
    fig = plt.figure(figsize=(18, 20))
    fig.suptitle('ADX', fontsize=40)
    ax = fig.add_axes([0.1, 0.3, 0.8, 0.4])
    ax2 = fig.add_axes([0.1, 0.1, 0.8, 0.2])
    ax.set_xticks(range(0, len(df.index), 20))
    ax.set_xticklabels(df.index[::23])
    mpf.original_flavor.candlestick2_ochl(ax, df['Open'], df['Close'], df['High'], df['Low'], width=0.6, colorup='r', colordown='g', alpha=0.75)
    ax2.set_xticks(range(0, len(df.index), 20))
    ax2.set_xticklabels(df.index[::23])
    ax2.plot(df['ADX'], label='ADX')
    ax.legend()
    ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()

class Example(Frame):
    def __init__(self, parent):
        Frame.__init__(self, parent)
        self.parent = parent
        self.initUI()
  
    def initUI(self):
        self.parent.title("Stock predition")
        self.style = Style()
        self.style.theme_use("default")
        
        global tabControl
        tabControl = Notebook(self) 

        global frame1
        global table_data
        frame1 = Frame(self, relief=RAISED, borderwidth=1)
        table_data = Table(frame1, dataframe= pd.DataFrame(), showtoolbar=True, showstatusbar=True)
        frame1.pack(fill=BOTH, expand=True)
        
        global frame2
        global frame21
        global table_indi
        frame2 = Frame(self, relief=RAISED, borderwidth=1)
        frame21 = Frame(frame2, relief=RAISED, borderwidth=1)
        table_indi = Table(frame21, dataframe= pd.DataFrame(), showtoolbar=True, showstatusbar=True)
        frame21.pack(fill=BOTH, expand=True)
        frame22 = Frame(frame2, relief=RAISED, borderwidth=1)
        a = Button(frame22, text="Bollinger band", command = bollgraph)
        a.pack(side=LEFT, padx=5, pady=5)
        b = Button(frame22, text="Relative Strength Index", command = rsigraph)
        b.pack(side=LEFT, padx=5, pady=5)
        c = Button(frame22, text="Moving Average Convergence Divergence", command = macdgraph)
        c.pack(side=LEFT, padx=5, pady=5)
        d = Button(frame22, text="Ichimoku", command = ichigraph)
        d.pack(side=LEFT, padx=5, pady=5)
        e = Button(frame22, text="Stop and Reverse", command = sargraph)
        e.pack(side=LEFT, padx=5, pady=5)
        f = Button(frame22, text="Stochastic", command = stochasgraph)
        f.pack(side=LEFT, padx=5, pady=5)
        g = Button(frame22, text="Average Directional Index", command = adxgraph)
        g.pack(side=LEFT, padx=5, pady=5)
        frame22.pack(side = BOTTOM)
        frame2.pack(fill=BOTH, expand=True)

        global frame3
        global txt_X_train
        global txt_Indi_train
        global txt_Y_train
        global filename
        frame3 = Frame(self, relief=RAISED, borderwidth=1)
        txt_X_train = Text(frame3,font=("Helvetica", 15), width = 60, highlightbackground="black", highlightthickness=1)
        txt_X_train.pack(side=LEFT, fill=Y, padx=5, pady=5 )
        txt_Indi_train = Text(frame3,font=("Helvetica", 15), width = 60, highlightbackground="black", highlightthickness=1)
        txt_Indi_train.pack(side=LEFT, fill=Y, padx=5, pady=5 )
        txt_Y_train = Text(frame3,font=("Helvetica", 15), width = 30, highlightbackground="black", highlightthickness=1)
        txt_Y_train.pack(side=LEFT, fill=Y, padx=5, pady=5 )
        frame3.pack(fill=BOTH, expand=True)

        global frame4
        global img_loss
        global img
        frame4 = Frame(self, relief=RAISED, borderwidth=1)
        img = ImageTk.PhotoImage(Image.open('placeholder.png'))
        img_loss = Label(frame4, image=img)
        img_loss.pack(side = LEFT, fill = BOTH, expand = True)
        frame4.pack(fill=BOTH, expand=True)
        
        self.pack(fill=BOTH, expand=True)

        tabControl.add(frame1, text = 'Data')
        tabControl.add(frame2, text = 'Indicators')
        tabControl.add(frame3, text = 'Preprocessed Data')
        tabControl.add(frame4, text = 'Model')
        tabControl.pack(expand = 1, fill ="both") 

        load_button = Button(self, text="Load", command = load_file)
        load_button.pack(side=LEFT, padx=5, pady=5)

        global cal_indi_button
        cal_indi_button = Button(self, text="Calculate Indicators",state = 'disable', command = cal_indi_func)
        cal_indi_button.pack(side=LEFT, padx=5, pady=5)

        global preprocess_button
        preprocess_button = Button(self, text="Preprocess",state = 'disable', command = preprocessing)
        preprocess_button.pack(side=LEFT, padx=5, pady=5)

        global train_button
        train_button = Button(self, text="Train Model",state = 'disable', command = train_model)
        train_button.pack(side=LEFT, padx=5, pady=5)

        global predict_button
        predict_button = Button(self, text="Predict",state = 'disable', command = run_predict)
        predict_button.pack(side=LEFT, padx=5, pady=5)

        global label
        label = Label(self)
        label.pack(side=LEFT, padx=5)

def quit_me():
    print('quit')
    root.quit()
    root.destroy()

df = pd.DataFrame()
root = Tk()
root.protocol("WM_DELETE_WINDOW", quit_me)
root.geometry("1500x800")
app = Example(root)
root.mainloop()




