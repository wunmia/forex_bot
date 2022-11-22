'''
    HI ALL AND WELCOME TO MY FOREX PREDICTION TOOL
    - This model takes historical FX Data from a Demo Trading Platform and Applies a range of technical indicators to it
    - The Machine Learning Package Scikit Learn is then used to Predict on a T+2 basis whether or not the FX pair will appreciate or depreciate
    - There are a number of optimisations in the script that will be explained throughout the code
    - The Final result is a set if 10 predictive indicators for each currency pair examined, which can be used on a real time machine
    learning model in order to predict he T+2 hours price
'''

import MetaTrader5 as mt5
from datetime import datetime
import pytz
import pandas as pd
import numpy as np
import sqlite3
import ta
import random
import warnings
import statistics
import sklearn
from sklearn import svm
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from operator import itemgetter
import pickle

'''This section has some base arguments custom functions/decorators that are generic and applied to many of my codes'''
pairs = ["EURUSD","GBPUSD","USDCAD","USDJPY"]  
warnings.filterwarnings('ignore')

'''Set speed sample sixe for first quick optimisations and slow sample size for the more rigorous optimisations, the recommended settings are commented out'''
speed_sample_size = 100 #4000
slow_sample_size = 500 #15000

def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)


'''This Class is downlaods historical time data, for the currencies that will be used in the optimisations'''
class HistoricalData():
    """loading variables into model, looking at some of the big fx currency pairs"""
    def __init__(self):
        self.timeout = 60000
        self.path = "C:\Program Files\MetaTrader 5"
        self.timezone = pytz.timezone("Europe/London")         # set time zone to UK
        self.utc_from = datetime(2021, 10, 20, tzinfo=self.timezone)
        self.utc_to = datetime.now() #hour = 13,  hour argument possible too
        self.datasets_list = []
        self.datasets_dict = {}
        creds_file = open("creds\\creds.txt").readlines()
        self.username = int(str(creds_file[0]).rstrip('\n').strip())
        self.password = str(creds_file[1]).rstrip('\n').strip()


    '''Connecting to demo platform'''
    def initialize_client(self):
        if not mt5.initialize(login=self.username, server="AdmiralMarkets-Demo",password=self.password, timeout=self.timeout, portable=False):
            print("initialize() failed, error code =",mt5.last_error())
            quit()
        print("Initialisation of Client Complete\n\n")

    '''pulling out one hour time frames with columns, high/low/open/close/volume'''
    def data_pull(self):
            for currency in pairs:
                rates = mt5.copy_rates_range(currency, mt5.TIMEFRAME_H1, self.utc_from, self.utc_to)
                rates_frame = pd.DataFrame(rates)
                rates_frame['time']=pd.to_datetime(rates_frame['time'], unit='s')
                rates_frame.rename(columns={"tick_volume":"volume"}, inplace=True)
                rates_frame = rates_frame.drop(columns=["spread","real_volume"])
                self.datasets_dict[currency]=rates_frame
                print(f"Price Data Pulled for {currency}")

    '''saves the data in an sqlite file'''
    def saving_down(self):
        conn = sqlite3.connect('historical_fx_data.sqlite')
        for k, v in self.datasets_dict.items():
            self.datasets_dict[k].to_sql(f"ta_{k}_1h", conn, if_exists='replace', index=False)
        conn.commit()
        print("\nData Pulling Process Complete")

'''This class adds technical analysis to underlying price data, comprising of variance, momentum, volume and regression features'''
'''for each technical indicator a combination of time, price and volume arguments are entered'''
class TA:
        def __init__(self):
            self.pair = pair
            self.conn = sqlite3.connect('historical_fx_data.sqlite')
            self.df = pd.read_sql(f"Select * From [ta_{self.pair}_1h]", self.conn)

        '''since the time windows are small (1 hrs) the average price through the hour is calculated and this is what 
        is used as the price for the machine learning models'''
        def calculated_columns(self):
            for column in self.df.columns.drop(["time"]):
                self.df[column]= pd.to_numeric(self.df[column], errors="coerce")
                self.df[column] = self.df[column]*10000
            self.df["OHLC_avg"] = (self.df["high"]+self.df["low"]+self.df["close"]+self.df["open"])/4
            self.df["OHLC_avg+2"] = self.df["OHLC_avg"].shift(periods=-2, axis = 0)
            self.df["delta"] = 100*(self.df["OHLC_avg+2"] - self.df["OHLC_avg"])/self.df["OHLC_avg"]
            self.df["trend"] = 0
            self.df["trend"][(self.df["delta"] >= self.df["delta"].quantile(0.63)) & (self.df["delta"] > 0) ] = 1
            self.df["trend"][(self.df["delta"] <= self.df["delta"].quantile(0.37)) & (self.df["delta"] < 0)] = -1
            print("\n\nTechnical Analysis Section\n\n")

        '''bollinger bands - measures the % outside the 2 standard deviation moving average'''
        def bb_features(self):
            print(f"doing bollinger bands for {pair}")
            for m in range(2,20):
                bb = ta.volatility.BollingerBands(close=self.df["close"], window=m, window_dev=2)
                self.df[f'bb_bbi: {m}'] = (self.df["OHLC_avg"]-bb.bollinger_mavg())/bb.bollinger_mavg()

        '''momentum features'''
        def momentum_features(self):
            print(f"doing momentums for {pair}")
            for n in range(2,20):
                self.df[f'roc mom: {n}'] = ta.momentum.roc(self.df["close"], window=n, fillna=False)
                self.df[f'williams r mom: {n}'] = ta.momentum.williams_r(high=self.df["high"], low=self.df["low"], close=self.df["close"], lbp=n+5, fillna=False)
                self.df[f'kama mom: {n}'] = ta.momentum.kama(close=self.df["close"], window=n, pow1=2, pow2=30, fillna=False)
            for n in range(7,16):
                for m in range(1,4):
                    self.df[f'momentum sma: {m}:{n}'] = ta.momentum.awesome_oscillator(high=self.df["high"],low=self.df["low"], window1=m, window2=n)
                    self.df[f'momentum_ppo: {m}:{n}'] = ta.momentum.ppo(close=self.df["close"], window_slow=n, window_fast=m, window_sign=3, fillna=False)
                    self.df[f'momentum_pvo: {m}:{n}'] = ta.momentum.pvo(volume=self.df["volume"], window_slow=n, window_fast=m, window_sign=3, fillna=False)
                    self.df[f'momentum_stoch: {m}:{n}'] = ta.momentum.stoch(high=self.df["high"], low=self.df["low"], close=self.df["close"], window=n, smooth_window=m, fillna=False)

        '''volume features'''
        def volume_features(self):
            print(f"doing volumes for {pair}")
            for period in range(15000, len(self.df), 1000):
                self.df[f'volvpt: {period}'] = ta.volume.volume_price_trend(close=self.df["close"][-period:], volume=self.df["volume"][-period:], fillna=False)
                self.df[f'voladi: {period}'] = ta.volume.acc_dist_index(high=self.df["high"][-period:], low=self.df["low"][-period:], close=self.df["close"][-period:], volume=self.df["volume"][-period:], fillna=False)
            for period in range(2,20):
                self.df[f'volvwap: {period}'] = ta.volume.volume_weighted_average_price(high=self.df["high"], low=self.df["low"], close=self.df["close"], volume=self.df["volume"], window=period, fillna=False)
                self.df[f'volforce: {period}'] = ta.volume.force_index(close=self.df["close"], volume=self.df["volume"], window=period, fillna=False)
                self.df[f'volea: {period}'] = ta.volume.ease_of_movement(high=self.df["high"], low=self.df["low"], volume=self.df["volume"], window=period, fillna=False)
                self.df[f'volmf: {period}'] = ta.volume.money_flow_index(high=self.df["high"], low=self.df["low"], close=self.df["close"], volume=self.df["volume"], window=period, fillna=False)

        '''trend features'''
        def trend_features(self):
            print(f"doing trends for {pair}")
            for n in range(10,20):
                for m in range(2,7):
                    self.df[f'macd: {m}:{n}'] = ta.trend.macd(self.df["close"], window_slow=n, window_fast=m, fillna=False)
                    self.df[f'stc: {m}:{n}'] = ta.trend.stc(self.df["close"], window_slow=n, window_fast=m, cycle=5, smooth1=3,smooth2=3, fillna=False)
                    self.df[f'mass_trend: {m}:{n}'] = ta.trend.mass_index(high=self.df["high"], low=self.df["low"], window_fast=m, window_slow=n, fillna=False)
                    self.df[f'macd_diff: {m}:{n}'] = ta.trend.macd_diff(close=self.df["close"], window_slow=n, window_fast=m, window_sign=3, fillna=False)
            for n in range(2,20):
                self.df[f'trend_trix{n}'] = ta.trend.trix(self.df["close"], window=n, fillna=False)
                self.df[f'ema: {n}'] = ta.trend.ema_indicator(self.df["close"], window=n, fillna=False)
                self.df[f'sma: {n}'] = ta.trend.sma_indicator(self.df["close"], window=n, fillna=False)
                self.df[f'trend _cci: {n}'] = ta.trend.cci(high=self.df["high"], low=self.df["low"], close=self.df["close"], window=n, constant=0.015, fillna=False)
                self.df[f'trend _adx: {n}'] = ta.trend.adx(high=self.df["high"], low=self.df["low"], close=self.df["close"], window=n, fillna=False)
                self.df[f'wma: {n}'] = ta.trend.wma_indicator(self.df["close"], window=n, fillna=False)

        '''misc features'''
        def other_features(self):
            print(f"doing others for {pair}")
            for period in range(15500, len(self.df), 1000):
                self.df[f'others_dlr:{period}'] = ta.others.daily_log_return(self.df["close"][-period:], fillna=False)
                self.df[f'others_dr:{period}'] = ta.others.daily_return(self.df["close"][-period:], fillna=False)
                self.df[f'others_cr:{period}'] = ta.others.cumulative_return(self.df["close"][-period:], fillna=False)

        '''one dataframe per'''
        def save(self):
            print(self.df)
            self.df = self.df.applymap(str)
            self.df.to_sql(name=f"ta_{self.pair}_1h",con=self.conn, if_exists='replace', index=False)
            self.conn.commit()

'''This class is used to reduce down the number of features for each currency pair from 400 to 10, using multiple learnings each time getting more rigorous/detailed '''
class LearningModels():
    def __init__(self):
        self.conn = sqlite3.connect("historical_fx_data.sqlite")
        self.overall_analysis = []

    '''takes all features (400+) and using SVC model to reduced down to the 150 most accurate features using a quick correlation analysis
    ,test is only repeated three times for speed, takes only last 4000 data points'''
    def first_learn(self):
        for pair in pairs:
            df = pd.read_sql(f"Select * From [ta_{pair}_1h]", self.conn)
            df = df.drop(columns=["time","open","high","low","close","volume","OHLC_avg","OHLC_avg+2","delta","kama mom: 2","kama mom: 3","kama mom: 4","kama mom: 5","kama mom: 6","kama mom: 7","kama mom: 8","kama mom: 9","kama mom: 10"]).replace([np.inf, -np.inf], 0)
            df = clean_dataset(df)
            print(pair)
            best_acc = 0
            performance = {}
            rank = []

            for indicator in df.columns.drop(["trend"]):
                score = []
                for n in range(0,3):
                    df_test = df[["trend", indicator]].dropna()[-speed_sample_size:]
                    x = np.array(df_test[indicator].fillna(0))
                    y = np.array(df_test["trend"].fillna(0))
                    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)

                    x_train= x_train.reshape(-1, 1)
                    y_train= y_train.reshape(-1, 1)
                    x_test = x_test.reshape(-1, 1)

                    norm = MinMaxScaler().fit(x_train)
                    x_train = norm.transform(x_train)
                    x_test = norm.transform(x_test)

                    clf = svm.SVC(C=1)
                    clf.fit(x_train, y_train)
                    y_pred = clf.predict(x_test)
                    acc = metrics.accuracy_score(y_test, y_pred)
                    score.append(acc)

                average = statistics.mean(score)
                performance[indicator] = average
                if average > best_acc:
                    best_acc = average
            for key, value in sorted(performance.items(), key = itemgetter(1), reverse = True): rank.append(key)
            features = ["trend"]
            for n, feat in enumerate(rank):
                features.append(feat)
                if n == 151: break
            df_ml = df[features][-slow_sample_size:]
            print(df_ml)
            df_ml.to_sql(f"mlf_{pair}_1h", self.conn, if_exists='replace', index=False)

    '''same process repeated but takes last 15000 data points and reduces features down to 50 from 150 with a more rigorous learning model'''
    def second_learn(self):
        for pair in pairs:
            df = pd.read_sql(f"Select * From [mlf_{pair}_1h]", self.conn)
            df = df[-slow_sample_size+1:-1]
            df = clean_dataset(df)
            best_acc = 0
            performance = {}
            rank = []

            for indicator in df.columns.drop(["trend"]):
                score = []
                for n in range(0,1):
                    df_test = df[["trend", indicator]].dropna()
                    x = np.array(df_test[indicator])
                    y = np.array(df_test["trend"])
                    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)

                    x_train= x_train.reshape(-1, 1)
                    y_train= y_train.reshape(-1, 1)
                    x_test = x_test.reshape(-1, 1)

                    norm = MinMaxScaler().fit(x_train)
                    x_train = norm.transform(x_train)
                    x_test = norm.transform(x_test)

                    clf = svm.SVC(C=1)
                    clf.fit(x_train, y_train)
                    y_pred = clf.predict(x_test)
                    acc = metrics.accuracy_score(y_test, y_pred)
                    score.append(acc)
                average = statistics.mean(score)
                performance[indicator] = average
                if average > best_acc:
                    best_acc = average
                    print(f"Best Indicator so far is {indicator}, Avarage score of {round(best_acc,3)}")
            for key, value in sorted(performance.items(), key = itemgetter(1), reverse = True): rank.append(key)

            features = ["trend"]
            for n, feat in enumerate(rank):
                features.append(feat)
                if n == 49: break
            print(f"pair: {pair}, features {features}")
            df_ml = df[features][-slow_sample_size:]
            print(df_ml)
            df_ml.to_sql(f"mlf1_{pair}_1h", self.conn, if_exists='replace', index=False)

    '''This id the most in-depth analysis, the model randomly selects 10 out of the 50 features and tests their performance on the hostorical data
    - model does the random selection 150 times for each currency pair and picks the best at the end - based on
    - this then scores the models on two metrics, accuracy (% of call made that were correct) and responsiveness (% of the time the model made a call)'''
    def third_learn(self):
        analysis = pd.DataFrame(columns=["responsiveness","accuracy","features"])
        def range_with_floats(start, stop, step):
            while stop > start:
                yield start
                start += step
        pairs_features_selection = {'EURUSD':10,'GBPUSD':10,'USDJPY':10,'USDCAD':10} #["EURUSD","USDJPY","GBPUSD","USDCAD"]
        analyses = []
        for pair, f in pairs_features_selection.items():
            df = pd.read_sql(f"Select * From [mlf1_{pair}_1h]", self.conn)
            df = df[50:-1].dropna()
            best_accuracy = 0
            trial = 0
            score = []
            accuracy_list = []
            features = []
            responsiveness = []
            for n in range(0,150):
                total_predictions = []
                pred_makes_call = []
                trial=trial+1
                features_permutation = random.sample(df.columns.to_list()[1:], f)
                df1 = df["trend"]
                df2 = df[features_permutation]
                x = np.array(df2)
                y = np.array(df1)
                x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.15, shuffle=False)

                norm = MinMaxScaler().fit(x_train)
                x_train = norm.transform(x_train)
                x_test = norm.transform(x_test)

                clf = svm.SVC(C=1)
                clf.fit(x_train, y_train)
                y_pred = clf.predict(x_test)
                perf = []
                for x in range(len(y_pred)):
                    total_predictions.append(y_test[x])
                    if y_pred[x] == 0: continue
                    pred_makes_call.append(y_pred[x])
                    if y_pred[x] == y_test[x]: perf.append(100)
                    if y_pred[x] != y_test[x]: perf.append(0)
                if len(perf) != 0:
                    accuracy = statistics.mean(perf)
                    score.append(accuracy)
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        features_permutation.append("trend")
                        accuracy_list.append(best_accuracy)
                        responsiveness.append((len(pred_makes_call)/len(total_predictions))*100)
                        g_guess = features_permutation
                        features.append(g_guess)
            analysis["responsiveness"], analysis["accuracy"], analysis["features"], analysis["product"], analysis["market"]= np.array(responsiveness), np.array(accuracy_list), pd.DataFrame({"features": features}), pair, "forex"
            analyses.append(analysis)
            print(analysis)
            analysis =  pd.DataFrame(columns=["responsiveness","accuracy","features"])
            average = statistics.mean(score)
        analyses = pd.concat(analyses)
        print(analyses)
        analyses.to_excel(f"forex analysis.xlsx")

'''After getting bgest 10 features we are now fine tuning the model by altering model parameter values'''
'''In the pair features dictionary we add our best pairs from the "forex analysis.xlsx file", '''
class ModelTuning:
    def __init__(self):
        self.conn = sqlite3.connect("historical_fx_data.sqlite")
        self.timeout = 60000
        self.predictions = []
        self.models = []
    '''This tunes the sensitivity parameter C by looping through 0-30 in the parameter ranges'''
    def first_opt(self):
        def range_with_floats(start, stop, step):
            while stop > start:
                yield start
                start += step
        analyses = []
        pairs_features = {
        "EURUSD":
            ['sma: 12', 'volea: 2', 'stc: 6:17', 'sma: 13', 'macd_diff: 6:17', 'volforce: 4', 'macd_diff: 3:14', 'wma: 3', 'volvwap: 7', 'volvwap: 13', 'trend'],
        "USDJPY":
            ['macd_diff: 2:16', 'momentum_stoch: 1:15', 'ema: 16', 'volforce: 4', 'roc mom: 3', 'stc: 5:18', 'volmf: 2', 'trend _cci: 10', 'ema: 5', 'trend _cci: 2', 'trend'],
        "GBPUSD":
            ['volforce: 3', 'williams r mom: 11', 'volforce: 7', 'trend _adx: 5', 'momentum_ppo: 1:8', 'williams r mom: 2', 'volmf: 7', 'momentum_stoch: 1:7', 'volforce: 15', 'volea: 10', 'trend'],
        "USDCAD":
            ['mass_trend: 5:16', 'mass_trend: 4:14', 'volforce: 10', 'mass_trend: 3:14', 'macd_diff: 2:10', 'macd: 6:19', 'volea: 7', 'stc: 2:10', 'macd: 4:17', 'volea: 16', 'trend']


        }
        c_score = []
        for pair, f in pairs_features.items():
            df = pd.read_sql(f"Select * From [mlf1_{pair}_1h]", self.conn)
            analysis = pd.DataFrame(columns=["responsiveness","accuracy","c"])
            accuracy = []
            responsiveness_score = []
            margin = []

            best_score = {"c":0}
            for c in range_with_floats(0.01,30.01, 0.5):
                trial = 0
                score = []
                responsiveness = []
                test_d = []
                pred_d = []
                for n in range(0,1):
                    trial=trial+1
                    df1 = df["trend"]
                    df2 = df[f].drop("trend", axis=1)
                    x = np.array(df2)
                    y = np.array(df1)
                    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.15, shuffle=False)
                    norm = MinMaxScaler().fit(x_train)
                    x_train = norm.transform(x_train)
                    x_test = norm.transform(x_test)
                    clf = svm.SVC(C=c)
                    clf.fit(x_train, y_train)
                    y_pred = clf.predict(x_test)
                    perf = []
                    for x in range(len(y_pred)):
                        test_d.append(y_test[x])
                        if y_pred[x] == 0: continue
                        pred_d.append(y_pred[x])
                        if y_pred[x] == y_test[x]: perf.append(100)
                        if y_pred[x] != y_test[x]: perf.append(0)
                    if len(perf) != 0:
                        acc = statistics.mean(perf)
                        score.append(acc)
                        capture = len(pred_d)/len(test_d)
                        responsiveness.append(capture)
                    if acc > list(best_score.values())[0]:
                        best_score = {c:acc}
                if len(score) > 0:
                    average = statistics.mean(score)
                    if average > 50:
                        responsiveness = statistics.mean(responsiveness)*100
                        accuracy.append(average)
                        responsiveness_score.append(responsiveness)
                        margin.append(c)
            c_score.append(list(best_score.keys())[0])
            analysis["responsiveness"], analysis["accuracy"], analysis["c"], analysis["product"], analysis["market"] = np.array(responsiveness_score), np.array(accuracy), np.array(margin), pair, "forex"
            analysis["combination"] = analysis["responsiveness"] + analysis["accuracy"]
            analyses.append(analysis)
        analyses = pd.concat(analyses)
        analyses.to_excel("optimised.xlsx")
        self.pairs_features = pairs_features
        self.c_score = c_score

    '''Trains on the max dataset possible with the Optimised Learning Models, it then saves the model for later deployment'''
    def final_train(self): #may need to have a rolling 50 period data set to to put in all the technical analysis features

        for n, (k,v) in enumerate(self.pairs_features.items()):
            df = pd.read_sql(f"Select * From [mlf_{k}_1h]", self.conn)

            print(f"Final {k} Optimisation")
            target = df["trend"]
            variables = df[v].drop("trend", axis=1)
            x = np.array(variables)
            y = np.array(target)
            x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=1/7500, shuffle=False)
            norm = MinMaxScaler().fit(x_train)
            x_train = norm.transform(x_train)
            x_test = norm.transform(x_test)
            try:
                print(self.c_score[n])
                clf = svm.SVC(C=self.c_score[n])
                clf.fit(x_train, y_train)
                self.models.append(clf)
                # save
                with open(f'models/{k} model.pkl', 'wb') as f:
                    pickle.dump(clf, f)

            except Exception as e:
                print(e)
                continue


if not input("Do you want to Skip the Data Pull and Clean (y)? \n") == "y":
    # Data Download
    data_class_object = HistoricalData()
    data_class_object.initialize_client()
    data_class_object.data_pull()
    data_class_object.saving_down()

    # Tehcnical Analysis
    for pair in pairs:
        ta_object = TA()
        ta_object.calculated_columns()
        ta_object.bb_features()
        ta_object.momentum_features()
        ta_object.volume_features()
        ta_object.trend_features()
        ta_object.other_features()
        ta_object.save()

#Market Machine Learning
#Features Selection
if not input("Do you want to Skip the Learning (y)? \n") == "y":
    learning_object = LearningModels()
    learning_object.first_learn()
    learning_object.second_learn()
    learning_object.third_learn()

if input("Do you want to updates pairs before continuing (y)? ") == "y": exit()
tuning_object =  ModelTuning()
tuning_object.first_opt()
tuning_object.final_train()

