#Leon Gurtler
#Liner Regression as a tool to predict share prices based on twitter mentions
import got3, time, datetime, sklearn, math
import numpy as np
import pandas as pd
from pandas_datareader import data as wb
from sklearn import preprocessing, svm, model_selection
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import scipy.sparse as sparse


tickers = [['AIZ', 'Assurant']]
comp_data = pd.DataFrame()
twitter_data = []


#append the relevant data of each s&P500 stock into the comp_data data frame. The historical data dates back 5 years and is taken form the yahoo finance API
for x in range(0, len(tickers)):
    comp_data[tickers[x][0]] =  wb.DataReader(tickers[x][0], data_source = 'yahoo', start = '2015-01-01')['Adj Close']
    date_array = comp_data.index.values

    for y in range(0, len(date_array)):
        Date_prior = (pd.to_datetime(str(date_array[y-1]))).strftime('%Y-%m-%d')
        Date_now = (pd.to_datetime(str(date_array[y]))).strftime('%Y-%m-%d')

        #calculate the day difference between the two timestamps, in order to be able to get the avarage twitter mentiones per day
        Day_number = (str((date_array[6] - date_array[5]) / (8.64*10**13))[0])

        #in order to get the most accurate result, the program searches for both, the ticker and the actual company name
        tweetCriteria_ticker = got3.manager.TweetCriteria().setQuerySearch(tickers[x][0]).setSince(Date_prior).setUntil(Date_now)
        tweetCriteria_name = got3.manager.TweetCriteria().setQuerySearch(tickers[x][1]).setSince(Date_prior).setUntil(Date_now)

        #to get a meaningful output, it is necessary to divide the total tweet mentions by the number of days they were counted on.
        #This is becuase the stock market is closed on weekends and bank holidays
        tweet_mention_count = (len(got3.manager.TweetManager.getTweets(tweetCriteria_name)) + (len(got3.manager.TweetManager.getTweets(tweetCriteria_ticker))) / int(Day_number))
        print(tweet_mention_count)
        twitter_data.append(tweet_mention_count)


#The following lines create a new (temporary) dataframe, which holds the Dates as an index and the mention count as a proper value. This is necessary
#in order to be able to join the two dataframes together
temp_frame = pd.DataFrame({'Date' : date_array, 'Mentions' : twitter_data})
temp_frame = temp_frame.set_index('Date')
comp_data = comp_data.join(temp_frame)


##Liner Regression
forecast_col = 'Mentions'
comp_data.fillna('-99999', inplace = True)

forecast_out = int(math.ceil(0.01*len(comp_data)))

comp_data['label'] = comp_data['MSFT'].shift(-forecast_out)

X = np.array(comp_data.drop(['label'], 1))
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
X = X[:-forecast_out:]


comp_data.dropna(inplace = True)
y = np.array(comp_data['label'])

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.2)

clf = LinearRegression(n_jobs = -1)  #-1 is for as many as possible
clf.fit(X_train, y_train)


accuracy = clf.score(X_test, y_test)
forecast_set = clf.predict(X_lately)

print(forecast_set, accuracy, forecast_out)
comp_data['Forecast'] = np.nan

last_date = comp_data.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    comp_data.loc[next_date] = [np.nan for _ in range(len(comp_data.columns)-1)] + [i]

comp_data[tickers[0]].plot()
comp_data['Forecast'].plot()
plt.legend(loc = 4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
