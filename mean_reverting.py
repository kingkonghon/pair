import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.tsa.stattools as sm2
import datetime as dt
import matplotlib.pyplot as plt

path = 'F:\\testdata\\dukas\\AUDNZD_M1.csv'
price_buffer = pd.read_csv(path,header=None)
close_price = price_buffer[4]
temp_date = np.array(pd.to_datetime(price_buffer[0], format='%Y.%m.%d %H:%M'))
close_price.index = temp_date
'''
start_time = dt.datetime(2014,12,1)
time_interval = dt.timedelta(days=2)
end_time = start_time + time_interval

for i in range(0,20):
    test_data = close_price.loc[start_time:end_time]
    result = sm2.adfuller(test_data)
    test_statistic = result[0]
    critical_value = result[4]['5%']
    if test_statistic < critical_value:
        print 'found:',start_time,end_time
    start_time = end_time
    end_time = start_time + time_interval
'''
start_time = dt.datetime(2014,12,23)
end_time = dt.datetime(2014,12,25)

plt.hist(close_price.loc[start_time:end_time],bins=50)
plt.show()