import pandas as pd
import statsmodels.api as sm
import statsmodels.tsa.stattools as sm2
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import os

price_buffer = pd.read_csv('F:\\testdata\\dukas\\AUDUSD_M1.csv',header=None)
temp_date = np.array(pd.to_datetime(price_buffer[0], format='%Y.%m.%d %H:%M'))
price_buffer.index = temp_date
close_price = pd.DataFrame(price_buffer[4])
close_price.columns = ['AUDUSD']

price_buffer = pd.read_csv('F:\\testdata\\dukas\\USDCAD_M1.csv',header=None)
temp_date = np.array(pd.to_datetime(price_buffer[0], format='%Y.%m.%d %H:%M'))
price_buffer.index = temp_date
close_price['USDCAD'] = price_buffer[4]
close_price['USDCAD'] = 1/close_price['USDCAD']

close_price = close_price.dropna(how='any')

start_time = dt.datetime(2014,1,1)
end_time = dt.datetime(2014,1,15)
test_time = dt.datetime(2014,1,29)
threshold_time = dt.datetime(2014,1,29)
final_time = dt.datetime(2016,1,1)
interval = test_time - end_time
threshold_time_interval = dt.timedelta(days=30)

intercept = []
beta = []
beta_index = []
residual = pd.Series([])
#threshold_s = pd.Series([])
#threshold_l = pd.Series([])
entry_point = pd.Series([])

'''
while end_time < final_time:
    is_nonstationary_serie = True
    test_price = close_price.loc[start_time:end_time]
    result = sm2.adfuller(test_price['AUDUSD'])
    test_statistic = result[0]
    critical_value = result[4]['5%']
    if test_statistic < critical_value:
        is_nonstationary_serie = False
        print 'AUDUSD stationary'
    result = sm2.adfuller(test_price['USDCAD'])
    test_statistic = result[0]
    critical_value = result[4]['5%']
    if test_statistic < critical_value:
        is_nonstationary_serie = False
        print 'USDCAD stationary'

    if is_nonstationary_serie:
        result = sm2.coint(test_price['AUDUSD'],test_price['USDCAD'])
        test_statistic = result[0]
        critical_value = result[2][1]
        if test_statistic < critical_value:
            y = test_price['AUDUSD']
            x = sm.add_constant(test_price['USDCAD'])
            result = sm.OLS(y,x).fit()
            intercept.append(result.params[0])
            beta.append(result.params[1])
            print start_time, end_time, result.params[0], result.params[1]
            this_residual = test_price['AUDUSD'] - result.params[0] - result.params[1] * test_price['USDCAD']
            residual = pd.concat([residual,this_residual])
        else:
            print 'not cointegrated'

    start_time += interval
    end_time += interval
'''
threshold_large = 100  # make up a very large threshold
threshold_small = -100
while test_time < final_time:
    regress_price = close_price.loc[start_time:end_time]
    test_price = close_price.loc[end_time:test_time]

    y = regress_price['AUDUSD']
    x = sm.add_constant(regress_price['USDCAD'])
    result = sm.OLS(y, x).fit()
    this_beta = round(result.params[1],2)
    intercept.append(result.params[0])
    beta.append(this_beta)
    beta_index.append(end_time)
    this_residual = test_price['AUDUSD'] - result.params[0] - this_beta * test_price['USDCAD']
    criterion = this_residual.map(lambda x:(x > threshold_large) or (x < threshold_small))
    this_entry_point = this_residual[criterion]
    entry_point = pd.concat([entry_point,this_entry_point])
    #this_threshold_l = pd.Series([threshold_large] * this_residual.size)
    #this_threshold_l.index = this_residual.index
    #this_residual_s = pd.Series([threshold_small] * this_residual.size)
    #this_residual_s.index = this_residual.index
    #threshold_l = pd.concat([threshold_l,this_threshold_l])
    #threshold_s = pd.concat([threshold_s,this_residual_s])
    residual = pd.concat([residual, this_residual])

    if test_time - threshold_time >= threshold_time_interval:
        try:
            past_residual = residual[threshold_time:test_time]
            extreme_range = int(past_residual.size * 0.05)
            threshold_large = past_residual.nlargest(extreme_range)[extreme_range - 1]
            threshold_small = past_residual.nsmallest(extreme_range)[extreme_range - 1]
            print threshold_large, threshold_small, threshold_time, test_time
            threshold_time = test_time
        except TypeError, IndexError:
            threshold_large = 100  # make up a very large threshold
            threshold_small = -100

    start_time += interval
    end_time += interval
    test_time += interval


s_beta = pd.Series(beta)
s_beta.index = beta_index

plt.subplot(311)
plt.plot(intercept)
plt.subplot(312)
plt.plot(beta)
plt.subplot(313)
plt.plot(residual)
plt.plot(entry_point)

plt.show()

#s_beta.to_csv('F:\\testdata\\beta.csv',header=False,mode='w')
#entry_point.to_csv('F:\\testdata\\entry_point.csv',header=False,mode='w')

'''
test_data = close_price.loc['20140101':'20140201']
result = sm2.adfuller(test_data)


price_buffer = pd.read_csv('F:\\testdata\\dukas\\USDCAD_M1.csv',header=None)

nonstationary_currency_name = []
nonstationary_currency_data =pd.DataFrame([])

for i,filename in enumerate(filenames):
    path = 'F:\\testdata\\dukas\\' + filename
    price_buffer = pd.read_csv(path,header=None)
    close_price = price_buffer[4]
    temp_date = np.array(pd.to_datetime(price_buffer[0], format='%Y.%m.%d %H:%M'))
    close_price.index = temp_date
    test_data = close_price.loc['20140101':'20140201']
    result = sm2.adfuller(test_data)
    test_statistic = result[0]
    critical_value = result[4]['5%']
    if test_statistic > critical_value:
        nonstationary_currency_data[filename[0:6]] = test_data
        nonstationary_currency_name.append(filename[0:6])
    if i==1:
        break

total_currency_num = len(nonstationary_currency_name)
nonstationary_currency_data = nonstationary_currency_data.dropna(how='any')

coin_pairs = []

for i in range(0,total_currency_num-1):
    for j in range(i+1,total_currency_num):
        result = sm2.coint(nonstationary_currency_data[nonstationary_currency_name[i]], nonstationary_currency_data[nonstationary_currency_name[j]])
        #print nonstationary_currency_data[nonstationary_currency_name[i]]
        #print nonstationary_currency_data[nonstationary_currency_name[j]]
        test_statistic = result[0]
        critical_value = result[2][1]
        if test_statistic > critical_value:
            coin_pairs.append((nonstationary_currency_name[i],nonstationary_currency_name[j]))

print coin_pairs
'''