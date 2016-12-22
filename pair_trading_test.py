import pandas as pd
import statsmodels.api as sm
import statsmodels.tsa.stattools as sm2
import numpy as np
import matplotlib.pyplot as plt
import datetime
import os
'''
price = pd.read_csv('F:\\usdx.csv')

temp_date = np.array(pd.to_datetime(price['Unnamed: 0'],format='%Y.%m.%d %H:%M'))

del price['Unnamed: 0']
price.index = temp_date
'''

'''
#aud vs usdx
#price_buffer = pd.read_csv('F:\\HistoryDatas -20160710\\AUDUSD_H4_UTC+0_00_130101_160709.csv',header=None)
price_buffer = pd.read_csv('F:\\AUDUSD_H4_UTC+0_00.csv',header=None)

temp_date = np.array(pd.to_datetime(price_buffer[0] + ' ' + price_buffer[1], format='%Y.%m.%d %H:%M'))

price_buffer.index = temp_date

price = pd.DataFrame(price_buffer[5])

price.columns = ['AUDUSD']
#price.columns = ['USDX','AUDUSD']

#nzd vs usdx
#price_buffer = pd.read_csv('F:\\HistoryDatas -20160710\\NZDUSD_H4_UTC+0_00_130101_160709.csv',header=None)
price_buffer = pd.read_csv('F:\\NZDUSD_H4_UTC+0_00_030804_160709.csv',header=None)

temp_date = np.array(pd.to_datetime(price_buffer[0] + ' ' + price_buffer[1], format='%Y.%m.%d %H:%M'))

price_buffer.index = temp_date

price = price.join(pd.DataFrame(price_buffer[5]),how='outer')

price.columns = ['AUDUSD','NZDUSD']

price = price.dropna(how='any')

#plt.plot(np.log(price['AUDUSD']),np.log(price['NZDUSD']),'r.')
#plt.show()

#(np.log(price[['AUDUSD','NZDUSD']])-np.log(price[['AUDUSD','NZDUSD']]).shift(1)).plot()
#plt.show()
'''
'''
sample = np.log(price.loc['20130101':'20160101'])

x = sm.add_constant(sample['NZDUSD'])
model = sm.OLS(sample['AUDUSD'],x)
result = model.fit()
intercept = result.params[0]
coefficient = result.params[1]
print result.summary()

test_data = np.log(price.loc['20160102':'20160708'])
result = test_data['AUDUSD'] - intercept - coefficient * test_data['NZDUSD']
result.plot()
plt.show()

portfolio = pd.Series([])
statistics = pd.DataFrame(columns=['mean','std'])
#portfolio_price = pd.Series([])
balance = 10000
equity = pd.Series(np.zeros(price['AUDUSD'].size),index=[price.index])

reg_params = pd.DataFrame(columns=['intercept','coefficient'])
adjust_times = 2 + (2016 - 2004) * 12 + 7
#adjust_times = 11
sigma_times = 0.75
sl_sigma_times = 10000

multiplier = 100000
total_trade_num = 0
winning_trade_num = 0

is_buy = False
is_sell = False
buy_lots = 0
sell_lots = 0

for i in range(0,adjust_times):
    start_y,start_m = divmod(8+i-1,12)
    end_y,end_m = divmod(11+i-1,12)
    test_y,test_m = divmod(12+i-1,12)
    start_date = datetime.datetime(2003+start_y,start_m+1,1)
    end_date = datetime.datetime(2003+end_y,end_m+1,1)
    test_date = datetime.datetime(2003+test_y,test_m+1,1)

    #print start_date,end_date,test_date

    sample = price.loc[start_date:end_date]
    sample = np.log(sample/sample.shift(1))
    sample = sample.dropna(how='any')
    x = sm.add_constant(sample['NZDUSD'])
    model = sm.OLS(sample['AUDUSD'], x)
    result = model.fit()
    intercept = result.params[0]
    coefficient = result.params[1]
    reg_params = reg_params.append(pd.DataFrame({'intercept':intercept,'coefficient':coefficient},index=[end_date]))

    historical_indicator = sample['AUDUSD'] - coefficient * sample['NZDUSD'] - intercept
    sigma = historical_indicator.std()
    mean = historical_indicator.mean()

    price_data = price.loc[end_date:test_date]
    test_data = np.log(price.loc[end_date:test_date])
    indicator = test_data['AUDUSD'] - coefficient * test_data['NZDUSD'] - intercept
    portfolio = portfolio.append(indicator)
    #statistics = statistics.append(pd.DataFrame({'mean':mean,'std':sigma},index=[indicator.index[0]]))

    for j in range(indicator.size):
        if is_buy == False:
            if not is_sell:
                equity[indicator.index[j]] = balance
            if (indicator[j] < mean - sigma_times * sigma) and (indicator[j] > mean - sl_sigma_times * sigma):
                is_buy = True
                buy_price = price_data['AUDUSD'][j] - coefficient * price_data['NZDUSD'][j]
                buy_lots = coefficient
                print 'buy', buy_price, indicator.index[j]
                total_trade_num += 1
        else:
            equity[indicator.index[j]] = balance + (price_data['AUDUSD'][j] - buy_lots * price_data['NZDUSD'][j] - buy_price) * multiplier

            if (indicator[j] > mean + sigma_times * sigma) or (indicator[j] < mean - sl_sigma_times * sigma):
                is_buy = False
                balance += (price_data['AUDUSD'][j] - buy_lots * price_data['NZDUSD'][j] - buy_price) * multiplier
                print 'close buy', price_data['AUDUSD'][j] - buy_lots * price_data['NZDUSD'][j], indicator.index[j]
                if price_data['AUDUSD'][j] - buy_lots * price_data['NZDUSD'][j] > buy_price:
                    winning_trade_num += 1
                    print 'win'
                else:
                    print 'lose'

        if is_sell == False:
            if not is_buy:
                equity[indicator.index[j]] = balance
            if (indicator[j] > mean + sigma_times * sigma) and (indicator[j] < mean + sl_sigma_times * sigma):
                is_sell = True
                sell_price = price_data['AUDUSD'][j] - coefficient * price_data['NZDUSD'][j]
                sell_lots = coefficient
                print 'sell', sell_price, indicator.index[j]
                total_trade_num += 1
        else:
            equity[indicator.index[j]] = balance + (sell_price + sell_lots * price_data['NZDUSD'][j] - price_data['AUDUSD'][j]) * multiplier

            if (indicator[j] < mean - sigma_times * sigma) or (indicator[j] > mean + sl_sigma_times * sigma):
                is_sell = False
                balance += (sell_price + sell_lots * price_data['NZDUSD'][j] - price_data['AUDUSD'][j]) * multiplier
                print 'close sell', price_data['AUDUSD'][j] - sell_lots * price_data['NZDUSD'][j], indicator.index[j]
                if price_data['AUDUSD'][j] - sell_lots * price_data['NZDUSD'][j] < sell_price:
                    winning_trade_num += 1
                    print 'win'
                else:
                    print 'lose'

    #portfolio_price = portfolio_price.append(price_data['AUDUSD'] - coefficient * price_data['NZDUSD'])

print balance,balance - total_trade_num * 15 * 2,total_trade_num,winning_trade_num
#portfolio = pd.DataFrame(portfolio)
#portfolio.columns = ['indicator']
#portfolio_price.name = 'actual'
#portfolio = portfolio.join(reg_params,how='outer')
#portfolio.ffill()
#print statistics
portfolio.plot()
plt.show()
'''
'''
sigma =  portfolio.std()
mean = portfolio.mean()
print sigma,mean
print reg_params

#trade.plot()
#plt.show()

''''''
lots = 1
contract_size = 10000
convert_to_usd = 1
multiplier = lots * contract_size * convert_to_usd
sigma_times = 0.75

current_month_index = 0
coefficient = reg_params['coefficient'][current_month_index]
intercept = reg_params['intercept'][current_month_index]

is_buy = False
is_sell = False
total_trade_num = 0
winning_trade_num = 0
profit = 0
for i in range(portfolio.size):
    if (current_month_index < reg_params.index.size - 1) and (portfolio.index[i] > reg_params.index[current_month_index+1]):
        #print reg_params.index[current_month_index+1]
        current_month_index = current_month_index + 1
        coefficient = reg_params['coefficient'][current_month_index]
        intercept = reg_params['intercept'][current_month_index]
    if is_buy == False:
        if portfolio[i] < mean - sigma_times * sigma:
            is_buy = True
            buy_price = price['AUDUSD'][i] - coefficient * price['NZDUSD'][i]
            total_trade_num += 1
            buy_weight = coefficient
    else:
        if portfolio[i] > mean + sigma_times * sigma:
            is_buy = False
            profit += (price['AUDUSD'][i] - buy_weight * price['NZDUSD'][i] - buy_price) * multiplier
            if price['AUDUSD'][i] - buy_weight * price['NZDUSD'][i] > buy_price:
                winning_trade_num += 1

    if is_sell == False:
        if portfolio[i] > mean + sigma_times * sigma:
            is_sell = True
            sell_price = price['AUDUSD'][i] - coefficient * price['NZDUSD'][i]
            total_trade_num += 1
            sell_weight = coefficient
    else:
        if portfolio[i] < mean - sigma_times * sigma:
            is_sell = False
            profit += (sell_price + sell_weight * price['NZDUSD'][i] - price['AUDUSD'][i]) * multiplier
            if price['AUDUSD'][i] - sell_weight * price['NZDUSD'][i] < sell_price:
                winning_trade_num += 1

print profit,profit - total_trade_num * 15 * (1+coefficient),total_trade_num,winning_trade_num
'''
'''
lag_price = price['AUDUSD'].shift(1)[1:]
delta_price = price['AUDUSD'][1:] - lag_price
lag_price = sm.add_constant(lag_price)
modle = sm.OLS(delta_price,lag_price)
result = modle.fit()

print result.summary()
#print result.params[0],result.params[1], -result.params[0]/result.params[1]
#print -np.log(2) / result.params[1]
print result.pvalues[1]


#result = sm2.adfuller(price['AUDUSD'],1,'ct',None,True,True)
#print result[4]['5%']
#print result
'''
#------------------------------------------------------------------------------------------------------------------------------------------------------------- new
filenames = os.listdir('F:\\testdata\\dukas\\')

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
#print filenames
