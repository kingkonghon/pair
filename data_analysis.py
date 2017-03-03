import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import numpy as np

def order_num_analysis():
    path = 'F:\\testdata\\OrderNum.csv'
    data_df = pd.read_csv(path,header=None,index_col=0)

    data_df.columns = ['num','profit','hours','percent']

    max_num = data_df['num'].max()
    num_array = []
    profit_array = []
    hours_array = []
    percent_array = []

    for num in range(1,max_num+1):
        temp_df = data_df.loc[lambda df: df.num == num]
        if temp_df.size == 0:
            continue
        num_array.append(num)
        profit_array.append(temp_df['profit'].mean())
        hours_array.append(temp_df['hours'].mean())
        percent_array.append(temp_df['percent'].mean())

    dict = {'num':num_array,'profit':profit_array,'percent':percent_array}
    result = pd.DataFrame(dict)
    result['ideal'] = (result['num'] + 1) * result['num'] / 2. * 0.1 * 50
    result['diff'] = result['ideal'] - result['profit']
    #result['sl'] =  - result['ideal'].shift(1)
    #result['sl'][0] = 0
    #result.columns = ['num','profit','percent']
    print result
    '''
    plt.subplot(211)
    plt.plot(num_array,profit_array)
    plt.subplot(212)
    plt.plot(num_array,percent_array)
    plt.show()
    '''

def fake_order_num_analysis():
    path = 'F:\\testdata\\SLAna.csv'
    data_df = pd.read_csv(path,header=None,index_col=0)
    data_df.columns = ['fake','num','immediate','final']
    max_fake = data_df['fake'].max()

    #data_df = data_df.loc[lambda df: df.immediate < 0]
    #data_df = data_df.loc[lambda df: df.num == 7]

    fake_array = []
    num_array = []
    immediate_array = []
    final_array = []

    for fake in range(1,max_fake+1):
        temp_df = data_df.loc[lambda df:df.fake == fake]
        if temp_df.size == 0:
            continue
        fake_array.append(fake)
        num_array.append(temp_df['num'].mean())
        immediate_array.append(temp_df['immediate'].mean())
        final_array.append(temp_df['final'].mean())

    dict = {'fake':fake_array,'num':num_array,'immediate':immediate_array,'final':final_array}
    result_df = pd.DataFrame(dict)
    print result_df

def event_analysis():
    symbol = ['AUD','CAD']

    path_event = 'F:\\testdata\\AUDCAD_event.csv'
    path_pricedata = 'F:\\testdata\\dukas\\AUDCAD_M1.csv'
    event_df = pd.read_csv(path_event,header=0,index_col=None)
    price_data_df = pd.read_csv(path_pricedata,header=None,index_col=None)

    #event
    event_df.index = pd.to_datetime(event_df['Unnamed: 0'],format='%Y-%m-%d %H:%M:%S')
    del event_df['Unnamed: 0']

    criterion = event_df['currency'].map(lambda x:x in symbol)

    search_event = event_df[criterion]
    search_event = search_event.loc[lambda df:df.impact == 'high']

    search_event = search_event.dropna(how='any')
    search_event = search_event.loc['20140101':'20161030']

    for i in range(0,len(search_event.index)):
        try:
            search_event['actual'][i] = float(search_event['actual'][i].strip('%BK'))
            search_event['forecast'][i] = float(search_event['forecast'][i].strip('%BK'))
            search_event['previous'][i] = float(search_event['previous'][i].strip('%BK'))
        except AttributeError:
            search_event['actual'][i] = 0
            search_event['forecast'][i] = 0
            search_event['previous'][i] = 0

    #print search_event

    #price
    price_data_df.index = pd.to_datetime(price_data_df[0], format='%Y.%m.%d %H:%M')
    del price_data_df[0]
    del price_data_df[5]
    price_data_df.columns = ['open','high','low','close']

    open_price = []
    high_price = []
    low_price = []
    close_price = []
    delta_t = dt.timedelta(hours=2)

    for t in search_event.index:
        try:
            open_price.append(price_data_df.loc[t]['open'])
            high_price.append(price_data_df['high'].loc[t:t+delta_t].max())
            low_price.append(price_data_df['low'].loc[t:t+delta_t].min())
            close_price.append(price_data_df.loc[t+delta_t]['close'])
        except KeyError:
            break

    search_event['open'] = open_price
    search_event['high'] = high_price
    search_event['low'] = low_price
    search_event['close'] = close_price
    search_event['diff'] = search_event['close'] - search_event['open']

    # searching for event that cause 500 points move in price
    '''
    search_event_deep = search_event.loc[lambda df:abs(df.open - df.close) > 0.005,['actual','forecast','description','diff']]

    event_count = {}

    for i in range(0,search_event_deep.index.size):
        des = search_event_deep['description'][i]
        if des not in event_count.keys():
            event_count[des] = 1
        else:
            event_count[des] += 1

    print event_count

    for des in event_count.keys():
        #total_event_count[des] = search_event.loc[lambda df:df.description == des].index.size
        search_temp = search_event.loc[lambda df:df.description == des,['currency','actual','forecast','previous','description','diff']]
        total_count_for_single_event = search_temp.index.size
        #print  event_count[des],total_count_for_single_event,float(event_count[des]) / total_count_for_single_event
        if float(event_count[des]) / total_count_for_single_event > 0.1:
            search_temp = search_temp.loc[lambda df:df['actual'] != df['forecast']]
            total_count_for_single_event = search_temp.index.size
            print search_temp

            search_temp_deep = search_temp.loc[lambda df:df['currency'] == 'AUD']
            search_temp_deep = search_temp_deep.loc[lambda df:df.actual > df.forecast]
            search_temp_deep = search_temp_deep.loc[lambda df:df['diff'] > 0]
            print search_temp_deep

            correct_num = search_temp_deep.index.size

            search_temp_deep = search_temp.loc[lambda df: df['currency'] == 'AUD']
            search_temp_deep = search_temp_deep.loc[lambda df: df.actual < df.forecast]
            search_temp_deep = search_temp_deep.loc[lambda df: df['diff'] < 0]

            correct_num += search_temp_deep.index.size

            search_temp_deep = search_temp.loc[lambda df: df['currency'] == 'CAD']
            search_temp_deep = search_temp_deep.loc[lambda df: df.actual < df.forecast]
            search_temp_deep = search_temp_deep.loc[lambda df: df['diff'] > 0]

            correct_num += search_temp_deep.index.size

            search_temp_deep = search_temp.loc[lambda df: df['currency'] == 'CAD']
            search_temp_deep = search_temp_deep.loc[lambda df: df.actual > df.forecast]
            search_temp_deep = search_temp_deep.loc[lambda df: df['diff'] < 0]

            correct_num += search_temp_deep.index.size

            print des, float(correct_num) / total_count_for_single_event
    '''
    #calculate average move of price
    #'''

    #----------------------------------------
    # analysis effect when actual != forecast
    #----------------------------------------
    search_event_deep = search_event.loc[lambda df:df.actual != df.forecast]

    average_move = {}
    event_num = {}

    #result_df = pd.DataFrame(columns=['WinPercent', 'AveMove','70%diff','70%AveMove'])
    result_df = pd.DataFrame(columns=['WP(AUD)', 'AM(AUD)','Num(AUD)',
                                      'WP(CAD)', 'AM(CAD)','Num(CAD)',
                                      '70u(AUD)','70uAM(AUD)','70uNum(AUD)','70d(AUD)','70dAM(AUD)','70dNum(AUD)',
                                      '70u(CAD)','70uAM(CAD)','70uNum(CAD)','70d(CAD)','70dAM(CAD)','70dNum(CAD)'])


    # find all event name, and absolute event impact
    for i in range(0,search_event_deep.index.size):
        des = search_event_deep['description'][i]
        if des not in average_move.keys():
            average_move[des] = abs(search_event_deep['diff'][i])
            event_num[des] = 1
        else:
            average_move[des] += abs(search_event_deep['diff'][i])
            event_num[des] += 1

    for des in average_move.keys():
        average_move[des] = average_move[des] / event_num[des]

    print sorted(average_move.items(),key=lambda d:d[1],reverse=True)
    #average_move.pop('Unemployment Rate')
    #average_move.pop('Employment Change')

    temp_buff = {}


    # ordinary event (positively correlated with AUDCAD)
    for des in average_move.keys():
        if des == 'Unemployment Rate' or des == 'Employment Change':
            continue

        # filter for each event
        search_temp = search_event_deep.loc[lambda df:df.description == des]

        #seperate different currency

        #if it is AUD
        search_temp_aud = search_temp.loc[lambda df:df.currency == 'AUD']

        if not search_temp_aud.empty:
            search_temp_deep = search_temp_aud.loc[lambda df:df.actual > df.forecast]
            u_correct_num = search_temp_deep.loc[lambda df:df['diff'] > 0].index.size
            u_sum_move = search_temp_deep['diff'].sum()
            u_total_num = search_temp_deep.index.size

            search_temp_deep = search_temp_aud.loc[lambda df: df.actual < df.forecast]
            d_correct_num = search_temp_deep.loc[lambda df: df['diff'] < 0].index.size
            d_sum_move = (-search_temp_deep['diff']).sum()
            d_total_num = search_temp_deep.index.size

            temp_buff['WP(AUD)'] = float(u_correct_num + d_correct_num) / search_temp_aud.index.size
            temp_buff['AM(AUD)'] = (u_sum_move + d_sum_move) / search_temp_aud.index.size
            temp_buff['Num(AUD)'] = search_temp_aud.index.size

            #find how far up is the difference between actual and forecast will have a 70% winning percent
            u_diff = 0

            d_diff = 0

            #AUD up suprise
            while float(u_correct_num)/u_total_num < 0.7:
                u_diff += 0.1

                search_temp_deep = search_temp_aud.loc[lambda df: df.actual > df.forecast + u_diff]
                #cannot find
                if search_temp_deep.empty:
                    temp_buff['70u(AUD)'] = None
                    temp_buff['70uAM(AUD)'] = None
                    temp_buff['70uNum(AUD)'] = None
                    break

                u_correct_num = search_temp_deep.loc[lambda df: df['diff'] > 0].index.size
                u_sum_move = search_temp_deep['diff'].sum()
                u_total_num = search_temp_deep.index.size
            else:
                temp_buff['70u(AUD)'] = u_diff
                temp_buff['70uAM(AUD)'] = u_sum_move / u_total_num
                temp_buff['70uNum(AUD)'] = u_total_num

            #AUD down suprise
            while float(d_correct_num)/d_total_num < 0.7:
                d_diff += 0.1

                search_temp_deep = search_temp_aud.loc[lambda df: df.actual < df.forecast - d_diff]
                #cannot find
                if search_temp_deep.empty:
                    temp_buff['70d(AUD)'] = None
                    temp_buff['70dAM(AUD)'] = None
                    temp_buff['70dNum(AUD)'] = None
                    break

                d_correct_num = search_temp_deep.loc[lambda df: df['diff'] < 0].index.size
                d_sum_move = (-search_temp_deep['diff']).sum()
                d_total_num = search_temp_deep.index.size
            else:
                temp_buff['70d(AUD)'] = d_diff
                temp_buff['70dAM(AUD)'] = d_sum_move / d_total_num
                temp_buff['70dNum(AUD)'] = d_total_num

        else:
            temp_buff['WP(AUD)'] = 0
            temp_buff['AM(AUD)'] = 0
            temp_buff['Num(AUD)'] = 0

            temp_buff['70u(AUD)'] = None
            temp_buff['70uAM(AUD)'] = None
            temp_buff['70uNum(AUD)'] = None
            temp_buff['70d(AUD)'] = None
            temp_buff['70dAM(AUD)'] = None
            temp_buff['70dNum(AUD)'] = None


        #CAD
        search_temp_cad = search_temp.loc[lambda df: df.currency == 'CAD']

        if not search_temp_cad.empty:
            search_temp_deep = search_temp_cad.loc[lambda df: df.actual > df.forecast]
            u_correct_num = search_temp_deep.loc[lambda df: df['diff'] < 0].index.size
            u_sum_move = (-search_temp_deep['diff']).sum()
            u_total_num = search_temp_deep.index.size

            search_temp_deep = search_temp_cad.loc[lambda df: df.actual < df.forecast]
            d_correct_num = search_temp_deep.loc[lambda df: df['diff'] > 0].index.size
            d_sum_move = search_temp_deep['diff'].sum()
            d_total_num = search_temp_deep.index.size

            temp_buff['WP(CAD)'] = float(u_correct_num + d_correct_num) / search_temp_cad.index.size
            temp_buff['AM(CAD)'] = (u_sum_move + d_sum_move) / search_temp_cad.index.size
            temp_buff['Num(CAD)'] = search_temp_cad.index.size

            # find how far down is the difference between actual and forecast will have a 70% winning percent
            u_diff = 0
            d_diff = 0

            #CAD up suprise
            while float(u_correct_num) / u_total_num < 0.7:
                u_diff += 0.1

                search_temp_deep = search_temp_cad.loc[lambda df: df.actual > df.forecast + u_diff]
                #cannot find
                if search_temp_deep.empty:
                    temp_buff['70u(CAD)'] = None
                    temp_buff['70uAM(CAD)'] = None
                    temp_buff['70uNum(CAD)'] = None
                    break

                u_correct_num = search_temp_deep.loc[lambda df: df['diff'] < 0].index.size
                u_sum_move = (-search_temp_deep['diff']).sum()
                u_total_num = search_temp_deep.index.size
            else:
                temp_buff['70u(CAD)'] = u_diff
                temp_buff['70uAM(CAD)'] = u_sum_move / u_total_num
                temp_buff['70uNum(CAD)'] = u_total_num

            #CAD down suprise
            while float(d_correct_num) / d_total_num < 0.7:
                d_diff += 0.1

                search_temp_deep = search_temp_cad.loc[lambda df: df.actual < df.forecast - d_diff]
                #cannot find
                if search_temp_deep.empty:
                    temp_buff['70d(CAD)'] = None
                    temp_buff['70dAM(CAD)'] = None
                    temp_buff['70dNum(CAD)'] = None
                    break

                d_correct_num = search_temp_deep.loc[lambda df: df['diff'] > 0].index.size
                d_sum_move = search_temp_deep['diff'].sum()
                d_total_num = search_temp_deep.index.size
            else:
                temp_buff['70d(CAD)'] = d_diff
                temp_buff['70dAM(CAD)'] = d_sum_move / d_total_num
                temp_buff['70dNum(CAD)'] = d_total_num

        else:
            temp_buff['WP(CAD)'] = 0
            temp_buff['AM(CAD)'] = 0
            temp_buff['Num(CAD)'] = 0

            temp_buff['70u(CAD)'] = None
            temp_buff['70uAM(CAD)'] = None
            temp_buff['70uNum(CAD)'] = None

            temp_buff['70d(CAD)'] = None
            temp_buff['70dAM(CAD)'] = None
            temp_buff['70dNum(CAD)'] = None

        #print aud_correct_num,aud_sum_move,cad_correct_num,cad_sum_move

        #print temp_buff
        #print temp_buff

        result_df.loc[des] = temp_buff

        #print des,float(correct_num)/search_temp.index.size, \
        #    sum_move / search_temp.index.size,\
        #    search_temp.index.size, search_temp_aud.index.size + search_temp_cad.index.size

    # unemployment rate
    search_temp = search_event_deep.loc[lambda df:df.description == 'Unemployment Rate']

    #AUD
    search_temp_aud = search_temp.loc[lambda df: df.currency == 'AUD']
    if search_temp_aud.empty:
        search_temp_deep = search_temp_aud.loc[lambda df: df.actual < df.forecast]
        d_correct_num = search_temp_deep.loc[lambda df: df['diff'] > 0].index.size
        d_sum_move = search_temp_deep['diff'].sum()
        d_total_num = search_temp_deep.index.size

        search_temp_deep = search_temp_aud.loc[lambda df: df.actual > df.forecast]
        u_correct_num = search_temp_deep.loc[lambda df: df['diff'] < 0].index.size
        u_sum_move = (-search_temp_deep['diff']).sum()
        u_total_num = search_temp_deep.index.size

        temp_buff['WP(AUD)'] = float(d_correct_num + u_correct_num) / (d_total_num + u_total_num)
        temp_buff['AM(AUD'] = (d_sum_move + u_sum_move) / (d_total_num + u_total_num)
        temp_buff['Num(AUD)'] = d_total_num + u_total_num

        #find how far the difference would have 70% winning percentage
        u_diff = 0
        d_diff = 0

        #down suprise (AUDCAD price rise)
        while float(d_correct_num) / d_total_num < 0.7:
            d_diff += 0.1

            search_temp_deep = search_temp_aud.loc[lambda df:df.actual < df.forecast - d_diff]
            #cannot find
            if search_temp_deep.empty:
                temp_buff['70d(AUD)'] = None
                temp_buff['70dAM(AUD)'] = None
                temp_buff['70dNum(AUD)'] = None
                break

            d_correct_num = search_temp_deep.loc[lambda df: df['diff'] > 0].index.size
            d_sum_move = search_temp_deep['diff'].sum()
            d_total_num = search_temp_deep.index.size
        else:
            temp_buff['70d(AUD'] = d_diff
            temp_buff['70dAM(AUD)'] = d_sum_move / d_total_num
            temp_buff['70dNum(AUD)'] = d_total_num

        #up suprise (AUDCAD price fall)
        while float(u_correct_num) / u_total_num < 0.7:
            u_diff += 0.1

            search_temp_deep = search_temp_aud.loc[lambda df:df.actual > df.forecast + u_diff]
            #cannot find
            if search_temp_deep.empty:
                temp_buff['70u(AUD)'] = None
                temp_buff['70uAM(AUD)'] = None
                temp_buff['70uNum(AUD)'] = None
                break

            u_correct_num = search_temp_deep.loc[lambda df: df['diff'] < 0].index.size
            u_sum_move = (-search_temp_deep['diff']).sum()
            u_total_num = search_temp_deep.index.size
        else:
            temp_buff['70u(AUD'] = u_diff
            temp_buff['70uAM(AUD)'] = u_sum_move / u_total_num
            temp_buff['70uNum(AUD)'] = u_total_num

    else:
        temp_buff['WP(AUD)'] = 0
        temp_buff['AM(AUD)'] = 0
        temp_buff['Num(AUD)']

        temp_buff['70u(AUD)'] = None
        temp_buff['70uAM(AUD)'] = None
        temp_buff['70uNum(AUD)'] = None

        temp_buff['70d(AUD)'] = None
        temp_buff['70dAM(AUD)'] = None
        temp_buff['70dNum(AUD)'] = None


    #CAD
    search_temp_cad = search_temp.loc[lambda df: df.currency == 'CAD']
    if search_temp_cad.empty:
        search_temp_deep = search_temp_cad.loc[lambda df: df.actual < df.forecast]
        correct_num += search_temp_deep.loc[lambda df: df['diff'] < 0].index.size
        sum_move += (-search_temp_deep['diff']).sum()

        search_temp_deep = search_temp_cad.loc[lambda df: df.actual > df.forecast]
        correct_num += search_temp_deep.loc[lambda df: df['diff'] > 0].index.size
        sum_move += search_temp_deep['diff'].sum()


    else:
        temp_buff['WP(CAD)'] = 0
        temp_buff['AM(CAD)'] = 0
        temp_buff['70u(CAD)'] = None
        temp_buff['70uAM(CAD)'] = None
        temp_buff['70d(CAD)'] = None
        temp_buff['70dAM(CAD)'] = None

    temp_buff['WinPercent'] = float(correct_num) / search_temp.index.size
    temp_buff['AveMove'] = sum_move / search_temp.index.size

    result_df.loc['Unemployment Rate'] = temp_buff

    print 'Unemployment Rate', float(correct_num) / search_temp.index.size, \
        sum_move / search_temp.index.size, \
        search_temp.index.size, search_temp_aud.index.size + search_temp_cad.index.size


    #employment change
    search_temp = search_event_deep.loc[lambda df: df.description == 'Employment Change']
    search_temp = search_temp.loc[lambda df: np.sign(df.actual) != np.sign(df.forecast)]

    #print search_temp

    search_temp_aud = search_temp.loc[lambda df: df.currency == 'AUD']
    search_temp_deep = search_temp_aud.loc[lambda df: df.forecast < 0]
    search_temp_deep = search_temp_deep.loc[lambda df: df.actual > 0]
    correct_num = search_temp_deep.loc[lambda df: df['diff'] > 0].index.size
    sum_move = search_temp_deep['diff'].sum()

    search_temp_deep = search_temp_aud.loc[lambda df: df.forecast > 0]
    search_temp_deep = search_temp_deep.loc[lambda df: df.actual < 0]
    correct_num += search_temp_deep.loc[lambda df: df['diff'] < 0].index.size
    sum_move += (-search_temp_deep['diff']).sum()

    search_temp_cad = search_temp.loc[lambda df: df.currency == 'CAD']
    search_temp_deep = search_temp_cad.loc[lambda df: df.forecast < 0]
    search_temp_deep = search_temp_deep.loc[lambda df: df.actual > 0]
    correct_num += search_temp_deep.loc[lambda df: df['diff'] < 0].index.size
    sum_move += (-search_temp_deep['diff']).sum()

    search_temp_deep = search_temp_cad.loc[lambda df: df.forecast > 0]
    search_temp_deep = search_temp_deep.loc[lambda df: df.actual < 0]
    correct_num += search_temp_deep.loc[lambda df: df['diff'] > 0].index.size
    sum_move += search_temp_deep['diff'].sum()

    temp_buff['WinPercent'] = float(correct_num) / search_temp.index.size
    temp_buff['AveMove'] = sum_move / search_temp.index.size

    result_df.loc['Employment Change'] = temp_buff

    print 'Employment Change', float(correct_num) / search_temp.index.size, \
        sum_move / search_temp.index.size, \
        search_temp.index.size, search_temp_aud.index.size + search_temp_cad.index.size


    #'''
    print result_df

    result_df.to_csv('E:\\forex_factory\\audcad_event_analysis(a!=f).csv',header=True, index=True, mode='w')

    # ----------------------------------------
    # analysis effect when actual == forecast, but actual != previous
    # ----------------------------------------
    search_event_deep = search_event.loc[lambda df: df.actual == df.forecast]
    search_event_deep = search_event_deep.loc[lambda df:df.actual != df.previous]

    average_move = {}
    event_num = {}

    # result_df = pd.DataFrame(columns=['WinPercent', 'AveMove','70%diff','70%AveMove'])
    result_df = pd.DataFrame(columns=['WP(AUD)', 'AM(AUD)', 'WP(CAD)', 'AM(CAD)', '70u(AUD)', '70uAM(AUD)', '70d(AUD)',
                                      '70dAM(AUD)', '70u(CAD)', '70uAM(CAD)', '70d(CAD)', '70dAM(CAD)', ])

    # find all event name
    for i in range(0, search_event_deep.index.size):
        des = search_event_deep['description'][i]
        if des not in average_move.keys():
            average_move[des] = abs(search_event_deep['diff'][i])
            event_num[des] = 1
        else:
            average_move[des] += abs(search_event_deep['diff'][i])
            event_num[des] += 1

    for des in average_move.keys():
        average_move[des] = average_move[des] / event_num[des]

    print sorted(average_move.items(), key=lambda d: d[1], reverse=True)
    # average_move.pop('Unemployment Rate')
    # average_move.pop('Employment Change')

    temp_buff = {}


if __name__ == '__main__':
    #order_num_analysis()

    #fake_order_num_analysis()

    event_analysis()
