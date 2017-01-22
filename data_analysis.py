import pandas as pd
import matplotlib.pyplot as plt

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
    path_event = 'F:\\testdata\\AUDCAD_event.csv'
    path_pricedata = 'F:\\testdata\\dukas\\AUDCAD_M1.csv'
    event_df = pd.read_csv(path_event,header=None,index_col=None)
    price_data_df = pd.read_csv(path_pricedata,header=None,index_col=None)

if __name__ == '__main__':
    #order_num_analysis()

    fake_order_num_analysis()
