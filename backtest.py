import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import dates

def Calreturns(maindata,period=1):
    returns = pd.DataFrame()
    for fid in maindata.fid.unique():
        data = maindata[maindata.fid == fid]
        data.loc[:,fid] = (data['open'].shift(-(period+1))/data['open'].shift(-1)-1).values
        data.index = data['tradetime'].values
        returns = pd.concat([returns,data[fid]], axis = 1)
    return returns

if __name__ == "__main__":
    df = pd.read_file
    returns = Calreturns(df)
    result = signal*returns
    result['yield'] = result.apply(lambda x: x.sum(), axis = 1)
    result['netvalue'] = result['yield'].cumsum() + 1
    result['draw'] = np.maximum.accumulate(result['netvalue']) - result['netvalue']

    b = pd.to_datetime(result.index.values[0])
    e = pd.to_datetime(result.index.values[-1])
    years = (e-b).days/365
    turnover = np.mean(signal.diff().abs().mean())
    sharpe = round(result['yield'].mean()/result['yield'].std()*(np.sqrt(252)),4)
    anyield = (result['netvalue'].iloc[-1]-1-len(signal)*turnover*0.0008)/years
    maxdraw = np.max(result['draw'])
    index_j = np.argmax(np.maximum.accumulate(result['netvalue']) - result['netvalue'])  # 结束位置
    index_i = np.argmax(result['netvalue'][:index_j])  # 开始位置
    index_dif = (result.index[index_j] -  result.index[index_i]).days

    fig = plt.figure()
    axis = fig.add_subplot(2, 1, 1)
    axis.plot(result['netvalue'])
    axis.plot([result.index[index_i], result.index[index_j]], [result['netvalue'][index_i], result['netvalue'][index_j]], 'o', color="r", markersize=5)
    axis.set_title('anyield:'+str(round(anyield,3))+'_maxdraw:'+str(round(maxdraw,3))+'_sharpe:'+str(round(sharpe,3))+'_days:'+str(index_dif))
    axis.xaxis.set_major_formatter(dates.DateFormatter('%Y-%m')) 
    axis.xaxis.set_major_locator(dates.MonthLocator(interval = 3))
    for label in axis.get_xticklabels():
        label.set_rotation(30)
        label.set_horizontalalignment('right')
    plt.show()