import pandas as pd 
import numpy as np
from pykalman import KalmanFilter
import os 
pd.set_option('mode.chained_assignment', None)

"""
    columns[0]是实际值 columns后面是观测值
"""
def calc_kalman(data,start):  
    data.dropna(inplace = True)
    data.reset_index(drop = True, inplace = True)
    df = data.copy()
    name = df.columns.tolist()
    start_index = int(len(df)*start)
    delta = 0.1    #过程协方差矩阵的噪音
    transition_covariance = delta/(1-delta) * np.eye(2) #过程协方差矩阵(默认单位矩阵
    observation_matrices = np.vstack(
                        [np.ones(start_index), df.iloc[:start_index][name[1]]]
                    ).T[:,np.newaxis]    #观测矩阵

    kf = KalmanFilter(
                    n_dim_obs = 1,  #the dimensionality of the observation space
                    n_dim_state = 2,    #the dimensionality of the state space
                    initial_state_mean = np.zeros(2),   #mean of initial state distribution
                    initial_state_covariance = np.ones((2,2)),  #covariance of initial state distribution
                    transition_matrices = np.eye(2), #转移矩阵为单位阵
                    transition_covariance = transition_covariance,
                    observation_matrices = observation_matrices,
                    observation_covariance = 0.005
                    )
    filter_mean, filter_cov = kf.filter(df.iloc[:start_index][name[0]].values)  #alpha:1 beta:0
    
    for i in range(start_index, len(df)):
        observation_matrix = np.array([[1,df[name[1]].values[i]]])
        observation = df[name[0]].values[i]

        #以上一个时刻的状态，状态的协方差以及当前的观测值，得到当前状态的估计
        next_filter_mean, next_filter_cov = kf.filter_update(
                filtered_state_mean = filter_mean[-1],
                filtered_state_covariance = filter_cov[-1],
                observation = observation,
                observation_matrix = observation_matrix)

        filter_mean = np.vstack((filter_mean, next_filter_mean))
        filter_cov = np.vstack((filter_cov, next_filter_cov.reshape(1,2,2)))

    #得到alpha和beta
    df['alpha'] = pd.Series(filter_mean[:,0], index = df.index) #截距
    df['beta'] = pd.Series(filter_mean[:,1], index = df.index) #斜率
    df['predict'] = df['alpha'] + df['beta']*df[name[0]]
    df.loc[:start_index,'predict'] = 0
    # df.to_csv('test.csv')
    return df 

def calc_kalman_multi(data,start):  
    data.dropna(inplace = True)
    data.reset_index(drop = True, inplace = True)
    df = data.copy()
    name = df.columns.tolist()
    start_index = int(len(df)*start)
    n = len(name)
    delta = 0.01    #过程协方差矩阵的噪音
    transition_covariance = delta/(1-delta) * np.eye(n) #过程协方差矩阵(默认单位矩阵
    observation_matrices = np.vstack(
                        [np.ones(start_index), df.iloc[:start_index][name[1]], df.iloc[:start_index][name[2]]]
                    ).T[:,np.newaxis]    #观测矩阵
    # Shape = observation_matrices.shape
    # observation_matrices = observation_matrices.reshape(Shape[0],1,Shape[1])
    # print(observation_matrices)
    kf = KalmanFilter(
                    n_dim_obs = 1,  #the dimensionality of the observation space
                    n_dim_state = n,    #the dimensionality of the state space
                    initial_state_mean = np.zeros(n),   #mean of initial state distribution
                    initial_state_covariance = np.ones((n,n)),  #covariance of initial state distribution
                    transition_matrices = np.eye(n), #转移矩阵为单位阵
                    transition_covariance = transition_covariance,
                    observation_matrices = observation_matrices,
                    observation_covariance = 0.005
                    )
    filter_mean, filter_cov = kf.filter(df.iloc[:start_index][name[0]].values)  #alpha:1 beta:0
    
    for i in range(start_index, len(df)):
        observation_matrix = np.array([[1,df[name[1]].values[i],df[name[2]].values[i]]])
        observation = df[name[0]].values[i]

        #以上一个时刻的状态，状态的协方差以及当前的观测值，得到当前状态的估计
        next_filter_mean, next_filter_cov = kf.filter_update(
                filtered_state_mean = filter_mean[-1],
                filtered_state_covariance = filter_cov[-1],
                observation = observation,
                observation_matrix = observation_matrix)

        filter_mean = np.vstack((filter_mean, next_filter_mean))
        filter_cov = np.vstack((filter_cov, next_filter_cov.reshape(1,n,n)))

    #得到alpha和beta
    df['alpha'] = pd.Series(filter_mean[:,0], index = df.index) #截距
    df['beta_1'] = pd.Series(filter_mean[:,1], index = df.index) #斜率
    df['beta_2'] = pd.Series(filter_mean[:,2], index = df.index) #斜率
    df['predict'] = df['alpha'] + df['beta_1']*df[name[1]] + df['beta_2']*df[name[2]]
    df.loc[:start_index,'predict'] = 0
    # df.to_csv('test.csv')
    return df 

if __name__ == "__main__":
    os.chdir('D:\Memory\Cornell\ORIE 5370')
    maindata = pd.read_hdf('D:\\Trading\\多因子\\maindata_1D.h5')
    df = maindata[maindata['fid'] == 'BTC']
    data = pd.DataFrame({'target':df['open'].shift(-2)/df['open'].shift(-1)-1,'return':df['close']/df['close'].shift()-1,'return_vol':df['volume']/df['volume'].shift()-1})
    res = calc_kalman(data,0.4)
    res = res[res['predict'] != 0]
    accuracy = np.sign(res['predict']) - np.sign(res['target'])
    accuracy = len(accuracy[accuracy == 0])/len(accuracy)
    print('精确度为' + str(round(accuracy,3)) + '%')
    res = calc_kalman_multi(data,0.4)
    res = res[res['predict'] != 0]
    accuracy = np.sign(res['predict']) - np.sign(res['target'])
    accuracy = len(accuracy[accuracy == 0])/len(accuracy)
    print('精确度为' + str(round(accuracy,3)) + '%')