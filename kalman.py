import threading
import time
import pandas as pd 
import numpy as np
from pykalman import KalmanFilter
import datetime
import pymysql

pd.set_option('expand_frame_repr', False)
pd.set_option('mode.chained_assignment', None)

def convert(x):
    return datetime.datetime.strptime(str(x),'%Y-%m-%d %H:%M:%S')

class download: 
    def __init__(self, coin_list, period):
        self.db = pymysql.connect(host="8.209.249.17", user="read_user", password='17NewBloc2020', database = 'min_data', autocommit =True)
        self.cursor = self.db.cursor()
        self.coin_list = coin_list
        self.period = period
        self.df = pd.DataFrame()

    def download(self):
        df = pd.DataFrame()
        for coin in self.coin_list:
            order = "SELECT tradetime, returns FROM `" + coin + "_" + self.period + "`"
            try:
                print("Downloading " + coin + " Data......")
                self.cursor.execute(order)
                myresult = self.cursor.fetchall()
                temp = pd.DataFrame(myresult)
                temp.columns = ["tradetime", coin]
                temp.loc[:, 'tradetime'] = temp['tradetime'].apply(lambda x: convert(x))
                df = pd.concat([df,temp[coin]], axis = 1)                    
            except Exception as e:
                    self.db.rollback()
                    print(e)
        df.index = temp['tradetime'].values
        self.cursor.close()
        self.df = df
        
    def calc_kalman_2(self, start):  
        df = self.df.dropna()
        name = df.columns.tolist()
        start_index = np.where(df.index.year == start)[0][0]
        delta = 0.1    #过程协方差矩阵的噪音
        transition_covariance = delta/(1-delta) * np.eye(2) #过程协方差矩阵(默认单位矩阵
        observation_matrices = np.vstack(
                            [np.ones(start_index), df.iloc[:start_index][name[0]]]
                        ).T[:,np.newaxis]    #观测矩阵
        # Shape = observation_matrices.shape
        # observation_matrices = observation_matrices.reshape(Shape[0],1,Shape[1])
        # print(observation_matrices)
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
        filter_mean, filter_cov = kf.filter(df.iloc[:start_index][name[1]].values)  #alpha:1 beta:0
        
        for i in range(start_index, len(df)):
            observation_matrix = np.array([[1,df[name[0]].values[i]]])
            observation = df[name[1]].values[i]

            #以上一个时刻的状态，状态的协方差以及当前的观测值，得到当前状态的估计
            next_filter_mean, next_filter_cov = kf.filter_update(
                    filtered_state_mean = filter_mean[-1],
                    filtered_state_covariance = filter_cov[-1],
                    observation = observation,
                    observation_matrix = observation_matrix)

            filter_mean = np.vstack((filter_mean, next_filter_mean))
            filter_cov = np.vstack((filter_cov, next_filter_cov.reshape(1,2,2)))

        #得到alpha和beta
        self.df['alpha'] = pd.Series(filter_mean[:,0], index = df.index) #截距
        self.df['beta'] = pd.Series(filter_mean[:,1], index = df.index) #斜率
        self.df.to_csv('D:\\New Bloc\\CTA\\研发\\测试数据\\kalman.csv')
        return self.df 
    
    def calc_kalman_3(self, start):  
        df = self.df.dropna()
        df.loc[:,'BTC'] = df['BTC'] - df['ETH']
        name = df.columns.tolist()
        n = len(name)
        # print(name)
        start_index = np.where(df.index.year == start)[0][0]
        delta = 0.01    #过程协方差矩阵的噪音
        transition_covariance = delta/(1-delta) * np.eye(n) #过程协方差矩阵(默认单位矩阵
        observation_matrices = np.vstack(
                            [np.ones(start_index), df.iloc[:start_index][name[1]], df.iloc[:start_index][name[0]]]
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
        filter_mean, filter_cov = kf.filter(df.iloc[:start_index][name[2]].values)  #alpha:1 beta:0
        
        for i in range(start_index, len(df)):
            observation_matrix = np.array([[1,df[name[1]].values[i],df[name[0]].values[i]]])
            observation = df[name[2]].values[i]

            #以上一个时刻的状态，状态的协方差以及当前的观测值，得到当前状态的估计
            next_filter_mean, next_filter_cov = kf.filter_update(
                    filtered_state_mean = filter_mean[-1],
                    filtered_state_covariance = filter_cov[-1],
                    observation = observation,
                    observation_matrix = observation_matrix)

            filter_mean = np.vstack((filter_mean, next_filter_mean))
            filter_cov = np.vstack((filter_cov, next_filter_cov.reshape(1,n,n)))

        #得到alpha和beta
        self.df['alpha'] = pd.Series(filter_mean[:,0], index = df.index) #截距
        self.df['beta_ETH'] = pd.Series(filter_mean[:,1], index = df.index) #斜率
        self.df['beta_E/B'] = pd.Series(filter_mean[:,2], index = df.index) #斜率
        self.df.to_csv('D:\\New Bloc\\CTA\\研发\\测试数据\\kalman.csv')
        return self.df 

if __name__ == "__main__":
    for c in  ['XRP', 'BNB', 'ADA', 'LTC', 'DOGE', 'LINK', 'DOT', 'ETC', 'EOS', 'BCH', 'UNI', 'TRX']:
        for period in ['4H','1D']:
            temp = ['BTC','ETH']
            temp.append(c)
            file = download(temp, period)
            file.download()
            file.calc_kalman_3(2020)
            print(c + '_' + period + '的Kalman滤波计算完成')
            break
