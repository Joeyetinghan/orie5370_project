import pandas as pd 
import numpy as np
from pykalman import KalmanFilter
import os 
import numba as nb
pd.set_option('mode.chained_assignment', None)

def calc_kalman(data,start):  
    data.dropna(inplace = True)
    indexs = data.index
    data.reset_index(drop = True, inplace = True)
    df = data.copy()
    name = df.columns.tolist()
    start_index = int(len(df)*start)
    delta = 0.1
    transition_covariance = delta/(1-delta) * np.eye(2)
    observation_matrices = np.vstack(
                        [np.ones(start_index), df.iloc[:start_index][name[1]]]
                    ).T[:,np.newaxis]

    kf = KalmanFilter(
                    n_dim_obs = 1,  #the dimensionality of the observation space
                    n_dim_state = 2,    #the dimensionality of the state space
                    initial_state_mean = np.zeros(2),   #mean of initial state distribution
                    initial_state_covariance = np.ones((2,2)),  #covariance of initial state distribution
                    transition_matrices = np.eye(2),
                    transition_covariance = transition_covariance,
                    observation_matrices = observation_matrices,
                    observation_covariance = 0.005
                    )
    filter_mean, filter_cov = kf.filter(df.iloc[:start_index][name[0]].values)  #alpha:1 beta:0
    
    for i in range(start_index, len(df)):
        observation_matrix = np.array([[1,df[name[1]].values[i]]])
        observation = df[name[0]].values[i]

        next_filter_mean, next_filter_cov = kf.filter_update(
                filtered_state_mean = filter_mean[-1],
                filtered_state_covariance = filter_cov[-1],
                observation = observation,
                observation_matrix = observation_matrix)

        filter_mean = np.vstack((filter_mean, next_filter_mean))
        filter_cov = np.vstack((filter_cov, next_filter_cov.reshape(1,2,2)))

    #get alpha和beta
    df['alpha'] = pd.Series(filter_mean[:,0], index = df.index) #slope
    df['beta'] = pd.Series(filter_mean[:,1], index = df.index) #intercept
    df['predict'] = df['alpha'].shift() + df['beta'].shift()*df[name[0]]
    df.loc[:start_index,'predict'] = np.nan
    df.index = indexs
    return df['predict']

def calc_kalman_multi(data,start):  
    data.dropna(inplace = True)
    indexs = data.index
    data.reset_index(drop = True, inplace = True)
    df = data.copy()
    name = df.columns.tolist()
    start_index = int(len(df)*start)
    n = len(name)
    delta = 0.01 
    transition_covariance = delta/(1-delta) * np.eye(n)
    observation_matrices = np.vstack(
                        [np.ones(start_index), df.iloc[:start_index][name[1]], df.iloc[:start_index][name[2]]]
                    ).T[:,np.newaxis]
    # Shape = observation_matrices.shape
    # observation_matrices = observation_matrices.reshape(Shape[0],1,Shape[1])
    # print(observation_matrices)
    kf = KalmanFilter(
                    n_dim_obs = 1,  #the dimensionality of the observation space
                    n_dim_state = n,    #the dimensionality of the state space
                    initial_state_mean = np.zeros(n),   #mean of initial state distribution
                    initial_state_covariance = np.ones((n,n)),  #covariance of initial state distribution
                    transition_matrices = np.eye(n), #identity matrix
                    transition_covariance = transition_covariance,
                    observation_matrices = observation_matrices,
                    observation_covariance = 0.005
                    )
    filter_mean, filter_cov = kf.filter(df.iloc[:start_index][name[0]].values)  #alpha:1 beta:0
    
    for i in range(start_index, len(df)):
        observation_matrix = np.array([[1,df[name[1]].values[i],df[name[2]].values[i]]])
        observation = df[name[0]].values[i]

        next_filter_mean, next_filter_cov = kf.filter_update(
                filtered_state_mean = filter_mean[-1],
                filtered_state_covariance = filter_cov[-1],
                observation = observation,
                observation_matrix = observation_matrix)

        filter_mean = np.vstack((filter_mean, next_filter_mean))
        filter_cov = np.vstack((filter_cov, next_filter_cov.reshape(1,n,n)))

    #get alpha和beta
    df['alpha'] = pd.Series(filter_mean[:,0], index = df.index)
    df['beta_1'] = pd.Series(filter_mean[:,1], index = df.index)
    df['beta_2'] = pd.Series(filter_mean[:,2], index = df.index) 
    df['predict'] = df['alpha'].shift() + df['beta_1'].shift()*df[name[1]] + df['beta_2'].shift()*df[name[2]]
    df.loc[:start_index,'predict'] = np.nan
    df.index = indexs
    return df['predict'] 

def rolling_kalman(data):  
    data.reset_index(drop = True, inplace = True)
    df = data.copy()
    df = pd.DataFrame({'target':df.shift(-1),'return':df})
    df.dropna(inplace = True)
    df.reset_index(drop = True, inplace = True)
    name = df.columns.tolist()
    delta = 0.1   
    transition_covariance = delta/(1-delta) * np.eye(2) #过程协方差矩阵(默认单位矩阵
    observation_matrices = np.vstack(
                        [np.ones(len(df)), df[name[1]]]
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
    filter_mean, filter_cov = kf.filter(df[name[0]].values)  #alpha:1 beta:0

    #得到alpha和beta
    df['alpha'] = pd.Series(filter_mean[:,0], index = df.index) #截距
    df['beta'] = pd.Series(filter_mean[:,1], index = df.index) #斜率
    df['predict'] = df['alpha'].shift() + df['beta'].shift()*df[name[0]]
    return df['predict'].iloc[-1]


@nb.njit()
def REGbeta_ts(x,yy,n):
    result = np.full(len(x),np.nan)
    x0 = x[~np.isnan(x)]
    y0 = yy[~np.isnan(yy)]
    for i in range(n+(len(x)-np.minimum(len(x0),len(y0))),len(x)+1):
        X = x[i-n:i]
        y = yy[i-n:i]
        X, y = X.astype(np.float64), y.astype(np.float64)
        X = X.reshape(X.shape[0], -1)
        A = np.vstack((X.T, np.ones((1, (len(X))), dtype=np.float64))).T
        res = np.linalg.lstsq(A, y)[0]
        result[i-1] = res[0]*10e5
    return result/10e5

@nb.njit()
def REGalpha_ts(x,yy,n):
    result = np.full(len(x),np.nan)
    x0 = x[~np.isnan(x)]
    y0 = yy[~np.isnan(yy)]
    for i in range(n+(len(x)-np.minimum(len(x0),len(y0))),len(x)+1):
        X = x[i-n:i]
        y = yy[i-n:i]
        X, y = X.astype(np.float64), y.astype(np.float64)
        X = X.reshape(X.shape[0], -1)
        A = np.vstack((X.T, np.ones((1, (len(X))), dtype=np.float64))).T
        res = np.linalg.lstsq(A, y)[0]
        result[i-1] = res[1]*10e5
    return result/10e5

def SMA_price(P, w):
    t = len(P)
    P_hat = []
    for i in range(t):
        if i>=w:
            p = np.mean(P[i-w:i])  # Predicted stock price
        else:
            p = P[i]
        P_hat.append(p)
    P_hat.append(np.mean(P[t-w:]))
    
    return P_hat[1:]

def EMA_price(P, alpha):
    t = len(P)
    P_hat = []
    P_hat.append(P[0])
    for i in range(1, t):
        p_hat = alpha * P[i-1] + (1-alpha) * P_hat[i-1]
        P_hat.append(p_hat)
    
    p_t1 = alpha * P[-1] + (1 - alpha) * P_hat[-1]
    P_hat.append(p_t1)

    return P_hat[1:]

# AOLMA
def get_alpha(r, r_hat, r_prev, r_h_prev, alpha, gamma):
    if r > r_hat and r_prev > r_h_prev:
        alpha += gamma
    elif r <= r_hat and r_prev <= r_h_prev:
        alpha += gamma
    else:
        alpha -= gamma
        
    if alpha>1 or alpha<0:
        alpha=0.5
    
    return alpha

def AOLMA(P, gamma):
    R, R_hat, P_hat, A = [1], [1], [], [0.5, 0.5]
    # R = [P[i + 1] / P[i] for i in range(len(P)-1)]
    P_hat.append(P[0])
        
    for i in range(len(P)-1):
        r = P[i+1] / P[i]
        phat = A[i + 1] * P[i] + (1 - A[i + 1]) * P_hat[i]
        rhat =  A[i + 1] + (1 - A[i + 1]) * R_hat[i] / R[i]
        R.append(r)
        P_hat.append(phat)
        R_hat.append(rhat)
        a = get_alpha(R[i+1], R_hat[i+1], R[i], R_hat[i], A[i+1], gamma)
        A.append(a)
    
    phat = A[-1] * P[-1] + (1 - A[-1]) * P_hat[-1]
    rhat = A[-1] + (1 - A[-1]) * R_hat[-1] / R[-1]
    P_hat.append(phat)
    R_hat.append(rhat)
    
    return P_hat[1:]