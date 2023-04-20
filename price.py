import numpy as np
import pandas as pd

# With the price for dates before t+1, calculate t+1, return predicted price from time 1 to t+1
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


# Only utilize the prices within a period
def EMA_price_new(temp_P, alpha, period):
    res = np.empty(len(temp_P))
    for k in range(period,len(temp_P)):
        P = temp_P[k-period:k+1]
        t = len(P)
        P_hat = []
        P_hat.append(P[0])
        for i in range(1, t):
            for j in range(1, t):
                p_hat = alpha * P[i-1] + (1-alpha) * P_hat[i-1]
                P_hat.append(p_hat)
        
        p_t1 = alpha * P[-1] + (1 - alpha) * P_hat[-1]
        res[k] = p_t1

    return res


def get_alpha(r, r_hat, r_prev, r_h_prev, alpha, gamma):
    if r > r_hat and r_prev > r_h_prev:
        alpha += gamma
    elif r <= r_hat and r_prev <= r_h_prev:
        alpha += gamma
    else:
        alpha -= gamma
    
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


# Provided by HBY
def AMA(price, N = 10, N1 = 2, N2 = 30):     #
    """
        N:  价格区间
        N1: 快线period
        N2: 慢线period
    """
    dif = abs(price-price.shift(N))    # 总体涨幅
    dif_sum = abs(price.diff()).rolling(N).sum()   # 总波动幅度
    roc = dif/dif_sum                                      # ROC:变动速率，
    fastest = 2 / (N1 + 1)                                 # 平滑系数：快系数
    slowest = 2 / (N2 + 1)                                 # 平滑系数：慢系数
    sm = roc*(fastest-slowest)+slowest                     # 平滑系数
    c = sm*sm
    ama = EMA(DMA(price,c),2)
    return ama

def DMA(price, A):  # 求S的动态移动平均，A作平滑因子,必须 0<A<1
    if isinstance(A, (int, float)):  
        return pd.Series(price).ewm(alpha=A, adjust=False).mean().values
    A = np.array(A)
    A[np.isnan(A)] = 1.0
    Y = np.zeros(len(price))
    Y[0] = price[0]
    for i in range(1, len(price)): 
        Y[i] = A[i] * price[i] + (1 - A[i]) * Y[i - 1]  # A支持序列
    return Y

def EMA(price, N): 
    return pd.Series(price).ewm(span=N, adjust=False).mean().values