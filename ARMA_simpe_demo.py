# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 15:03:35 2019
@author: Administrator
example on datacamp,
only for recalling some functions.
"""
import pandas as pd
from statsmodels.tsa.arima_model import ARMA,ARIMA
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

#1.stocks.
df = pd.ExcelFile('./data/stocks_ARMA_simple_demo.xlsx').parse('Sheet1',header=None)
#391 rows.
df.iloc[0,0] = 0
df.columns = ['DATE','CLOSE']
df['DATE'] = pd.to_numeric(df['DATE'])
df = df.set_index('DATE')

print("缺失"+str(391-len(df))+"行.")           #缺失值行.
all_set = set(range(391))
existed_set = set(df.index)
print("缺失行:",all_set-existed_set)
    
df = df.reindex(range(391),method='ffill')      #补齐行.
df.index = pd.date_range(start='2017-09-01 9:30', end='2017-09-01 16:00',freq='1min')
df.plot(grid=True)

pct = df.pct_change()       #变化率.
pct = pct.dropna()

plot_acf(pct,lags=30)       #自相关.
plt.show()

ma1 = ARMA(pct,order=(0,1))
res = ma1.fit()
print(res.params)

print(('\n'*80))
#########################################################################
#2.temperature.
df = pd.ExcelFile('./data/temperature_ARMA_simple_demo.xlsx').parse('Sheet1',header=None)
df.columns = ['years','tavg']
df = df.set_index('years')

#自相关，和偏相关图不是截尾，也不是拖尾.p > 0.05.
df.index = pd.to_datetime(df.index,format='%Y')     #to_datetime
df.plot()
plt.show()

#0 hypo: random walk with drift.
result = adfuller(df['tavg'])
print('adf test:',result[1])

#p>0.5,非平稳.一阶差分.
chg = df.diff()
chg = chg.dropna()
fix,axes = plt.subplots(2,1)
plot_acf(chg,lags=20,ax=axes[0])
plot_pacf(chg,lags=20,ax=axes[1])
plt.show()

#AR(1).
mod_ar1 = ARMA(chg,order=(1,0))
res_ar1 = mod_ar1.fit()
print('AIC of AR(1):',res_ar1.aic)
#AR(2).
mod_ar2 = ARMA(chg,order=(2,0))
res_ar2 = mod_ar2.fit()
print('AIC of AR(2):',res_ar2.aic)
#ARMA(1,1).
mod_arma11 = ARMA(chg,order=(1,1))
res_arma11 = mod_arma11.fit()
print('AIC of ARMA(1,1):',res_arma11.aic)

#forcast. ARIMA.
mod = ARIMA(df,order=(1,1,1))          #order(p,d,q) or(p,q) arima定义.
res = mod.fit()
res.plot_predict(start='1872-01-01',end='2046-01-01')
plt.show()