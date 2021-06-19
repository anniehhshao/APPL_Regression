# -*- coding: utf-8 -*-
"""
Created on Sat Jun  5 15:00:27 2021

@author: user
"""

import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager
#a = sorted([f.name for f in matplotlib.font_manager.fontManager.ttflist])
 
#for i in a:
    #print(i)
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']

data=pd.read_csv("C:/Users/user/Downloads/AAPL.csv")
data['y']=(data['Close']-data['Yesterday_Close'])/data['Yesterday_Close']
#data['y']=data['y']+data['Yesterday_Close']
#print(data)

data_y=data['y'].values
data_y=np.append(data_y,[0])
i=data.shape[0]
dt_new_y = np.zeros(i)
dt_new_y[0 : i] = data_y.flatten()[1 : i+1] 
data['y']=pd.Series(dt_new_y)
#print(data)
data=data.drop(data.shape[0]-1,axis=0)
#data.set_index("Date" , inplace=True)
data["Date"]=pd.to_datetime(data["Date"])
data["5 MA"]=pd.Series(data['Close']).rolling(5,min_periods=5).mean()
data["10 MA"]=pd.Series(data['Close']).rolling(10,min_periods=10).mean()
data=data.fillna(0)
train_set = data[:200]
test_set = data[200:]
train_y = train_set["y"]
train_x = train_set[["Open", "Yesterday_Close", "Close","Volume","5 MA"]]
test_x = test_set[["Open", "Yesterday_Close", "Close","Volume","5 MA"]]
test_y = test_set["y"]
reg = linear_model.LinearRegression()
reg.fit(train_x, train_y)
print(reg.coef_)
y_train_hat = reg.predict(train_x)
print(mean_squared_error(train_y, y_train_hat))
y_test_hat = reg.predict(test_x)
print(mean_squared_error(test_y, y_test_hat))
reg_decision = DecisionTreeRegressor()
reg_decision.fit(train_x, train_y)
y_train_hat_decision = reg_decision.predict(train_x)
#print("Coefficients: ",reg_decision.score(train_x, train_y))
print(mean_squared_error(train_y, y_train_hat_decision))
y_test_hat_decision = reg_decision.predict(test_x)
print("均方誤差=",mean_squared_error(test_y, y_test_hat_decision))
print("均方誤差/平均值=",mean_squared_error(test_y/np.mean(test_y), y_test_hat_decision/np.mean(y_test_hat_decision))*100,"%")


#regr_1 = DecisionTreeRegressor(max_depth=5) #最大深度為2的決策樹
#regr_2 = DecisionTreeRegressor() #最大深度為5的決策樹

#regr_1.fit(train_x, train_y)
#regr_2.fit(train_x, train_y)

X_test_open = pd.Series(np.arange(60, 140, 1), name='Open')
X_test_Yesterday_Close = pd.Series(np.arange(60, 140, 1), name='Yesterday_Close')
X_test_Close = pd.Series(np.arange(60, 140, 1), name='Close')
X_test_Volume = pd.Series(np.arange(60, 140, 1), name='Volume')
X_test=pd.concat([X_test_open,X_test_Yesterday_Close,X_test_Close,X_test_Volume], axis=1)



fig,ax1=plt.subplots()
#y_1 = regr_1.predict(X_test)
y_2=reg_decision.predict(train_x)

ax1.scatter(test_x["Close"],test_y, c="darkorange", label="test_data")
ax1.scatter(train_x["Close"],train_y, c="red", label="train_data")
#plt.plot(X_test_open , y_1, color="cornflowerblue", label="max_depth=5", linewidth=2)
ax1.plot(train_x["Close"], y_2, color="yellowgreen", label="模擬值", linewidth=2)
ax1.set_xlabel("收盤價") #x軸代表data數值
ax1.set_ylabel("隔天報酬率(y)") #y軸代表target數值
ax1.set_title("Decision Tree Regression") #標示圖片的標題

plt.legend(ncol=3) #繪出圖例
plt.savefig("D:/APPL_decision_tree.png",dpi=200)
plt.show()
plt.clf()

y_2=reg.predict(train_x)
plt.scatter(test_x["Close"],test_y, c="darkorange", label="test_data")
plt.scatter(train_x["Close"],train_y, c="red", label="train_data")
plt.plot(train_x["Close"] , y_2, color="yellowgreen", label="模擬值", linewidth=2)
plt.xlabel("收盤價") #x軸代表data數值
plt.ylabel("隔天報酬率(y)") #y軸代表target數值
plt.title("Linear Regression") #標示圖片的標題
plt.legend(ncol=3) #繪出圖例
plt.savefig("D:/APPL_linear_regression.png",dpi=200)
plt.show()
plt.clf()



y_2=reg_decision.predict(data[["Open", "Yesterday_Close", "Close","Volume","5 MA"]])
fig,ax1=plt.subplots()
ax1.scatter(data["Date"],data["y"], c="darkorange", label="實際報酬率")
#plt.plot(X_test_open , y_1, color="cornflowerblue", label="max_depth=5", linewidth=2)
ax1.plot(data["Date"] , y_2, color="red", label="模擬值", linewidth=2)


ax1.set_ylabel("隔天報酬率") #y軸代表target數值

ax1.set_title("Decision Tree Regression with Time Scale") #標示圖片的標題
ax2=ax1.twinx()
ax2.plot(data["Date"],data["Close"],c='yellowgreen',label="收盤價")
ax2.plot(data["Date"][5:],data["5 MA"][5:],c='b',label="5 MA")
ax2.plot(data["Date"][10:],data["10 MA"][5:],c='p',label="10 MA")
ax2.set_ylabel('收盤價')
ax1.legend( loc='upper left',ncol=2) #繪出圖例
ax2.legend(bbox_to_anchor=(0.98, 0.14),ncol=2) #繪出圖例

plt.savefig("D:/Decision Tree Regression with Time.png",dpi=200)
plt.show()
plt.clf()


