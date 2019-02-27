import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt

data = pd.read_csv('2330.TW_deal_sim.csv')
price = data['Close']
# price[0,243]
x_p = np.array(price)#:243
# x_price
x_price = x_p[0:243]

i = []
for x in range(1,267):
    i.append(x)
x_range = np.array(i)


x_count = np.array([1,267])
x_train_time = np.array([220.5,235.5])



x_for_slope = np.polyfit(x_count, x_train_time, 1)
slope = np.poly1d(x_for_slope) 

# slope_plot = slope(x_count)

slope_price = slope(x_range)
slope_price
loss = x_p-slope_price
np.mean(loss)
np.mean(loss)





#===================計算漲跌==============================
h = []
n = 0
for t in range(len(x_p)):
    if x_p[n+1]> x_p[n]:
        h.append(1) #漲
    elif x_p[n+1]== x_p[n]:
        h.append(0) #不漲也不跌
    else:
        h.append(2) #跌
    n+=1
len(h)
data_trend = pd.DataFrame(h)
data_trend.to_csv('2330trend.csv')

#===================計算交易點 ==============================


p= []
for t in range(len(x_p)):
    
    if t ==0 or t==265:
        p.append(0)
    elif x_p[t]<x_p[t+1] and sum(p[0:t])==0:
        p.append(1)
    elif sum(p[0:t])==1:
        p.append(-1)
    else:
        p.append(0)

            

for n,g in enumerate(p):
    if g==-1:
        p[n]=2


    

##0持平 1買 2賣
# len(q)
data_trade = pd.DataFrame(p)
# data_trade
data_trade.to_csv('2330trade.csv')

plt.plot(x_range,x_p)
plt.plot(x_count,slope(x_count), color='g')
# plt.scatter(x_range, q ,color = 'r')
plt.show()

