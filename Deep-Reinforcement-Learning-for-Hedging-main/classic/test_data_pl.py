import numpy as np
from financial_models.asset_price_models import GBM
import matplotlib.pyplot as plt
import pandas as pd


# data=np.load('classical_res_data_BSM.npy')
# first_data = data[0]
# ins = first_data[0]
# print(ins['data'])
# print(len(ins))
#data_CI=np.load('classical_res_data_CI.npy')
#print(len(data_CI))

volatility = 0.15
strike_price = 1
starting_price = 1
final_coupon = 0.1
mu = 0.05
T = 1.0
num_steps = 252
dt = T / num_steps
risk_free_interest_rate = 0.01
apm = GBM(mu=mu, dt=dt, s_0=starting_price, sigma=volatility)

price = []
for j in range(10):
    apm.reset()
    price_sub=[]
    price_temp = apm.get_current_price()
    price_sub.append(price_temp)
    for p in range(num_steps):
        apm.compute_next_price()
        next_price = apm.get_current_price()
        price_temp = next_price
        price_sub.append(price_temp)
    price.append(price_sub)
        #apm = GBM(mu=mu, dt=dt, s_0=price, sigma=volatility)

price_df = pd.DataFrame(price)
price_df = price_df.T

plt.figure()
price_df.plot(cmap='winter')
plt.xlabel("days")
plt.ylabel("price")
legend = plt.legend()
legend.remove()
plt.savefig("GBM Simulations.png")

plt.show()

#plt.figure(figsize = (20,10))

#plt.title("Daily Volatility: " + str(volatility))
#plt.plot(price)
#plt.ylabel('Stock Prices')
#plt.xlabel('Prediction Days')
    
#plt.show()