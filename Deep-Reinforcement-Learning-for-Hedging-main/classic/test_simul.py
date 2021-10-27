import numpy as np
from financial_models.option_price_models import BSM, DigitCI
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator

def generate_data(apm, opm, num_steps, dt, n=3):
    D = []
    for j in range(n):
        apm.reset()
        price = apm.get_current_price()
        opt_price = opm.compute_option_price(T, price, mode="ttm")
        for p in range(num_steps):
            apm.compute_next_price()
            next_price = apm.get_current_price()
            next_opt_price = opm.compute_option_price(T - (p + 1) * dt, next_price, mode="ttm")
            delta = opm.compute_delta_ttm(T - p * dt, price)
            D.append({"p": price, "np": next_price, "op": opt_price, "nop": next_opt_price, "ttm": T - p * dt,
                      "nttm": T - (p + 1) * dt, "delta": delta})
            price = next_price
            opt_price = next_opt_price
    return D

def test_data(apm, opm, num_steps, dt, n=10):
    losses = []
    delta_losses = []
    data_gen = {"id":0,"data":[]}
    for i in range(n):
        D = generate_data(apm, opm, num_steps, dt, n=1)
        data_gen["id"] = i
        data_gen["data"].append(D)
    return data_gen


volatility = np.linspace(0.04, 0.2, 9)
strike_price = np.linspace(0.8, 1.25,46)
starting_price = 1
final_coupon = 0.5

mu = 0.0
T = 1.0
num_steps = 252
dt = T / num_steps
risk_free_interest_rate = 0.02

# data_eu_call_simul = []
# for i in range(len(volatility)):
#     vol = volatility[i]
#     delta_line = []
#     for j in range(len(strike_price)):
#         k =  strike_price[j]
#         #opm = DigitCI(strike_price=strike_price, risk_free_interest_rate=risk_free_interest_rate, volatility=volatility, T=T, dt=dt, final_coupon=final_coupon)
#         opm = BSM(strike_price=k, risk_free_interest_rate=risk_free_interest_rate, volatility=vol, T=T, dt=dt)
#         delta_line.append(opm.compute_delta(n=0, asset_price = 1))
#     data_eu_call_simul.append(delta_line)

# print(data_eu_call_simul)

# data_binary_call_simul = []
# for i in range(len(volatility)):
#     vol = volatility[i]
#     delta_line = []
#     for j in range(len(strike_price)):
#         k = strike_price[j]
#         opm = DigitCI(strike_price=k, risk_free_interest_rate=risk_free_interest_rate, volatility=vol, T=T, dt=dt, final_coupon=final_coupon)
#         delta_line.append(opm.compute_delta(n=0, asset_price = 1)) 
#     data_binary_call_simul.append(delta_line)

# print(data_binary_call_simul)

for i in np.arange(0,1, 0.25):

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    X = volatility
    Y = strike_price
    X, Y = np.meshgrid(X, Y)
    opm = DigitCI(strike_price=Y, risk_free_interest_rate=risk_free_interest_rate, volatility=X, T=T, dt=dt, final_coupon=final_coupon)
    Z = opm.compute_delta(n=i*num_steps, asset_price = 1)

    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)

    # Customize the z axis.
    ax.set_zlim(0, np.max(Z)+0.2)
    ax.set_xlabel("volatility")
    ax.set_ylabel("strike price")
    ax.set_zlabel("delta")
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.savefig("delta_surface_binary_"+str(1-i)+"Y.png")
    plt.show()
    plt.close()


for i in np.arange(0,1, 0.25):

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    X = volatility
    Y = strike_price
    X, Y = np.meshgrid(X, Y)
    opm = BSM(strike_price=Y, risk_free_interest_rate=risk_free_interest_rate, volatility=X, T=T, dt=dt)
    Z = opm.compute_delta(n=i*num_steps, asset_price = 1)

    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)

    # Customize the z axis.
    ax.set_zlim(0, np.max(Z)+0.2)
    ax.set_xlabel("volatility")
    ax.set_ylabel("strike price")
    ax.set_zlabel("delta")
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.savefig("delta_surface_european_option_"+str(1-i)+"Y.png")
    plt.show()
    plt.close()

