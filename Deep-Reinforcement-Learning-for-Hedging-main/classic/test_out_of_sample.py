import numpy as np
import torch.nn as nn
import torch
import random
import torch.nn.functional as F
from financial_models.option_price_models import BSM, DigitCI
from financial_models.asset_price_models import GBM
import matplotlib.pyplot as plt

class FullyConnected(nn.Module):
    def __init__(self, input, hidden, out_size, num_layers, f):
        super(FullyConnected, self).__init__()

        self.num_layers = num_layers

        self.first_layer = nn.Linear(input, hidden)

        self.linear = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(num_layers)])

        self.f = f

        self.out_layer = nn.Linear(hidden, out_size)

    def forward(self, x):
        x = self.f(self.first_layer(x))
        for layer in range(self.num_layers):
            x = self.linear[layer](x)
            x = self.f(x)

        x = self.f(x)
        x = self.out_layer(x)

        return x

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


volatility = 0.15
strike_price = 1
starting_price = 1
final_coupon = 0.1
mu = 0.0
T = 1.0
num_steps = 252
dt = T / num_steps
risk_free_interest_rate = 0.01
norm_factor = 10000

apm = GBM(mu=mu, dt=dt, s_0=starting_price, sigma=volatility)
opm_dc = DigitCI(strike_price=strike_price, risk_free_interest_rate=risk_free_interest_rate, volatility=volatility, T=T, dt=dt, final_coupon=final_coupon)
opm_ec = BSM(strike_price=strike_price, risk_free_interest_rate=risk_free_interest_rate, volatility=volatility, T=T, dt=dt)
num_eps = 520

model = FullyConnected(4, 16, 1, 5, f=torch.nn.functional.relu)
print(torch.load("model_classic_cost_Eur_Call.pth"))

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Loading the saved model
model_ec= torch.load("model_classic_cost_Eur_Call.pth")
model_ec.double()
model_ec.eval()
model_dc= torch.load("model_classic_cost_binary_Call.pth")
model_dc.double()
model_dc.eval()


test_res_ec = []
test_res_dc = []
# Disable grad
with torch.no_grad():
        
    for _ in range(1000):
        # Retrieve item
        old_out_ec = 0
        D_ec = generate_data(apm, opm_ec, num_steps, dt, n=1)
        loss_ec = 0
        for tupel in D_ec:
            inp_ec = torch.tensor(np.array([old_out_ec, tupel["delta"], tupel["p"], tupel["ttm"]]), dtype=torch.float64)
            out_ec = model_ec(inp_ec)
            trading_costs_ec = (T / num_steps) * (abs(old_out_ec - out_ec) + 0.01 * (old_out_ec - out_ec) ** 2)
            pl_ec = (-tupel["nop"] + tupel["op"]) + out_ec * (tupel["np"] - tupel["p"]) - trading_costs_ec
            loss_ec += norm_factor * (torch.pow(pl_ec, 2) - 1 / 1000 * pl_ec)
            old_out_ec = out_ec.detach().numpy()[0]
        print("loss ec:", loss_ec.detach().numpy())
        test_res_ec.append(loss_ec)

        old_out_dc = 0
        D_dc = generate_data(apm, opm_dc, num_steps, dt, n=1)
        loss_dc = 0
        for tupel in D_dc:
            inp_dc = torch.tensor(np.array([old_out_dc, tupel["delta"], tupel["p"], tupel["ttm"]]), dtype=torch.float64)
            out_dc = model_dc(inp_dc)
            trading_costs_dc = (T / num_steps) * (abs(old_out_dc - out_dc) + 0.01 * (old_out_dc - out_dc) ** 2)
            pl_dc = (-tupel["nop"] + tupel["op"]) + out_dc * (tupel["np"] - tupel["p"]) - trading_costs_dc
            loss_dc += norm_factor * (torch.pow(pl_dc, 2) - 1 / 1000 * pl_dc)
            old_out_dc = out_dc.detach().numpy()[0]
        print("loss dc:", loss_dc.detach().numpy())
        test_res_dc.append(loss_dc)
    

print(test_res_ec)
plt.hist(test_res_ec)
plt.ylabel("loss")
plt.savefig("Classic Deep Hedging Eur Call Loss Hist.png")

print(test_res_dc)
plt.hist(test_res_dc)
plt.ylabel("loss")
plt.savefig("Classic Deep Hedging Digital Call Loss Hist.png")

plt.hist(test_res_ec, alpha=0.5, label='European Call')
plt.hist(test_res_dc, alpha=0.5, label='Binary Call')
plt.legend(loc='upper right')
plt.show()
plt.savefig("Classic Deep Hedging Eur / Digit Call Loss Hist.png")

