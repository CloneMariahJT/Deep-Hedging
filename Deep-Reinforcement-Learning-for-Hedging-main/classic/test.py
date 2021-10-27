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

def test_data(model, apm, opm, num_steps, dt, n=10):
    losses = []
    delta_losses = []
    data_gen = []
    data_gen_sub = {"id":0,"data":[],"losses":[],"delta_losses":[]}
    for i in range(n):
        D = generate_data(apm, opm, num_steps, dt, n=1)
        data_gen_sub['id']=i
        data_gen_sub["data"].append(D)
        old_delta = 0
        old_out = 0
        for tupel in D:
            inp = torch.tensor(np.array([old_out, tupel["delta"], tupel["p"], tupel["ttm"]])).double()
            out = model(inp)
            trading_costs = (T / num_steps) * (abs(old_out - out) + 0.01 * (old_out - out) ** 2)
            pl = (-tupel["nop"] + tupel["op"]) + out * (tupel["np"] - tupel["p"]) - trading_costs
            loss = torch.pow(pl, 2) - 1 / 1000 * pl
            losses.append(loss.detach().numpy()[0])

            trading_costs_delta = (T / num_steps) * (abs(old_delta - tupel["delta"]) + 0.01 *
                                                     (old_delta - tupel["delta"]) ** 2)
            pl_delta = (-tupel["nop"] + tupel["op"]) + tupel["delta"] * (tupel["np"] - tupel["p"]) - trading_costs_delta
            delta_loss = (pl_delta) ** 2 - 1/1000 *pl_delta
            delta_losses.append(delta_loss)
            old_delta = tupel["delta"]
            old_out = out.detach().numpy()[0]
        data_gen_sub["losses"].append(losses)
        data_gen_sub["delta_losses"].append(delta_losses)
        data_gen.append(data_gen_sub)

    return losses, delta_losses, data_gen

volatility = 0.15
strike_price = 1
starting_price = 1
final_coupon = 0.1
mu = 0.0
T = 1.0
num_steps = 252
dt = T / num_steps
risk_free_interest_rate = 0.01

seed = 345
np.random.seed(seed)
torch.manual_seed(seed)

model = FullyConnected(4, 16, 1, 5, f=torch.nn.functional.relu)
model.double()
apm = GBM(mu=mu, dt=dt, s_0=starting_price, sigma=volatility)
opm = DigitCI(strike_price=strike_price, risk_free_interest_rate=risk_free_interest_rate, volatility=volatility, T=T, dt=dt, final_coupon=final_coupon)
#opm = BSM(strike_price=strike_price, risk_free_interest_rate=risk_free_interest_rate, volatility=volatility, T=T, dt=dt)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
num_eps = 200
norm_factor = 10000

test_res = []
test_res_delta = []
data_gen = []

for i in range(num_eps):
    print("episode: ", i)
    
    if i%20 == 0:
        test_data_temp = []
        losses, delta_losses, test_data_temp = test_data(model, apm, opm, num_steps=num_steps, dt=dt, n=20)
        mean_test_result = np.mean(losses)
        test_res.append(mean_test_result)
        mean_delta_test_results = np.mean(delta_losses)
        test_res_delta.append(mean_delta_test_results)
        print("test_result: ", mean_test_result, mean_delta_test_results)
        data_gen.append(test_data_temp)

    if i == 120:
        for q in optimizer.param_groups:
            q["lr"] = 0.001

    if i == 200:
        for q in optimizer.param_groups:
            q["lr"] = 0.0001

    if i == 260:
        for q in optimizer.param_groups:
            q["lr"] = 0.00001

    if i == 300:
        for q in optimizer.param_groups:
            q["lr"] = 0.000001

    if i == 1350:
        for q in optimizer.param_groups:
            q["lr"] = 0.0000001

    loss = 0
    for _ in range(10):
        D = generate_data(apm, opm, num_steps, dt, n=1)
        old_out = 0
        for tupel in D:
            inp = torch.tensor(np.array([old_out, tupel["delta"], tupel["p"], tupel["ttm"]]), dtype=torch.float64)
            out = model(inp)
            trading_costs = (T / num_steps) * (abs(old_out - out) + 0.01 * (old_out - out) ** 2)
            pl = (-tupel["nop"] + tupel["op"]) + out * (tupel["np"] - tupel["p"]) - trading_costs
            loss += norm_factor * (torch.pow(pl, 2) - 1 / 1000 * pl)
            old_out = out.detach().numpy()[0]

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print("loss:", loss.detach().numpy())

np.save('classical_res_data_CI.npy', data_gen) 
print(data_gen)
print(test_res)
print(test_res_delta)
