import torch.nn.functional as F
import matplotlib.pyplot as plt
from machin.frame.algorithms import PPO
from machin.utils.logging import default_logger as logger
import gym
import gym_hedging
import torch as t 
from torch import nn
import random
import numpy as np
from torch.distributions import Categorical
from financial_models.asset_price_models import GBM
from financial_models.option_price_models import BSM, DigitCI


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
        for p in range(int(num_steps)):
            apm.compute_next_price()
            next_price = apm.get_current_price()
            next_opt_price = opm.compute_option_price(T - (p + 1) * dt, next_price, mode="ttm")
            delta = opm.compute_delta_ttm(T - p * dt, price)
            D.append({"p": price, "np": next_price, "op": opt_price, "nop": next_opt_price, "ttm": T - p * dt,
                      "nttm": T - (p + 1) * dt, "delta": delta})
            price = next_price
            opt_price = next_opt_price
    return D

# model definition
class Actor(nn.Module):
    def __init__(self, state_dim, action_num):
        super().__init__()

        self.fc1 = nn.Linear(state_dim, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, action_num)

    def forward(self, state, action=None):
        a = t.relu(self.fc1(state))
        a = t.relu(self.fc2(a))
        probs = t.softmax(self.fc3(a), dim=1)
        dist = Categorical(probs=probs)
        act = action if action is not None else dist.sample()
        act_entropy = dist.entropy()
        act_log_prob = dist.log_prob(act.flatten())
        return act, act_log_prob, act_entropy


class Critic(nn.Module):
    def __init__(self, state_dim):
        super().__init__()

        self.fc1 = nn.Linear(state_dim, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 1)

    def forward(self, state):
        v = t.relu(self.fc1(state))
        v = t.relu(self.fc2(v))
        v = self.fc3(v)
        return v

mu = 0
dt = 1/128
T = 1
num_steps = T/dt
s_0 = 1
strike_price = s_0
sigma = 0.15
volatility = 0.15
r = 0.01
norm_factor = 10000
final_coupon = 0.1

apm = GBM(mu=mu, dt=dt, s_0=s_0, sigma=sigma)

#opm = BSM(strike_price=strike_price, risk_free_interest_rate=r, volatility=sigma, T=T, dt=dt)
opm = DigitCI(strike_price=strike_price, risk_free_interest_rate=r, volatility=sigma, T=T, dt=dt, final_coupon=final_coupon)

env = gym.make('hedging-v0', asset_price_model=apm, dt=dt, T=T, num_steps=num_steps, trading_cost_para=1,
                     L=1, strike_price=strike_price, int_holdings=False, initial_holding=0, mode="PL",
               option_price_model=opm)

action_num = 21

# configurations
observe_dim = 5
max_episodes = 500
max_steps = 200
solved_reward = 190
solved_repeat = 5

actor = Actor(observe_dim, action_num)
critic = Critic(observe_dim)

ppo = PPO(actor, critic, t.optim.Adam, nn.MSELoss(reduction="sum"))

episode, step, reward_fulfilled = 0, 0, 0
smoothed_total_reward = 0

rew = []
rew_m_l = []

for j in range(max_episodes):
    print("episode: ", j)
    #state = env.reset()
    #state = state[[0,1,2,4]]
    #tmp_observations = []

    total_reward = 0
    terminal = False
    step = 0
    state = t.tensor(env.reset(), dtype=t.float32).view(1, observe_dim)

    tmp_observations = []
    while not terminal and step <= max_steps:
        step += 1
        with t.no_grad():
            old_state = state
            # agent model inference
            action = ppo.act({"state": old_state})[0]
            state, reward, terminal, _ = env.step(action.item())
            state = t.tensor(state, dtype=t.float32).view(1, observe_dim)
            #print(reward)
            total_reward += np.mean(reward)
            adj_reward = np.mean(reward)
            adj_reward = -(adj_reward ** 2) + 1 / 1000 * adj_reward
            rew.append(adj_reward)
            tmp_observations.append({
                "state": {"state": old_state},
                "action": {"action": action},

                "next_state": {"state": state},
                "reward": adj_reward,
                "terminal": terminal or step == max_steps})            
    # update
    ppo.store_episode(tmp_observations)
    ppo.update()
    rew_m_l.append(np.mean(rew))

    # show reward
    smoothed_total_reward = smoothed_total_reward * 0.9 + adj_reward * 0.1
    logger.info(f"Episode {j} total reward={smoothed_total_reward:.2f}")

    if smoothed_total_reward > solved_reward:
        reward_fulfilled += 1
        if reward_fulfilled >= solved_repeat:
            logger.info("Environment solved!")
            exit(0)
    else:
        reward_fulfilled = 0


#ppo.save("ppo_model")

plt.figure()
t_time = list(range(1, len(rew_m_l)+1))
plt.plot(t_time, rew_m_l, label="RL PPO Reward")
plt.xlabel("epoch times")
plt.ylabel("Reward")
plt.legend()
plt.savefig("PPO Hedging Digital Option.png")

plt.show()
plt.clf()
plt.cla()
plt.close()


##### Out-of-sample testo PPO #####
episode, step, reward_fulfilled = 0, 0, 0
smoothed_total_reward = 0
rew = []
rew_m_l = []

for j in range(max_episodes):
    reward_list = []

    total_reward = 0
    terminal = False
    step = 0

    state =state = t.tensor(env.reset(), dtype=t.float32).view(1, observe_dim)

    tmp_observations = []
    while not terminal and step <= max_steps:
        step += 1
        with t.no_grad():
            old_state = state
            # agent model inference
            action = ppo.act({"state": old_state})[0]
            state, reward, terminal, _ = env.step(action.item())
            state = t.tensor(state, dtype=t.float32).view(1, observe_dim)
            #print(reward)
            total_reward += np.mean(reward)
            adj_reward = np.mean(reward)
            adj_reward = -(adj_reward ** 2) + 1 / 1000 * adj_reward
            rew.append(adj_reward)
            tmp_observations.append({
                "state": {"state": old_state},
                "action": {"action": action},
                "next_state": {"state": state},
                "reward": adj_reward,
                "terminal": terminal or step == max_steps})            
    
    rew_m_l.append(np.mean(rew))
    print(np.mean(rew))
    env.render()


print(rew_m_l)
plt.hist(rew_m_l, alpha=0.5, label='PPO')
plt.title("PPO Hedging Digital Call Reward Distribution")
plt.savefig("PPO Hedging Digital Call Hist.png")
plt.show()


risk_free_interest_rate = 0.01
opm_dc = DigitCI(strike_price=strike_price, risk_free_interest_rate=risk_free_interest_rate, volatility=volatility, T=T, dt=dt, final_coupon=final_coupon)
num_eps = 520

model = FullyConnected(4, 16, 1, 5, f=t.nn.functional.relu)

optimizer = t.optim.Adam(model.parameters(), lr=0.01)

# Loading the saved model
model_dc= t.load("model_classic_cost_binary_Call.pth")
model_dc.double()
model_dc.eval()


test_res_dc = []
# Disable grad
with t.no_grad():
    for _ in range(max_episodes):
        # Retrieve item
        old_out_dc = 0
        D_dc = generate_data(apm, opm_dc, num_steps, dt, n=1)
        loss_dc = 0
        for tupel in D_dc:
            inp_dc = t.tensor(np.array([old_out_dc, tupel["delta"], tupel["p"], tupel["ttm"]]), dtype=t.float64)
            out_dc = model_dc(inp_dc)
            trading_costs_dc = (T / num_steps) * (abs(old_out_dc - out_dc) + 0.01 * (old_out_dc - out_dc) ** 2)
            pl_dc = (-tupel["nop"] + tupel["op"]) + out_dc * (tupel["np"] - tupel["p"]) - trading_costs_dc
            loss_dc += t.pow(pl_dc, 2) - 1 / 1000 * pl_dc
            old_out_dc = out_dc.detach().numpy()[0]
        print("loss dc:", loss_dc.detach().numpy())
        test_res_dc.append(loss_dc)
    

rew_m_l = ( rew_m_l- np.min(rew_m_l)) / (np.max(rew_m_l) - np.min(rew_m_l))
print(rew_m_l)
test_res_dc = ( test_res_dc- np.min(test_res_dc)) / (np.max(test_res_dc) - np.min(test_res_dc))
print(test_res_dc)
plt.hist(test_res_dc, alpha=0.5, label='FNN')
plt.title("FFN Hedging Digital Call Reward Distribution")
plt.savefig("Classic Deep Hedging Digital Call Hist.png")
plt.show()
plt.close()

# plt.hist(test_res_dc, alpha=0.5, label='FNN')
# plt.hist(rew_m_l, alpha=0.5, label='PPO')
# plt.legend()
# plt.savefig("Classic Deep Hedging vs PPO Hedging Digital Call Hist.png")

plt.close()

plt.scatter(rew_m_l, test_res_dc)
plt.xlabel("PPO")
plt.ylabel("FFN")
plt.title("Classic Deep Hedging vs PPO Hedging Digital Call")
plt.savefig("Classic Deep Hedging vs PPO Hedging Digital Call Hist.png")
plt.show()
