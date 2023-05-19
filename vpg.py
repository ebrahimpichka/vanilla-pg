# import dependencies
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import gym
from collections import deque
import random
import matplotlib.pyplot as plt 

class PolicyNet(nn.Module):
    def __init__(self, state_dim, n_actions, n_hidden):
        super(PolicyNet, self).__init__()
        self.linear1 = nn.Linear(state_dim, n_hidden)
        self.linear2 = nn.Linear(n_hidden, n_actions)
        self.rewards, self.saved_actions = [], []

    def forward(self, x):
        out = F.relu(self.linear1(x))
        out = self.linear2(out)
        aprob = F.softmax(out, dim=1) # Softmax for categorical probabilities
        return aprob

class ValueNet(nn.Module):
    def __init__(self, state_dim, n_hidden):
        super(ValueNet, self).__init__()
        self.linear1 = nn.Linear(state_dim, n_hidden)
        self.linear2 = nn.Linear(n_hidden, 1)

    def forward(self, x):
        out = F.relu(self.linear1(x))
        V = self.linear2(out)
        return V

# create environment
env = gym.make("CartPole-v1") # sample toy environment

# instantiate the policy and value networks
policy = PolicyNet(state_dim=env.observation_space.shape[0], n_actions=env.action_space.n, n_hidden=64)
value = ValueNet(state_dim=env.observation_space.shape[0], n_hidden=64)

# instantiate an optimizer
policy_optimizer = torch.optim.SGD(policy.parameters(), lr=3e-7)
value_optimizer = torch.optim.SGD(value.parameters(), lr=1e-7)

# initialize gamma and stats
gamma=0.99
num_episodes = 4000
returns_deq = deque(maxlen=100)
memory_buffer_deq = deque(maxlen=5000)
to_plot_avgret = []
to_plot_polutil = []
to_plot_valloss = []

for n_ep in range(num_episodes):
    rewards = []
    actions = []
    states  = []
    # reset environment
    state = env.reset()
    done = False

    while not done:
        # recieve action probabilities from policy function
        probs = policy(torch.tensor(state).unsqueeze(0).float())

        # sample an action from the policy distribution
        policy_prob_dist = Categorical(probs)
        action = policy_prob_dist.sample()

        # take that action in the environment
        new_state, reward, done, info = env.step(action.item())

        # store state, action and reward
        states.append(state)
        actions.append(action)
        rewards.append(reward)

        memory_buffer_deq.append((state, reward, new_state))

        state = new_state

    ### UPDATE POLICY NET ###
    rewards = np.array(rewards)
    # calculate rewards-to-go
    R = torch.tensor([np.sum(rewards[i:]*(gamma**np.array(range(i, len(rewards))))) for i in range(len(rewards))])

    # cast states and actions to tensors
    states = torch.tensor(states).float()
    actions = torch.tensor(actions)

    # calculate baseline V(s)
    with torch.no_grad():
        baseline = value(states)

    # calculate utility func
    probs = policy(states)
    sampler = Categorical(probs)
    log_probs = - sampler.log_prob(actions)   # "-" is because we are doing gradient ascent
    utility = torch.sum(log_probs * (R-baseline)) # loss that when differentiated with autograd gives the gradient of J(θ)
    
    # update policy weights
    policy_optimizer.zero_grad()
    utility.backward()
    policy_optimizer.step()

    ### UPDATE VALUE NET ###

    # getting batch experience data 
    batch_experience = random.sample(list(memory_buffer_deq), min(512, len(memory_buffer_deq)))
    state_batch = torch.tensor([exp[0] for exp in batch_experience])
    reward_batch = torch.tensor([exp[1] for exp in batch_experience]).view(-1,1)
    new_state_batch = torch.tensor([exp[2] for exp in batch_experience])


    with torch.no_grad():
        target = reward_batch + gamma*value(new_state_batch)
    current_state_value = value(new_state_batch)

    value_loss = torch.nn.functional.mse_loss(current_state_value, target)
    # update value weights
    value_optimizer.zero_grad()
    value_loss.backward()
    value_optimizer.step()

    to_plot_avgret.append(np.mean(returns_deq))
    to_plot_polutil.append(utility.item())
    to_plot_valloss.append(value_loss.item())

    # calculate average return and print it out
    returns_deq.append(np.sum(rewards))
    if n_ep%100==0:
        print("Episode: {:6d}\tAvg. Return: {:6.2f}".format(n_ep, np.mean(returns_deq)))

# close environment
env.close()


plt.figure(figsize=(8,29))

plt.subplot(3,1,1)
plt.title("Averagre Return")
plt.plot(list(range(1,num_episodes+1)), to_plot_avgret)
plt.grid()
plt.xlabel("episode")
plt.ylabel("avg return")

plt.subplot(3,1,2)
plt.title("Policy Utility")
plt.plot(list(range(1,num_episodes+1)), to_plot_polutil)
plt.grid()
plt.xlabel("episode")
plt.ylabel("Policy Utility")

plt.subplot(3,1,3)
plt.title("Value Loss")
plt.plot(list(range(1,num_episodes+1)), to_plot_valloss)
plt.grid()
plt.xlabel("episode")
plt.ylabel("value loss")

plt.tight_layout(pad=1.5)
plt.show()