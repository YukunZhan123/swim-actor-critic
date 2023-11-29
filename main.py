import sys
import time
import os
import socket
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import csv
from swim_api import get_system_state, perform_action  # Replace with your actual SWIM API calls

torch.autograd.set_detect_anomaly(True)


# Define Actor network architecture
class Actor(nn.Module):
    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, action_size)
        )
        self.log_softmax = nn.LogSoftmax(dim=-1)  # Use LogSoftmax

    def forward(self, state):
        x = self.network(state)
        return self.log_softmax(x)  # Return log probabilities


# Define Critic network architecture
class Critic(nn.Module):
    def __init__(self, state_size, action_size):
        super(Critic, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size + action_size, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state, action_one_hot):
        x = torch.cat([state, action_one_hot], dim=1)
        return self.network(x)


# Actor-Critic Manager
class ActorCriticManager:
    def __init__(self, state_size, action_size, epsilon, actor_lr=0.0002, critic_lr=0.001):
        self.actor = Actor(state_size, action_size)
        self.critic = Critic(state_size, action_size)
        # Set custom learning rates for the actor and critic optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.epsilon = epsilon
        self.gamma = 0  # Discount factor for future rewards
        self.critic_loss = 0
        self.actor_loss = 0

    def select_action(self, state):
        state_tensor = torch.FloatTensor(state)
        with torch.no_grad():
            action_probs = self.actor(state_tensor).numpy()
            print("best action is: ", np.argmax(action_probs), action_probs)
        two_choices = np.argpartition(action_probs, -2)[-2:] 
        if random.random() < self.epsilon:
            print("random explore")
            action = np.random.choice(len(action_probs))
            self.epsilon *= 0.9999
        else:
            print("choose best action")
            action = np.random.choice(two_choices)
        action_one_hot = np.zeros(9)
        action_one_hot[action] = 1
        return action, action_one_hot

    def update(self, state, action, action_one_hot, reward, next_state, done):
        print("reward ", reward)
        server = int(state[4])
        dimmer = float(state[3])
        s_to_one_hot = {1: [1, 0, 0], 2: [0, 1, 0], 3: [0, 0, 1]}
        d_to_one_hot = {0.1: [1, 0, 0], 0.5: [0, 1, 0], 0.9: [0, 0, 1]}
        state = state[:3] + d_to_one_hot[dimmer] + state[4:]
        state = state[:-1] + s_to_one_hot[server]
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
        reward_tensor = torch.FloatTensor([reward]).unsqueeze(0)
        done_tensor = torch.FloatTensor([done]).unsqueeze(0)
        action_tensor = torch.FloatTensor(action_one_hot).unsqueeze(0)

        # Calculate the critic's estimate of the state's value and next state's value
        value = self.critic(state_tensor, action_tensor)
        print("predicted: ", value.item())
        # next_value = self.critic(next_state_tensor).detach()

        # Calculate the temporal difference target
        # td_target = reward_tensor + self.gamma * next_value * (1 - done_tensor)
        td_target = reward_tensor
        # Compute the critic loss
        critic_loss = (td_target - value).pow(2)
        self.critic_loss = critic_loss.item()
        print("critic_loss ", critic_loss.item())
        self.critic_optimizer.zero_grad()
        critic_loss.backward(retain_graph=True)  # Default is retain_graph=False
        self.critic_optimizer.step()

        # Compute the actor loss
        action_probs = self.actor(state_tensor)
        action_index = torch.LongTensor([action]).unsqueeze(0)
        # Compute the actor loss using log probabilities
        # Log probabilities
        action_log_probs = action_probs.gather(1, action_index).squeeze(1)
        actor_loss = -action_log_probs * (td_target - value.detach()).squeeze()
        self.actor_loss = actor_loss.item()
        print("actor_loss ", actor_loss.item())

        # Reset gradients and perform a backward pass for the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()


# Utility function
# def calculate_utility(state):
#     # Define target and threshold levels for response time and throughput
#     TARGET_RT = 0.06  # Target response time (lower is better)
#     TARGET_TP = 7  # Target throughput (higher is better)
#
#
#     # Calculate a combined throughput
#     combined_tp = state[1] * 6
#     combined_rt = state[0] * 0.05
#
#     # Utility is higher when the combined response time is lower than the target, and combined throughput meets the target
#     rt_utility = max(0, TARGET_RT - combined_rt) / TARGET_RT
#     tp_utility = min(1, combined_tp / TARGET_TP)
#
#     # Cost efficiency is assumed to be inversely proportional to the number of servers
#     cost_utility = 1 / (state[-1] * 3) if state[-1] else 0
#
#
#     # The overall utility is a weighted sum of response time utility, throughput utility, and cost utility
#     utility = rt_utility * 0.4 + tp_utility * 0.4 + cost_utility * 0.2
#     return utility

def calculate_utility(state, maxServers, maxServiceRate, RT_THRESHOLD):
    basicRevenue = 1
    optRevenue = 1.5
    serverCost = 10

    precision = 1e-5

    maxThroughput = maxServers * maxServiceRate

    # Unpacking state values (assuming state is [avgResponseTime, avgThroughput, arrivalRateMean, dimmer, avgServers])
    avgResponseTime = state[0] * 0.05  # Assuming state[0] is the average response time
    avgThroughput = state[1] * 6  # Assuming state[1] is the average throughput
    arrivalRateMean = state[2] * 13  # Assuming state[2] is the mean arrival rate
    dimmer = state[3]  # Assuming state[3] is the dimmer value
    avgServers = state[4]  # Assuming state[4] is the average number of servers

    Ur = (arrivalRateMean * ((1 - dimmer) * basicRevenue + dimmer * optRevenue))
    Uc = serverCost * (maxServers - avgServers)
    UrOpt = arrivalRateMean * optRevenue

    utility = 0
    if avgResponseTime <= RT_THRESHOLD and Ur >= UrOpt - precision:
        utility = Ur - Uc
    elif avgResponseTime <= RT_THRESHOLD:
        utility = Ur
    else:
        utility = (max(0.0, arrivalRateMean - maxThroughput) * optRevenue) - Uc

    return utility


def reset():
    print("resetting environment")
    host = "127.0.0.1"
    port = 4242
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    conn = s.connect((host, port))
    s.sendall(b'set_dimmer 0.5')
    s.recv(1024)


if len(sys.argv) == 2:
    epsilon = float(sys.argv[1])
else:
    epsilon = 1

# Real-time execution loop
state_size = 9  # Size of the state vector
# action_choices = [["add", 0], ["remove", 0], ["nothing", 0.25], ["nothing", -0.25], ["nothing", 0], ["add", 0.25], ["add", -0.25], ["remove", 0.25], ["remove", -0.25]]
action_choices = [["add", 0], ["remove", 0], ["nothing", 0.4], ["nothing", -0.4], ["nothing", 0], ["add", 0.4],
                  ["add", -0.4], ["remove", 0.4], ["remove", -0.4]]
action_size = 9  # Number of actions

# Define paths to your saved model files
actor_model_path = 'actor.pth'
critic_model_path = 'critic.pth'

# Initialize your manager
manager = ActorCriticManager(state_size, action_size, epsilon)

# Check if both the actor and critic model files exist
if os.path.exists(actor_model_path) and os.path.exists(critic_model_path):
    print("Loading saved model weights.")
    manager.actor.load_state_dict(torch.load(actor_model_path))
    manager.critic.load_state_dict(torch.load(critic_model_path))
else:
    print("No saved model weights found, initializing new models.")

reset()

# Add a counter for iterations
iteration_counter = 0
maxServiceRate = 1 / 0.04452713
maxServers = 3
RT_THRESHOLD = 0.075  # Example response time threshold

csv_file_path = 'log_data.csv'
csv_header = ['Iteration', 'Reward', 'Critic Loss', 'Actor Loss']

if os.path.exists(csv_file_path):
    write_header = False
else:
    write_header = True

while True:  # Replace with the condition appropriate for your application
    # Monitor
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print("it: ", iteration_counter)

    if iteration_counter % 10 == 0:
        # Save the actor and critic networks
        torch.save(manager.actor.state_dict(), f'actor.pth')
        torch.save(manager.critic.state_dict(), f'critic.pth')

    state = get_system_state()
    state_modified = state[:]
    server = int(state[4])
    dimmer = float(state[3])
    s_to_one_hot = {1: [1, 0, 0], 2: [0, 1, 0], 3: [0, 0, 1]}
    d_to_one_hot = {0.1: [1, 0, 0], 0.5: [0, 1, 0], 0.9: [0, 0, 1]}
    state_modified = state_modified[:3] + d_to_one_hot[dimmer] + state_modified[4:]
    state_modified = state_modified[:-1] + s_to_one_hot[server]

    # Plan
    action, action_one_hot = manager.select_action(state_modified)
    print(action, action_one_hot)

    # Execute
    done = perform_action(state, action_choices[action])  # Implement this function
    time.sleep(5)
    next_state = get_system_state()

    # Analyze
    if done:
        reward = -30
    else:
        reward = calculate_utility(next_state, maxServers, maxServiceRate, RT_THRESHOLD)
    # Update the manager
    manager.update(state, int(action), action_one_hot, reward, next_state, done)

    # record data
    with open(csv_file_path, 'a', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)

        # Write header if it's a new file
        if write_header:
            csv_writer.writerow(csv_header)
            write_header = False

        # Write data for each iteration
        csv_writer.writerow([iteration_counter, reward, manager.critic_loss, manager.actor_loss])

    iteration_counter += 1  # Increment the counter

    if done:  # Implement the logic to determine if the episode has ended
        reset()
        continue
