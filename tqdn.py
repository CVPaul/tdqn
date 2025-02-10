import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

# Define the Deep Q-Network (DQN) model
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Experience Replay Buffer
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
    
    def push(self, experience):
        self.memory.append(experience)
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

# TDQN Algorithm
class TDQN:
    def __init__(self, state_dim, action_dim, capacity=10000, gamma=0.99, epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.995, lr=0.001, target_update_freq=10):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.memory = ReplayMemory(capacity)
        
        # Initialize networks
        self.policy_net = DQN(state_dim, action_dim)
        self.target_net = DQN(state_dim, action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()
        
        self.target_update_freq = target_update_freq
        self.steps_done = 0

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            with torch.no_grad():
                return torch.argmax(self.policy_net(torch.tensor(state, dtype=torch.float32))).item()
    
    def train_step(self, batch_size):
        if len(self.memory) < batch_size:
            return
        
        batch = self.memory.sample(batch_size)
        states, actions, rewards, next_states = zip(*batch)
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        
        q_values = self.policy_net(states).gather(1, actions).squeeze()
        next_q_values = self.target_net(next_states).max(1)[0].detach()
        targets = rewards + (self.gamma * next_q_values)
        
        loss = self.loss_fn(q_values, targets)
        
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
    
    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def anneal_epsilon(self):
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
    
    def train(self, env, num_episodes, max_steps, batch_size):
        for episode in range(num_episodes):
            state = env.reset()
            for t in range(max_steps):
                action = self.select_action(state)
                next_state, reward, done, _ = env.step(action)
                self.memory.push((state, action, reward, next_state))
                self.train_step(batch_size)
                state = next_state
                if done:
                    break
                
                if t % self.target_update_freq == 0:
                    self.update_target_network()
            
            self.anneal_epsilon()