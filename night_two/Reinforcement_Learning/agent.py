import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import random
from collections import deque

# Choose the device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DDPGAgent:
    def __init__(self, state_dim, action_dim, max_action, lstm_hidden_dim, num_lstm_layers, dropout_rate, max_buffer_size):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device

        # Actor Network
        self.actor_model = Actor(state_dim, action_dim, max_action, lstm_hidden_dim, num_lstm_layers, dropout_rate).to(self.device)
        self.target_actor_model = Actor(state_dim, action_dim, max_action, lstm_hidden_dim, num_lstm_layers, dropout_rate).to(self.device)
        
        # Initialize target actor model with actor model weights
        self.target_actor_model.load_state_dict(self.actor_model.state_dict())

        # Critic Network
        self.critic_model = Critic(state_dim, action_dim, lstm_hidden_dim, num_lstm_layers, dropout_rate).to(device)
        self.target_critic_model = Critic(state_dim, action_dim, lstm_hidden_dim, num_lstm_layers, dropout_rate).to(device)
        
        # Initialize target critic model with critic model weights
        self.target_critic_model.load_state_dict(self.critic_model.state_dict())

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor_model.parameters(), lr=0.001)
        self.critic_optimizer = torch.optim.Adam(self.critic_model.parameters(), lr=0.002)

        # Noise process
        self.noise = OUNoise(action_dim)

        # Replay Memory
        self.replay_buffer = ReplayBuffer(max_buffer_size)

    def get_action(self, state):
        # State is a NumPy array. Convert it to PyTorch tensor for the model.
        print(f"\nBefore conversion - Type of state: {type(state)}")  # Should be <class 'numpy.ndarray'>
        print(f"Before conversion - Dtype of state: {state.dtype}")  # Should be a numeric type, not 'object'
        print(f"Before conversion - Shape of state: {state.shape}")  # Just for additional information

        # Ensure state is a PyTorch tensor
        if not isinstance(state, torch.Tensor):
            print("\nConverting state to a tensor")
            state = torch.tensor(state, dtype=torch.float32).to(self.device)

        print(f"\nAfter conversion - Type of state: {type(state)}")
        print(f"After conversion - Dtype of state: {state.dtype}")
        print(f"After conversion - Shape of state: {state.shape}")

        self.actor_model.eval()
        with torch.no_grad():
            action = self.actor_model(state).numpy()
        self.actor_model.train()
        return action

    def act(self, state):
        action = self.actor_model(state)
        return self.noise.get_action(action)

    def learn(self, batch_size, gamma=0.99, tau=0.005):
        # Get a batch of experiences from the memory buffer
        experiences = self.replay_buffer.sample(batch_size)
        state, action, reward, next_state, done = [torch.FloatTensor(t).to(self.device) for t in zip(*experiences)]

        # Get predicted next-state actions and Q values from target models
        next_action = self.target_actor_model(next_state)
        next_q_value = self.target_critic_model(next_state, next_action)

        # Compute the target Q value
        target_q_value = reward + ((1 - done) * gamma * next_q_value).detach()

        # Get expected Q value from critic model
        expected_q_value = self.critic_model(state, action)

        # Compute Critic loss
        critic_loss = self.compute_critic_loss(expected_q_value, target_q_value)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Compute actor loss
        actor_loss = -self.critic_model(state, self.actor_model(state)).mean()

        # Optimize the actor 
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update the frozen target models
        self.soft_update(self.actor_model, self.target_actor_model, tau)
        self.soft_update(self.critic_model, self.target_critic_model, tau)

    def compute_critic_loss(self, expected_q, target_q):
        return F.mse_loss(expected_q, target_q)

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)
    
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, lstm_hidden_dim, num_lstm_layers, dropout_rate):
        super(Actor, self).__init__()

        # Define LSTM layer
        self.lstm = nn.LSTM(input_size=state_dim, hidden_size=lstm_hidden_dim, num_layers=num_lstm_layers, batch_first=True)

        # Define dropout layer
        self.dropout = nn.Dropout(dropout_rate)
        
        self.bn1 = nn.BatchNorm1d(400)
        self.bn2 = nn.BatchNorm1d(300)
        
        # Define fully connected layers
        self.fc1 = nn.Linear(lstm_hidden_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, action_dim)
        
        self.max_action = max_action

    def forward(self, state):
        # Assume state input is of shape (batch_size, sequence_length, state_dim)
        # LSTM output shape is (batch_size, sequence_length, hidden_dim)
        _, (lstm_out, _) = self.lstm(state)
        
        # Use the final hidden state from LSTM output
        # Note: You might need to reshape the LSTM output depending on your PyTorch version
        lstm_out = lstm_out[-1]
        
        a = F.relu(self.dropout(self.bn1(self.fc1(lstm_out))))
        a = F.relu(self.dropout(self.bn2(self.fc2(a))))
        a = self.max_action * torch.tanh(self.fc3(a))  # Scale the output to match the action space
        
        return a 
    
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, lstm_hidden_dim, num_lstm_layers, dropout_rate):
        super(Critic, self).__init__()

        # Define LSTM layer
        self.lstm = nn.LSTM(input_size=state_dim + action_dim, hidden_size=lstm_hidden_dim, num_layers=num_lstm_layers, batch_first=True)
        
        # Define dropout layer
        self.dropout = nn.Dropout(dropout_rate)
        
        self.bn1 = nn.BatchNorm1d(400)
        self.bn2 = nn.BatchNorm1d(300)
        
        # Define fully connected layers
        self.fc1 = nn.Linear(lstm_hidden_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, 1)  # The Critic network outputs a single value

    def forward(self, state, action):
        # Assume state and action inputs are of shape (batch_size, sequence_length, state_dim) and (batch_size, sequence_length, action_dim) respectively
        # Concatenate state and action along the feature dimension
        state_action = torch.cat([state, action], dim=-1)

        # LSTM output shape is (batch_size, sequence_length, hidden_dim)
        _, (lstm_out, _) = self.lstm(state_action)
        
        # Use the final hidden state from LSTM output
        # Note: You might need to reshape the LSTM output depending on your PyTorch version
        lstm_out = lstm_out[-1]

        q = F.relu(self.dropout(self.bn1(self.fc1(lstm_out))))
        q = F.relu(self.dropout(self.bn2(self.fc2(q))))
        q = self.fc3(q)  # The output is a Q-value, so no activation function is applied at the last layer

        return q
    
class OUNoise:
    def __init__(self, action_dim, mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.3, decay_period=100000):
        self.mu           = mu
        self.theta        = theta
        self.sigma        = max_sigma
        self.max_sigma    = max_sigma
        self.min_sigma    = min_sigma
        self.decay_period = decay_period
        self.action_dim   = action_dim
        self.reset()
        
    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu
        
    def evolve_state(self):
        x  = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state
    
    def get_action(self, action, t=0):
        state = state.drop('Date', axis=0)  # drop the 'Date' column
        state = torch.tensor(state.values, dtype=torch.float32)  # convert to a tensor
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        return action + ou_state
    
class ReplayBuffer:
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)

    def add(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)

        state, action, reward, next_state, done = zip(*batch)

        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)