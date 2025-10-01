import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from collections import deque
import random
import gymnasium as gym

class CarDQNAgent:
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float = 0.001,
        initial_epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        final_epsilon: float = 0.01,
        discount_factor: float = 0.95,
        buffer_size: int = 10000,
        batch_size: int = 64,
        target_update_freq: int = 200
    ):
        """Initialize DQN agent."""
        self.env = env
        self.state_size = 3  # x, y, theta
        self.action_size = env.action_space.n
        
        # Build neural networks
        self.q_network = self._build_model(learning_rate)
        self.target_network = self._build_model(learning_rate)
        self.update_target_network()
        
        # Replay buffer
        self.replay_buffer = deque(maxlen=buffer_size)
        
        # Training parameters
        self.batch_size = batch_size
        self.discount_factor = discount_factor
        self.target_update_freq = target_update_freq
        
        # Exploration parameters
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        
        # Tracking
        self.steps = 0
        self.training_losses = []
    
    def _build_model(self, learning_rate):
        """Build the Q-network architecture."""
        model = keras.Sequential([
            layers.Input(shape=(self.state_size,)),
            layers.Dense(128, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(self.action_size, activation='linear')  # Q-values
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='mse'
        )
        return model
    
    def preprocess_obs(self, obs):
        """Convert dict observation to normalized numpy array."""
        x = obs["agent"][0] / self.env.size  # Normalize to [0, 1]
        y = obs["agent"][1] / self.env.size
        theta = obs["theta"][0] / (2 * np.pi)  # Normalize to [0, 1]
        
        return np.array([x, y, theta], dtype=np.float32)
    
    def get_action(self, obs: dict) -> int:
        """Choose action using epsilon-greedy strategy."""
        # Exploration
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        
        # Exploitation
        state = self.preprocess_obs(obs)
        state = np.expand_dims(state, axis=0)  # Add batch dimension
        
        q_values = self.q_network.predict(state, verbose=0)
        return np.argmax(q_values[0])
    
    def remember(self, obs, action, reward, next_obs, done):
        """Store experience in replay buffer."""
        state = self.preprocess_obs(obs)
        next_state = self.preprocess_obs(next_obs)
        self.replay_buffer.append((state, action, reward, next_state, done))
    
    def replay(self):
        """Train the model on a batch of experiences."""
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # Sample batch from buffer
        batch = random.sample(self.replay_buffer, self.batch_size)
        
        # Prepare batch arrays
        states = np.array([e[0] for e in batch])
        actions = np.array([e[1] for e in batch])
        rewards = np.array([e[2] for e in batch])
        next_states = np.array([e[3] for e in batch])
        dones = np.array([e[4] for e in batch])
        
        # Current Q-values
        current_q_values = self.q_network.predict(states, verbose=0)
        
        # Next Q-values from target network
        next_q_values = self.target_network.predict(next_states, verbose=0)
        
        # Calculate targets
        targets = current_q_values.copy()
        for i in range(self.batch_size):
            if dones[i]:
                targets[i][actions[i]] = rewards[i]
            else:
                targets[i][actions[i]] = rewards[i] + self.discount_factor * np.max(next_q_values[i])
        
        # Train the network
        loss = self.q_network.fit(states, targets, epochs=1, verbose=0, batch_size=self.batch_size)
        self.training_losses.append(loss.history['loss'][0])
        
        # Update steps counter
        self.steps += 1
        
        # Update target network periodically
        if self.steps % self.target_update_freq == 0:
            self.update_target_network()
    
    def update_target_network(self):
        """Copy weights from main network to target network."""
        self.target_network.set_weights(self.q_network.get_weights())
    
    def decay_epsilon(self):
        """Reduce exploration rate after each episode."""
        self.epsilon = max(self.final_epsilon, self.epsilon * self.epsilon_decay)
    
    def train_episode(self):
        """Run one complete episode of training."""
        obs, info = self.env.reset()
        total_reward = 0
        done = False
        steps = 0
        
        while not done and steps < 1000:  # Max steps per episode
            # Choose and execute action
            action = self.get_action(obs)
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            # Store experience
            self.remember(obs, action, reward, next_obs, done)
            
            # Learn from experience
            self.replay()
            
            # Update state
            obs = next_obs
            total_reward += reward
            steps += 1
        
        # Decay epsilon after episode
        self.decay_epsilon()
        
        return total_reward, steps

# Training loop
def train_agent(env, n_episodes=1000):
    """Main training loop."""
    agent = CarDQNAgent(env)
    rewards_history = []
    
    for episode in range(n_episodes):
        episode_reward, steps = agent.train_episode()
        rewards_history.append(episode_reward)
        
        # Print progress
        if episode % 100 == 0:
            avg_reward = np.mean(rewards_history[-100:]) if len(rewards_history) >= 100 else np.mean(rewards_history)
            print(f"Episode {episode}, Average Reward: {avg_reward:.2f}, Epsilon: {agent.epsilon:.3f}")
    
    return agent, rewards_history

