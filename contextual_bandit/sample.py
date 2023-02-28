import numpy as np

class ContextBandit:
    def __init__(self, num_arms, num_features):
        self.num_arms = num_arms
        self.num_features = num_features
        self.arms = np.random.rand(num_arms, num_features)
        self.theta = np.zeros(num_features)
    
    def pull(self, x):
        # Calculate the predicted reward for each arm using the context
        estimates = np.dot(self.arms, self.theta)
        
        # Choose the arm with the highest estimated reward
        chosen_arm = np.argmax(estimates)
        
        # Generate a random reward based on the chosen arm and the context
        reward = np.random.normal(np.dot(self.arms[chosen_arm], x), 1)
        
        # Update the parameter theta to improve future predictions
        self.theta += (reward - estimates[chosen_arm]) * self.arms[chosen_arm]
        
        return reward

# Define the function to run the experiment
def run_experiment(num_arms, num_features, num_episodes):
    bandit = ContextBandit(num_arms, num_features)
    rewards = np.zeros(num_episodes)
    
    for i in range(num_episodes):
        # Generate a random context for each episode
        x = np.random.rand(num_features)
        
        # Pull the arm and update the rewards
        rewards[i] = bandit.pull(x)
    
    # Calculate the cumulative reward over all episodes
    cumulative_rewards = np.cumsum(rewards)
    
    return cumulative_rewards

# Run the experiment with 3 arms and 2 features for 1000 episodes
cumulative_rewards = run_experiment(3, 2, 1000)

# Print the cumulative reward at each episode
print(cumulative_rewards)
