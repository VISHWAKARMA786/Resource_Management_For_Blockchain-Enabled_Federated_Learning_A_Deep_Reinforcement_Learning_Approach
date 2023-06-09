import tensorflow as tf
from central_server import CentralServer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from client import Client
import matplotlib.pyplot as plt
import time
import numpy as np
import pandas as pd


MAX_EPISODE_LENGTH = 10                   # define the max episode length
TASK_COMPLETION_STATE = "order_complete"  # define the task completion state
# define the time limit per episode (in seconds)
TIME_LIMIT = 60
# define user-defined termination command
TERMINATION_COMMAND = "terminate"


# Step 1: Preprocess Text Data (Example: Tokenization)
def preprocess_text(text):
    # Perform tokenization
    tokens = text.split()
    return tokens


# step 3: Define State Shape
state_shape = (100, 20)  # Examples values

# Step 4: Define Action Space
action_space = 400  # num_actions  # Example value

# Step 5: Load and Split Dataset
news_data = pd.read_csv('abcnews-date-text.csv')
train_data, test_data = train_test_split(news_data, test_size=0.2, random_state=42)
train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)

# Step 2: Convert Text to Numeric Representation (Example: TF-IDF)
vectorizer = TfidfVectorizer(tokenizer=preprocess_text)
train_features = vectorizer.fit_transform(train_data)
val_features = vectorizer.transform(val_data)
test_features = vectorizer.transform(test_data)


# Step 6: Define Step, Reward, and Termination Functions
def step(state, action):
    # Perform an environment step based on the chosen action
    next_state = 150  # Update the state based on the chosen action
    reward = 30  # Calculate the reward based on the transition
    done = 35  # Check if the episode is done

    return next_state, reward, done


def reward(state, action, next_state):
    # Calculate the reward based on the current state, chosen action, and next state
    reward = 2  # Calculate the reward value

    return reward


def is_done(state, episode_length, start_time):
    if episode_length >= MAX_EPISODE_LENGTH:
        return True
    if state == TASK_COMPLETION_STATE:
        return True
    current_time = time.time()
    if current_time - start_time >= TIME_LIMIT:
        return True
    if state.lower() == TERMINATION_COMMAND:
        return True
    return False


# Step 7: Modify the main() function with the new code
def main():
    num_iterations = 100

    # Create an instance of the central server
    central_server = CentralServer(state_shape, action_space)
    central_server.start()

    # Distribute initial model to clients
    for client in central_server.clients:
        client.set_model(central_server.model)

    best_validation_loss = float('inf')
    consecutive_no_improvement = 0

    # Training loop
    for iteration in range(num_iterations):
        central_server.aggregate_model_updates()

        validation_loss = evaluate_model(central_server.model, val_features)
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            consecutive_no_improvement = 0
        else:
            consecutive_no_improvement += 1
            if consecutive_no_improvement >= 5:
                print("Convergence Reached. Stopping Training.")
                break

    # Evaluate the final global model
    test_loss = evaluate_model(central_server.model, test_features)
    print("Test Loss:", test_loss)


def evaluate_model(model, features):
    # Evaluate the model on the provided features
    # Return the loss or any other metrics of interest
    return 0.0


if __name__ == "__main__":
    main()
