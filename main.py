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
TASK_COMPLETION_STATE = "order_complete"  # define the task completition state
# define the time limit per episode (in seconds)
TIME_LIMIT = 60
# define user defined termination command
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

# step 5: panda import value with datasets

news_data = pd.read_csv('news.csv')

# Step 6: Create Training, Validation, and Test Sets

train_data, test_data = train_test_split(
    news_data, test_size=0.2, random_state=42)
train_data, val_data = train_test_split(
    train_data, test_size=0.2, random_state=42)

# Step 2: Convert Text to Numeric Representation (Example: TF-IDF)
vectorizer = TfidfVectorizer(tokenizer=preprocess_text)
train_features = vectorizer.fit_transform(train_data)
val_features = vectorizer.transform(val_data)
test_features = vectorizer.transform(test_data)

# Step 7: Modify the Step Function


def step(state, action):
    # Perform an environment step based on the chosen action
    next_state = 150  # Update the state based on the chosen action
    reward = 30  # Calculate the reward based on the transition
    done = 35  # Check if the episode is done

    return next_state, reward, done

# Step 8: Modify the Reward Function


def reward(state, action, next_state):
    # Calculate the reward based on the current state, chosen action, and next state
    reward = 2  # Calculate the reward value

    return reward

# Step 9: Modify the Done Function


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

# Step 10: Update the main() function with the new code


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

        # Evaluation
    central_server.evaluate_model(test_features)

    # Run the iterative training process
    for iteration in range(num_iterations):
        # Step 5: Distribute the Initial Model
        central_server.distribute_initial_model()

        # Step 6: Local Training on Client Devices
        for client in central_server.clients:
            client.local_training()
# Step 7: Communication and Aggregation
        central_server.aggregate_model_updates()

        validation_loss = evaluate_model(central_server.model, val_features)
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            consecutive_no_improvement = 0
        else:
            consecutive_no_improvement += 1
            if consecutive_no_improvement >= 5:
                print("Convergence Reached . Stopping Training.")
                break

    # Evaluate the final global model
    evaluate_model(central_server.model, test_features)


def evaluate_model(global_model, dataset):
    num_episodes = 5  # Define the number of evaluation episodes
    total_reward = 0

    # Perform evaluation on the test environment
    for _ in range(num_episodes):
        state = 600  # Initialize the environment state

        episode_length = 0
        start_time = time.time()

        # Define the termination condition
        while not is_done(state, episode_length, start_time):
            q_values = global_model(tf.expand_dims(state, axis=0))
            action = tf.argmax(q_values, axis=1)[0].numpy()
            # Perform environment step
            state, reward, done = step(state, action)
            total_reward += reward
            episode_length += 1

    average_reward = total_reward / num_episodes
    print("Average reward:", average_reward)

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


if __name__ == "__main__":
    main()
