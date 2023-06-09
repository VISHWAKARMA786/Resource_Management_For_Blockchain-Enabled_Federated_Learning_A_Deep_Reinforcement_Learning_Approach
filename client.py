import tensorflow as tf
from network import DQN
import socket
import os

class Client:
    def __init__(self, socket, state_shape, action_space):
        self.socket = socket
        self.model = DQN(state_shape, action_space)

    def local_training(self, local_data):
        optimizer = tf.keras.optimizers.Adam()
        loss_fn = tf.keras.losses.MeanSquaredError()

        inputs, targets = local_data

        # Define the number of epochs
        num_epochs = 20

        for epoch in range(num_epochs):
            with tf.GradientTape() as tape:
                # Get inputs and targets from the local dataset
                # inputs, targets = self.get_local_data()

                # Forward pass
                q_values = self.model(inputs)

                # Calculate the loss
                loss = loss_fn(targets, q_values)

            # Backpropagation and update the model weights
            gradients = tape.gradient(loss, self.model.trainable_variables)
            optimizer.apply_gradients(
                zip(gradients, self.model.trainable_variables))

    # Implement the logic to retrieve local data for training
    def get_local_data(self):
        inputs = tf.random.normal((batch_size,) + input_shape)
        targets = tf.random.normal((batch_size, action_space))
        return inputs, targets

    # Implement the logic to save and send model weights to the central server
    def send_model_weights(self, model):
        model.save("temp_model.h5")
        with open("temp_model.h5", "rb") as f:
            serialized_model = f.read()
        self.socket.sendall(serialized_model)

    # Implement the logic to receive and load model weights from the central server
    def receive_model_weights(self):
        received_data = self.socket.recv(1024)
        with open("temp_model.h5", "wb") as f:
            f.write(received_data)
        received_model = tf.keras.models.load_model("temp_model.h5")
        return received_model


if __name__ == "__main__":
    state_shape = (84, 84, 4)
    action_space = 4
    batch_size = 5
    input_shape = (9,)  # Convert input_shape into a tuple

    # create a socket and Connect to the central server 
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_address = ('localhost', 8888)
    client_socket.connect(server_address)

    # create an instance for the client 
    client = Client(client_socket, state_shape, action_space)

    # Dummy Data for Local training 
    dummy_inputs = tf.random.normal((batch_size,) + input_shape)
    dummy_targets = tf.random.normal((batch_size, action_space))
    dummy_local_data = (dummy_inputs, dummy_targets)

    #  Perform Local Training 
    client.local_training(dummy_local_data)

    # Get updated model weight 
    updated_weights = client.model

    # Send the updated weights to the central server 
    client.send_model_weights(updated_weights)

    received_model = client.receive_model_weights()
    client.model = received_model

    # close the client socket 
    client.socket.close()

    # Clean up the temporary model file
    os.remove("temp_model.h5")
