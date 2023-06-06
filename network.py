import tensorflow as tf

# Define the DQN network architecture
class DQN(tf.keras.Model):
    def __init__(self, state_shape, action_space):
        super(DQN, self).__init__()
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(64, activation='relu')
        self.output_layer = tf.keras.layers.Dense(action_space)

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        q_values = self.output_layer(x)
        return q_values
