import tensorflow as tf

class DQN(tf.keras.Model):
    def __init__(self, state_shape, action_space):
        super(DQN, self).__init__()
        self.state_shape = state_shape
        self.action_space = action_space

        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(64, activation='relu')
        self.output_layer = tf.keras.layers.Dense(action_space, activation='linear')

    def call(self, inputs):
        x = self.flatten(inputs)
        x = self.dense1(x)
        x = self.dense2(x)
        outputs = self.output_layer(x)
        return outputs


class DDQN(tf.keras.Model):
    def __init__(self, state_shape, action_space):
        super(DDQN, self).__init__()
        self.state_shape = state_shape
        self.action_space = action_space

        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(64, activation='relu')
        self.value_output_layer = tf.keras.layers.Dense(1, activation='linear')
        self.advantage_output_layer = tf.keras.layers.Dense(action_space, activation='linear')

    def call(self, inputs):
        x = self.flatten(inputs)
        x = self.dense1(x)
        x = self.dense2(x)
        value = self.value_output_layer(x)
        advantage = self.advantage_output_layer(x)
        q_values = value + (advantage - tf.reduce_mean(advantage, axis=1, keepdims=True))
        return q_values
        
        pass


class DDPG(tf.keras.Model):
    def __init__(self, state_shape, action_space):
        super(DDPG, self).__init__()
        self.state_shape = state_shape
        self.action_space = action_space

        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(64, activation='relu')
        self.actor_output_layer = tf.keras.layers.Dense(action_space, activation='tanh')
        self.critic_output_layer = tf.keras.layers.Dense(1, activation='linear')

    def call(self, inputs):
        x = self.flatten(inputs)
        x = self.dense1(x)
        x = self.dense2(x)
        actor_output = self.actor_output_layer(x)
        critic_output = self.critic_output_layer(x)
        return actor_output, critic_output

class QNetwork(tf.keras.Model):
    def __init__(self, state_shape, action_space):
        super(QNetwork, self).__init__()
        self.state_shape = state_shape
        self.action_space = action_space

        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(64, activation='relu')
        self.output_layer = tf.keras.layers.Dense(action_space, activation='linear')

    def call(self, inputs):
        x = self.flatten(inputs)
        x = self.dense1(x)
        x = self.dense2(x)
        q_values = self.output_layer(x)
        return q_values


class RandomPolicy:
    def select_action(self, q_values):
        num_actions = q_values.shape[-1]
        return tf.random.uniform(shape=(), minval=0, maxval=num_actions, dtype=tf.int32)


class GreedyPolicy:
    def select_action(self, q_values):
        return tf.argmax(q_values, axis=-1)
