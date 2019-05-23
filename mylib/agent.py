from keras import models
from keras import layers
from keras import optimizers

import numpy as np
import random
from collections import deque
import os

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allocator_type = 'BFC'
config.gpu_options.per_process_gpu_memory_fraction = 0.3
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))


class Agent:
    def __init__(self, state_size, model_name="", buff_size=30):
        self.feature_size = 7
        self.state_size = state_size
        self.action_size = 3 # 0:Sit, 1:Buy, 2:Sell
        self.buff = deque(maxlen=500)
        self.buff_size= buff_size
        self.epsilon = 0.01
        self.gamma = 0.99
        self.base_money = 1000
        self.money = self.base_money
        self.buy_history = deque(maxlen=1000)
        self.hold_stock = 0

        if(model_name == ""):
            self.model = self.create_model()
        else:
            self.model = models.load_model("models/" + model_name)
            print("Model {} Loaded!".format(model_name))


    def create_model(self):
        model = models.Sequential()
        model.add(layers.Dense(units=64, input_dim=self.feature_size, activation="relu"))
        model.add(layers.Dense(units=32, activation="relu"))
        model.add(layers.Dense(units=32, activation="relu"))
        model.add(layers.Dense(self.action_size, activation="linear"))
        model.compile(loss="mse", optimizer=optimizers.Adam(lr=0.001))

        return model


    def update_epsilon(self, epsilon):
        self.epsilon = epsilon


    def choose_action(self, state, epsilon=None):
        if(epsilon == None): epsilon = self.epsilon

        if random.random() <= epsilon:
            return random.randrange(self.action_size)

        action = np.argmax(self.model.predict(state)[0])
        return action


    def buy(self, price, amount=1):
        if(self.money < (price * amount)):
            return False
        else:
            for i in range(amount):
                self.buy_history.append(price)
            self.hold_stock += amount
            self.money -= (price * amount)
            return self.money


    def sell(self, price, amount=1):
        if(self.hold_stock < amount):
            return False
        else:
            self.buy_history.pop()
            self.hold_stock -= amount
            self.money += (price * amount)
            return self.money


    def deep_q_learning(self):
        while(self.buff):
            (state, action, reward, next_state, done) = self.buff.popleft()
            if(not done):
                q_value = reward + self.gamma * np.max(self.model.predict(next_state)[0])
            else:
                q_value = reward

            predict_q_value = self.model.predict(state)
            # if(action != 0):
            #     print(predict_q_value[0][action], q_value, action)
            predict_q_value[0][action] = q_value
            self.model.fit(state, predict_q_value, epochs=1, verbose=0)


    def reset(self):
        self.money = self.base_money
        self.buff.clear()
        self.buy_history.clear()
        self.hold_stock = 0


    def save_model(self, name="model", path="models"):
        self.model.save(os.path.join(path, name))


    def store_q_value(self, state, action, reward, next_state, done):
        self.buff.append((state, action, reward, next_state, done))
        if(len(self.buff) == self.buff_size):
            self.deep_q_learning()