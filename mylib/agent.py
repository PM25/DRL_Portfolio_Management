from keras import models
from keras import layers
from keras import optimizers

import numpy as np
import random
from collections import deque


class Agent:
	def __init__(self, state_size, model_name=""):
		self.feature_size = 7
		self.state_size = state_size
		self.action_size = 3 # 0:Sit, 1:Buy, 2:Sell
		self.memory = deque(maxlen=1000)
		self.epsilon = 0.01
		self.gamma = 0.9
		self.money = 10000
		self.buy_history = deque(maxlen=1000)
		self.hold_stock = 0

		if(model_name == ""):
			self.model = self.create_model()
		else:
			self.model = models.load_model("models/" + model_name)


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

		actions = self.model.predict(state)
		return np.argmax(actions[0])


	def buy(self, price, amount=1):
		if(self.money < (price * amount)):
			return False
		else:
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


	def deep_q_learning(self, state, reward, action, next_state, done):
		if(not done):
			q_value = reward + self.gamma * np.max(self.model.predict(next_state)[0])
		else:
			q_value = reward

		t = self.model.predict(state)
		t[0][action] = q_value
		self.model.fit(state, t, verbose=0)


	def reset(self):
		self.money = 10000
		self.memory.clear()
		self.buy_history.clear()
		self.hold_stock = 0