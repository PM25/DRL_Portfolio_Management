from mylib.agent import Agent
from mylib import data

import sys
import numpy as np


if(len(sys.argv) == 1) :
	stock_path = input("Stock File Path:")
	window_size = int(input("Window Size:"))
	episode_size = int(input("Episodes:"))
elif(len(sys.argv) == 4):
	stock_path = sys.argv[1]
	window_size = int(sys.argv[2])
	episode_size = int(sys.argv[3])
else:
	sys.exit("Error: Wrong number of parameter.")



agent = Agent(window_size)

stockdata = data.StockData(stock_path)
stock_data = stockdata.stock_data

for episode_step in range(episode_size):
	state = data.get_state(stockdata.data, 0, window_size)

	for sample_step in range(1, stockdata.sample_size):
		reward = agent.money - 10000
		done = True if (sample_step != stockdata.sample_size-1) else False
		next_state = data.get_state(stockdata.data, sample_step, window_size)
		close_price = stockdata.data[sample_step][1]

		action = agent.choose_action(state)
		if(action == 0): # Sit
			pass
		elif(action == 1): # Buy
			money = agent.buy(close_price)
			if(money != False):
				reward = money - 10000
		elif(action == 2): # Sell
			money = agent.sell(close_price)
			if(money != False):
				reward = money - 10000
		agent.deep_q_learning(state, reward, action, next_state, done)
	print("Total Reward", reward)
	agent.reset()