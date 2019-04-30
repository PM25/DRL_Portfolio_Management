from mylib.agent import Agent
from mylib import data

import sys
import numpy as np


if(len(sys.argv) == 1) :
	stock_path = input("Stock File Path:")
	if(stock_path == ""): stock_path = "data/train/2002.TW.csv"
	window_size = int(input("Window Size:"))
	if(window_size == 0): window_size = 10
	episode_size = int(input("Episodes:"))
	if(episode_size == 0): episode_size = 500
	model_name = input("Model Name:")
	if(model_name == ""): model_name = "episode_20"
elif(len(sys.argv) == 4):
	stock_path = sys.argv[1]
	window_size = int(sys.argv[2])
	episode_size = int(sys.argv[3])
	model_name = sys.argv[4]
else:
	sys.exit("Error: Wrong number of parameter.")


agent = Agent(window_size, model_name)

stockdata = data.StockData(stock_path)
stock_data = stockdata.raw_data

for episode_step in range(1, episode_size+1):
	state = stockdata.get_state(0, window_size)

	buy_count = 0
	sell_count = 0
	for sample_step in range(1, stockdata.sample_size):
		reward = agent.money - agent.base_money
		done = True if (sample_step != stockdata.sample_size-1) else False
		next_state = stockdata.get_state(sample_step, window_size)
		close_price = float(stockdata.raw_data[sample_step][1])

		action = agent.choose_action(state)
		if(action == 0): # Sit
			pass
		elif(action == 1): # Buy
			money = agent.buy(close_price)
			if(money != False):
				reward = money - agent.base_money
				buy_count += 1
		elif(action == 2): # Sell
			money = agent.sell(close_price)
			if(money != False):
				reward = money - agent.base_money
				sell_count += 1
		agent.deep_q_learning(state, reward, action, next_state, done)
	print("BUY", buy_count, ", SELL", sell_count)
	print("Total Reward", reward)
	agent.reset()
	if(episode_step % 10 == 0):
		agent.save_model("episode_{}".format(episode_step))