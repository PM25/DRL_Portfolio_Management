from mylib.agent import Agent
from mylib import data
import argparse
import json
import os
import numpy as np

parser = argparse.ArgumentParser(description="Portfolio Management Predict")
parser.add_argument("--test", "-t", type=str, help="Path to stock test data.")
parser.add_argument("--window", "-w", type=int, help="Window Size.")
parser.add_argument("--model", "-m", type=str, help="Model Name.")
args = parser.parse_args()

# Default Data
if (args.test == None): args.test = "data/test/2002.TW.csv"
if (args.window == None): args.window = 10
if (args.model == None):
    if(not os.path.isfile("models/metadata.json")):
        start_episode = 1
        stockdata = data.StockData(args.train)
        agent = Agent(args.window)
    else:
        with open("models/metadata.json", 'r') as in_file:
            metadata = json.load(in_file)
        agent = Agent(args.window, metadata["model"])
        std = np.array(metadata["std"])
        mean = np.array(metadata["mean"])
        start_episode = metadata["episode"]
        stockdata = data.StockData(args.test, mean, std)


window_size = 10
model = agent.model

stockdata = data.StockData(args.test)
stock_data = stockdata.raw_data


state = stockdata.get_state(0, args.window)
buy_count = 0
sell_count = 0
for sample_step in range(1, stockdata.sample_size):
    reward = agent.money - agent.base_money
    done = True if (sample_step != stockdata.sample_size - 1) else False
    next_state = stockdata.get_state(sample_step, args.window)
    close_price = float(stockdata.raw_data[sample_step][1])

    action = agent.choose_action(state, 0)
    if (action == 0):  # Sit
        pass
    elif (action == 1):  # Buy
        money = agent.buy(close_price)
        if (money != False):
            print("BUY", close_price)
            reward = money - agent.base_money
            buy_count += 1
        else:
            print("Failed BUY")
    elif (action == 2):  # Sell
        money = agent.sell(close_price)
        if (money != False):
            print("SELL", close_price)
            reward = money - agent.base_money
            sell_count += 1
        else:
            print("Failed SELL")
    print("BUY {}, SELL {}".format(buy_count, sell_count))
    print("REWARD", reward)
    agent.deep_q_learning(state, reward, action, next_state, done, False)
print("Total Reward", reward)
