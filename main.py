from mylib.agent import Agent
from mylib.data import StockData
from mylib import graph

import numpy as np
import argparse
import json
import os


parser = argparse.ArgumentParser(description="Portfolio Management Model Training")
parser.add_argument("--train", "-t", type=str, default="data/train/2002.TW.csv", help="Path to stock training data.")
parser.add_argument("--window", "-w", type=int, default=10, help="Window Size.")
parser.add_argument("--episode", "-e", type=int, default=1000, help="Episode Size.")
parser.add_argument("--model", "-m", type=str, default=None, help="Model Name.")
args = parser.parse_args()

# Load Previous trained model.
model_base_path = "models"
meta_path = os.path.join(model_base_path, "metadata.json")

if (os.path.isfile(meta_path)):
    with open("models/metadata.json", 'r') as meta_file:
        meta_data = json.load(meta_file)
    mean = np.array(meta_data["mean"])
    std = np.array(meta_data["std"])
    stock_data = StockData(args.train, mean, std)
    if(args.model != None):
        model_path = os.path.join(model_base_path, args.model)
    else:
        model_path = meta_data["model"]
    agent = Agent(args.window, meta_data["model"])
else:
    stock_data = StockData(args.train)
    mean = stock_data.mean
    std = stock_data.std
    agent = Agent(args.window)


stock_raw_data = stock_data.raw_data
out_data = {"std": list(std), "mean": list(mean)}


buy_count = 0
sell_count = 0
reward = 0
total_buy = 0
price_history = []
action_history = []
next_state = stock_data.get_state(0, args.window)

for sample_step in range(0, stock_data.sample_size):
    done = True if (sample_step == stock_data.sample_size - 1) else False
    state = next_state
    if(sample_step != stock_data.sample_size - 1):
        next_state = stock_data.get_state(sample_step+1, args.window)
    close_price = float(stock_data.raw_data[sample_step][4])
    date = stock_data.raw_data[sample_step][0]
    price_history.append((date, close_price))

    action = agent.choose_action(state)
    action_history.append(action)
    if (action == 0):  # Sit
        reward -= 1
    elif (action == 1):  # Buy
        money = agent.buy(close_price)
        if (money != False):
            buy_count += 1
            reward = agent.money - agent.base_money
            total_buy += close_price
    elif (action == 2):  # Sell
        money = agent.sell(close_price)
        if (money != False):
            sell_count += 1
            reward = agent.money - agent.base_money
        else:
            reward -= 5

    if(sample_step == stock_data.sample_size - 1):
        while(agent.sell(close_price)): pass
        reward = agent.money - agent.base_money
    agent.store_q_value(state, action, reward, next_state, done)

print("BUY {}, SELL {}".format(buy_count, sell_count))
print("Total Reward: {:.1f} ({:.2%})".format(reward, reward/total_buy))
graph.draw_stock_predict(price_history, action_history)