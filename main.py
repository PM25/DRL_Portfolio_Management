from mylib.agent import Agent
from mylib.data import StockData
from mylib import graph

import numpy as np
import argparse
import json
import os


parser = argparse.ArgumentParser(description="Portfolio Management Model Training")
parser.add_argument("--train", "-t", type=str, default="data/test/2002.TW.csv", help="Path to stock training data.")
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
    feature_size = stock_data.feature_size + 2
    agent = Agent(args.window, feature_size, model_path)
else:
    stock_data = StockData(args.train)
    mean = stock_data.mean
    std = stock_data.std
    feature_size = stock_data.feature_size + 2
    agent = Agent(args.window, feature_size)


stock_raw_data = stock_data.raw_data
out_data = {"std": list(std), "mean": list(mean)}


buy_count = 0
sell_count = 0
reward = 0
total_buy = 0
price_history = []
action_history = []
extra_features = [[agent.hold_stock, agent.money]]
next_state = stock_data.get_state(0, args.window, extra_features)

for sample_step in range(0, stock_data.sample_size):
    done = True if (sample_step == stock_data.sample_size - 1) else False
    state = next_state
    close_price = float(stock_data.raw_data[sample_step][4])
    date = stock_data.raw_data[sample_step][0]
    price_history.append((date, close_price))

    # Choose Action
    action = agent.choose_action(state)
    action_history.append(action)
    if (action == 0):  # Sit
        reward -= 0
    elif (action == 1):  # Buy
        money = agent.buy(close_price)
        if (money != False):
            buy_count += 1
            reward = agent.money - agent.base_money
            total_buy += close_price
        else:
            reward -= 1
    elif (action == 2):  # Sell
        money = agent.sell(close_price)
        if (money != False):
            sell_count += 1
            reward = agent.money - agent.base_money
        else:
            reward -= 10

    if(sample_step == stock_data.sample_size - 1):
        while(agent.sell(close_price)): pass
        reward = agent.money - agent.base_money

    extra_features.append([agent.hold_stock, agent.money])
    while (len(extra_features) > agent.window_size):
        extra_features.pop(0)
    if (sample_step != stock_data.sample_size - 1):
        next_state = stock_data.get_state(sample_step + 1, args.window, extra_features)

    agent.store_q_value(state, action, reward, next_state, done)

print("BUY {}, SELL {}".format(buy_count, sell_count))
print("Total Earn: {:.1f} ({:.2%})".format(reward, reward/total_buy))
graph.draw_stock_predict(price_history, action_history)