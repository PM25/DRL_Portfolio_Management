from mylib.agent import Agent
from mylib.data import StockData
from mylib import graph

import sys
import numpy as np
import argparse
import json
import os


parser = argparse.ArgumentParser(description="Portfolio Management Model Training")
parser.add_argument("--validation", "-v", type=str, default="data/test/_2002.TW.csv", help="Path to stock data for prediction.")
parser.add_argument("--model", "-m", type=str, default=None, help="Model Name.")
parser.add_argument("--money", type=int, default=1000, help="Initial given money")
args = parser.parse_args()

# Load Previous trained model.
model_base_path = "models"
meta_path = os.path.join(model_base_path, "metadata.json")

if (os.path.isfile(meta_path)):
    with open("models/metadata.json", 'r') as meta_file:
        meta_data = json.load(meta_file)
    mean = np.array(meta_data["mean"])
    std = np.array(meta_data["std"])
    stock_data = StockData(args.validation, mean, std)
    window_size = int(meta_data["window"])
    if(args.model != None):
        model_path = os.path.join(model_base_path, args.model)
    else:
        model_path = os.path.join(model_base_path, meta_data["model"])
    feature_size = stock_data.feature_size + 2
    agent = Agent(window_size, feature_size, model_path, base_money=args.money)
else:
    sys.exit("Error: Can't Find Model!")


stock_raw_data = stock_data.raw_data
out_data = {"std": list(std), "mean": list(mean)}


buy_count = 0
sell_count = 0
reward = 0
total_buy = 0
price_history = []
action_history = []
money_hold_history = [agent.money]
property_history = [agent.money]
stock_hold_history = [0]
extra_features = [[agent.hold_stock, agent.money]]
next_state = stock_data.get_state(0, window_size, extra_features)

for sample_step in range(0, stock_data.sample_size):
    done = True if (sample_step == stock_data.sample_size - 1) else False
    state = next_state
    close_price = float(stock_data.raw_data[sample_step][2])
    date = stock_data.raw_data[sample_step][1]
    price_history.append((date, close_price))

    # Choose Action
    action = agent.choose_action(state, 0)
    action_history.append(action)
    if (action == 0):  # Sit
        reward -= 0.8
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
            reward -= 7

    if(sample_step == stock_data.sample_size - 1):
        while(agent.sell(close_price)): pass
        reward = agent.money - agent.base_money

    money_hold_history.append(agent.money)
    property_history.append(agent.money + (agent.hold_stock * close_price))
    stock_hold_history.append(agent.hold_stock)

    extra_features.append([agent.hold_stock, agent.money])
    while (len(extra_features) > agent.window_size):
        extra_features.pop(0)
    if (sample_step != stock_data.sample_size - 1):
        next_state = stock_data.get_state(sample_step + 1, window_size, extra_features)

    agent.store_q_value(state, action, reward, next_state, done)

print("BUY {}, SELL {}".format(buy_count, sell_count))
print("Total Earn: {:.1f} ({:.2%})".format(reward, reward/total_buy))
graph.draw_stock_predict(price_history, action_history, False)
graph.draw_info(money_hold_history, property_history, stock_hold_history)