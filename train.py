from mylib.agent import Agent
from mylib import data

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

# Default Data
if (args.train == None): args.train = "data/train/2002.TW.csv"
if (args.window == None): args.window = 10
if (args.episode == None): args.episode = 1000
if (args.model == None):
    if(not os.path.isfile("models/metadata.json")):
        start_episode = 1
        stockdata = data.StockData(args.train)
        feature_size = stockdata.feature_size + 2 # Hold stock & Possess money
        agent = Agent(args.window, feature_size)
    else:
        with open("models/metadata.json", 'r') as in_file:
            metadata = json.load(in_file)
        std = np.array(metadata["std"])
        mean = np.array(metadata["mean"])
        start_episode = metadata["episode"] + 1
        stockdata = data.StockData(args.train, mean, std)
        feature_size = stockdata.feature_size + 2
        agent = Agent(args.window, feature_size, metadata["model"])

stock_data = stockdata.raw_data
mean = stockdata.mean
std = stockdata.std
out_data = {}
out_data["std"] = list(std)
out_data["mean"] = list(mean)
end_episode = start_episode + args.episode + 1


for episode_step in range(start_episode, end_episode):
    extra_features = [[agent.hold_stock, agent.money]]
    next_state = stockdata.get_state(0, args.window, extra_features)

    buy_count = 0
    sell_count = 0
    reward = 0
    total_buy = 0
    for sample_step in range(0, stockdata.sample_size):
        done = True if (sample_step == stockdata.sample_size - 1) else False
        state = next_state
        close_price = float(stockdata.raw_data[sample_step][4])

        action = agent.choose_action(state)
        if (action == 0):  # Sit
            reward -= 1
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

        if(sample_step == stockdata.sample_size - 1):
            while(agent.sell(close_price)): pass
            reward = agent.money - agent.base_money

        extra_features.append([agent.hold_stock, agent.money])
        while(len(extra_features) > agent.window_size):
            extra_features.pop(0)
        if(sample_step != stockdata.sample_size - 1):
            next_state = stockdata.get_state(sample_step+1, args.window, extra_features)

        agent.store_q_value(state, action, reward, next_state, done)
    print("BUY {}, SELL {}".format(buy_count, sell_count))
    print("Total Reward: {:.1f} ({:.2%})".format(reward, reward/total_buy))
    print()
    agent.reset()
    if (episode_step % 10 == 0 and episode_step != 0):
        print('-' * 10)
        print("Save Model: episode_{}".format(episode_step))
        print('-' * 10)
        agent.save_model("episode_{}".format(episode_step))

        out_data["model"] = "episode_{}".format(episode_step)
        out_data["episode"] = episode_step
        out_data["std"] = list(std)
        out_data["mean"] = list(mean)
        with open("models/metadata.json", 'w') as out_file:
            json.dump(out_data, out_file, ensure_ascii=False, indent=4)
