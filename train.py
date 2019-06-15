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

<<<<<<< Updated upstream
# Default Data
if (args.train == None): args.train = "data/train/2002.TW.csv"
if (args.window == None): args.window = 10
if (args.episode == None): args.episode = 1000
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
        start_episode = metadata["episode"] + 1
        stockdata = data.StockData(args.train, mean, std)

stock_data = stockdata.raw_data
mean = stockdata.mean
std = stockdata.std
out_data = {}
out_data["std"] = list(std)
out_data["mean"] = list(mean)
=======
# Load Previous trained model.
model_base_path = "models"
meta_path = os.path.join(model_base_path, "metadata.json")

if (os.path.isfile(meta_path)):
    with open("models/metadata.json", 'r') as meta_file:
        meta_data = json.load(meta_file)

    mean = np.array(meta_data["mean"])
    std = np.array(meta_data["std"])
    start_episode = meta_data["episode"] + 1
    args.window = meta_data["window"]

    stock_data = StockData(args.train, mean, std)
    if(args.model != None):
        model_path = os.path.join(model_base_path, args.model)
    else:
        model_path = os.path.join(model_base_path, meta_data["model"])
    feature_size = stock_data.feature_size
    agent = Agent(args.window, feature_size, model_path, base_money=args.money)
else:
    stock_data = StockData(args.train)
    mean = stock_data.mean
    std = stock_data.std
    start_episode = 0
    feature_size = stock_data.feature_size
    agent = Agent(args.window, feature_size, base_money=args.money)


stock_raw_data = stock_data.raw_data
out_data = {"std": list(std), "mean": list(mean)}
>>>>>>> Stashed changes
end_episode = start_episode + args.episode + 1
max_reward = 0


<<<<<<< Updated upstream
=======



>>>>>>> Stashed changes
for episode_step in range(start_episode, end_episode):
    next_state = stockdata.get_state(0, args.window)

    buy_count = 0
    sell_count = 0
    reward = 0
    total_buy = 0
    for sample_step in range(0, stockdata.sample_size):
        done = True if (sample_step == stockdata.sample_size - 1) else False
        state = next_state
<<<<<<< Updated upstream
        if(sample_step != stockdata.sample_size - 1):
            next_state = stockdata.get_state(sample_step+1, args.window)
        close_price = float(stockdata.raw_data[sample_step][4])
=======
        close_price = float(stock_data.raw_data[sample_step][1])
>>>>>>> Stashed changes

        action = agent.choose_action(state)
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

        if(sample_step == stockdata.sample_size - 1):
            while(agent.sell(close_price)): pass
            reward = agent.money - agent.base_money
        agent.store_q_value(state, action, reward, next_state, done)
    print("BUY {}, SELL {}".format(buy_count, sell_count))
    print("Total Reward: {:.1f} ({:.2%})".format(reward, reward/total_buy))
    print()
    agent.reset()
<<<<<<< Updated upstream
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
=======

    if (reward > max_reward):
        max_reward = reward
        model_name = "episode_{}".format(episode_step)
        agent.save_model(model_name, episode_step, list(std), list(mean))
>>>>>>> Stashed changes
