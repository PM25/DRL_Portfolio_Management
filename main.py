import sys
from mylib.agent import Agent
from mylib import data

if (len(sys.argv) == 1):
    stock_path = input("Stock File Path:")
    if (stock_path == ""): stock_path = "data/validation/2002.TW.csv"
    model_name = input("Model Name:")
    if (model_name == ""): model_name = "episode_10"
elif (len(sys.argv) == 4):
    stock_path = sys.argv[1]
    model_name = sys.argv[2]
else:
    sys.exit("Error: Wrong number of parameter.")

window_size = 10
agent = Agent(window_size, model_name)
model = agent.model

stockdata = data.StockData(stock_path)
stock_data = stockdata.stock_data


state = data.get_state(stockdata.data, 0, window_size)
buy_count = 0
sell_count = 0
for sample_step in range(1, stockdata.sample_size):
    reward = agent.money - agent.base_money
    done = True if (sample_step != stockdata.sample_size - 1) else False
    next_state = data.get_state(stockdata.data, sample_step, window_size)
    close_price = float(stockdata.stock_data[sample_step][1])

    action = agent.choose_action(state)
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
    agent.deep_q_learning(state, reward, action, next_state, done)
print("Total Reward", reward)
