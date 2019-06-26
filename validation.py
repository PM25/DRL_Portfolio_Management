import argparse
from collections import Counter

from mylib.stock import Stock
from mylib.environment import Environment
from mylib.actor import Actor
from mylib.critic import Critic
from mylib.graph import Graph

import random


# Arguments
parser = argparse.ArgumentParser(description="Portfolio Management Model Training (Actor Critic)")
parser.add_argument("--data", "-d", type=str, default="data", help="Path to stock training data.")
# parser.add_argument("--model", "-m", type=str, help="Model Name.", required=True)
parser.add_argument("--model", "-m", type=str, default="0.pkl", help="Model Name.")
parser.add_argument("--cash", type=int, default=2000, help="Initial given cash.")
args = parser.parse_args()

# Start from here!
if __name__ == "__main__":
    # Environment (Stock)
    stock = Stock(args.data)
    files_name = stock.get_files_name()
    files_df = stock.read_files(files_name[0:5], split_ratio=[.7, .3])[1]
    actor_cash = args.cash / len(files_df)
    graph = Graph()

    for file_df in files_df:
        file_df = file_df.fillna(file_df.mean())
        env = Environment(file_df)
        actor = Actor(env=env, action_sz=7, default_cash=actor_cash, model=args.model, enable_cuda=True, seed=random.randint(0, 1000))
        graph.append_actor(actor)

        for step in range(env.row_sz):
            state = actor.get_state()
            action = actor.choose_action(state)
            next_state, reward = actor.step(action)
            actor.record()

        counter = Counter(actor.history["ACTION"])
        hold_count = counter[actor.action_median]
        buy_count = sum([counter[i] for i in range(0, actor.action_median)])
        sell_count = sum([counter[i] for i in range(actor.action_median + 1, actor.action_sz)])
        print("HOLD {} | BUY {} | SELL {}".format(hold_count, buy_count, sell_count))
        print("Reward {} \n".format(actor.get_reward()))

    graph.draw_all()