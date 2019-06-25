import argparse
from collections import Counter

from mylib.stock import Stock
from mylib.environment import Environment
from mylib.actor import Actor
from mylib.critic import Critic
from mylib.graph import Graph


# Arguments
parser = argparse.ArgumentParser(description="Portfolio Management Model Training (Actor Critic)")
parser.add_argument("--data", "-d", type=str, default="data", help="Path to stock training data.")
# parser.add_argument("--model", "-m", type=str, help="Model Name.", required=True)
parser.add_argument("--model", "-m", type=str, default="1.pkl", help="Model Name.")
parser.add_argument("--cash", type=int, default=2000, help="Initial given cash.")
args = parser.parse_args()

# Start from here!
if __name__ == "__main__":
    # Environment (Stock)
    stock = Stock(args.data)
    files_name = stock.get_files_name()
    files_df, _ = stock.read_files(files_name[0:3], split_ratio=[1, 0])
    actor_cash = args.cash / len(files_df)
    graph = Graph()

    for file_df in files_df:
        file_df = file_df.fillna(file_df.mean())
        env = Environment(file_df)
        actor = Actor(env=env, action_sz=3, default_cash=actor_cash, model=args.model, enable_cuda=True)
        graph.append_actor(actor)

        for step in range(env.row_sz):
            state = actor.get_state()
            action = actor.choose_action(state)
            next_state, reward = actor.step(action)
            actor.record()

        counter = Counter(actor.history["ACTION"])
        print("HOLD {} | BUY {} | SELL {}".format(counter[1], counter[0], counter[2]))
        print("Reward {} \n".format(actor.get_reward()))

    graph.draw_all()