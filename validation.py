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
parser.add_argument("--model", "-m", type=str, default="1.pkl", help="Model Name.", required=False)
parser.add_argument("--cash", type=int, default=2000, help="Initial given cash.")
args = parser.parse_args()

# Start from here!
if __name__ == "__main__":
    if __name__ == "__main__":
        # Environment (Stock)
        stock = Stock(args.data)
        files_name = stock.get_files_name()
        files_df, _ = stock.read_files(files_name[0:1])

        for file_df in files_df:
            file_df = file_df.fillna(file_df.mean())
            env = Environment(file_df)
            actor = Actor(env=env, action_sz=3, default_cash=args.cash, model=args.model)
            graph = Graph(file_df["DATE"].values, file_df["CLOSE"].values)

            env.reset()
            actor.reset()
            for step in range(env.row_sz):
                state = actor.get_state()
                action = actor.choose_action(state)
                next_state, reward = actor.step(action)
                actor.record()

            counter = Counter(actor.history["ACTION"])
            print("HOLD {} | BUY {} | SELL {}".format(counter[0], counter[1], counter[2]))
            print("Reward {} \n".format(actor.portfolio_value()))

            graph.add_dots_on_line(actor.history["DATE"], actor.history["ACTION"])
            graph.draw()