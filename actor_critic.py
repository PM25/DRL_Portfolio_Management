import argparse
from collections import Counter

from mylib.stock import Stock
from mylib.environment import Environment
from mylib.actor import Actor
from mylib.critic import Critic
from mylib.graph import Graph


# Arguments
parser = argparse.ArgumentParser(description="Portfolio Management Model Training (Actor Critic)")
parser.add_argument("--train", "-t", type=str, default="data", help="Path to stock training data.")
parser.add_argument("--episode", "-e", type=int, default=5000, help="Episode Size.")
parser.add_argument("--model", "-m", type=str, default=None, help="Model Name.")
parser.add_argument("--cash", type=int, default=2000, help="Initial given cash.")
args = parser.parse_args()

# Start from here!
if __name__ == "__main__":
    # Environment (Stock)
    stock = Stock(args.train)
    files_name = stock.get_files_name()
    train_files_df, val_files_df = stock.read_files(files_name[0:1], [.8, .2])

    for file_df in train_files_df:
        features_sz = len(file_df.columns)
        action_sz = 3  # 0: Hold, 1: Buy, 2: Sell
        env = Environment(file_df)
        actor = Actor(input_sz=features_sz, output_sz=action_sz, env=env, default_cash=args.cash).cuda()
        critic = Critic(input_sz=features_sz).cuda()
        graph = Graph(file_df["DATE"].values, file_df["CLOSE"].values)

        # Iteration
        for episode in range(args.episode):
            print("[ Episode {}/{} ]".format(episode, args.episode))

            env.reset()
            actor.reset()
            for idx, state in file_df.iterrows():
                action = actor.choose_action(state)
                next_state, reward = actor.step(action)
                td_error = critic.learn(state, reward, next_state)
                actor.learn(td_error)
                actor.record()
                if(idx % 100 == 0):
                    print("Episode {} | Step {} | Reward {}".format(episode, idx, actor.portfolio_value()))
                    counter = Counter(actor.history["ACTION"])
                    print("HOLD {} | BUY {} | SELL {}".format(counter[0], counter[1], counter[2]))
                    print('-' * 100)

            # graph.add_dots_on_line(actor.history["DATE"], actor.history["ACTION"])
            # graph.draw()
