import argparse

from mylib.stock import Stock
from mylib.environment import Environment
from mylib.actor import Actor
from mylib.critic import Critic


# Arguments
parser = argparse.ArgumentParser(description="Portfolio Management Model Training (Actor Critic)")
parser.add_argument("--train", "-t", type=str, default="data", help="Path to stock training data.")
parser.add_argument("--episode", "-e", type=int, default=5000, help="Episode Size.")
parser.add_argument("--model", "-m", type=str, default=None, help="Model Name.")
parser.add_argument("--money", type=int, default=10000, help="Initial given money.")
args = parser.parse_args()

# Start from here!
if __name__ == "__main__":
    # Environment (Stock)
    stock = Stock(args.train)
    files_name = stock.get_files_name()
    train_files_df, val_files_df = stock.read_files(files_name[0:1], [.8, .2])

    # Iteration
    for episode in range(args.episode):
        print("[ Episode {}/{} ]".format(episode, args.episode))

        for file_df in train_files_df:
            features_sz = len(file_df.columns)
            action_sz = 3 # 0: Hold, 1: Buy, 2: Sell
            env = Environment(file_df)
            actor = Actor(input_sz=features_sz, output_sz=action_sz, env=env).cuda()
            critic = Critic(input_sz=features_sz).cuda()

            for idx, state in file_df.iterrows():
                action = actor.choose_action(state)
                next_state, reward = actor.step(action)
                td_error = critic.learn(state, reward, next_state)
                actor.learn(state, action, td_error)
                if(idx % 100 == 0):
                    print("Episode {} | Step {} | Reward {}".format(episode, idx, actor.portfolio_value()))