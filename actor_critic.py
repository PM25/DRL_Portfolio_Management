import argparse
from collections import Counter

from mylib.stock import Stock
from mylib.environment import Environment
from mylib.actor import Actor
from mylib.critic import Critic


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
        file_df = file_df.fillna(file_df.mean())
        env = Environment(file_df)
        actor = Actor(env=env, action_sz=3, default_cash=args.cash)
        critic = Critic(input_sz=actor.input_sz).cuda()

        # Iteration
        for episode in range(args.episode):
            print("[ Episode {}/{} ]".format(episode, args.episode))
            print('-' * 25)

            env.reset()
            actor.reset()
            for step in range(env.row_sz):
                state = actor.get_state()
                action = actor.choose_action(state)
                next_state, reward = actor.step(action)
                td_error = critic.learn(state, reward, next_state)
                actor.learn(td_error)
                actor.record()

            counter = Counter(actor.history["ACTION"])
            print("Episode {} | Reward {}".format(episode, actor.portfolio_value()))
            print("HOLD {} | BUY {} | SELL {} \n".format(counter[0], counter[1], counter[2]))
            if(actor.portfolio_value() > (actor.default_cash * 1.01)):
                actor.save_model(str(episode) + '.pkl')
