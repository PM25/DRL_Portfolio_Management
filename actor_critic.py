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
parser.add_argument("--cash", type=int, default=3000, help="Initial given cash.")
args = parser.parse_args()

# Start from here!
if __name__ == "__main__":
    # Environment (Stock)
    stock = Stock(args.train)
    files_name = stock.get_files_name()
    train_files_df, val_files_df = stock.read_files(files_name, [.8, .2])

    actor_cash = args.cash / len(train_files_df)

    # Iteration
    for episode in range(args.episode):
        print("[ Episode {}/{} ]".format(episode, args.episode))
        print('-' * 25)

        for (idx, file_df) in enumerate(train_files_df, 1):
            file_df = file_df.fillna(file_df.mean())
            env = Environment(file_df)
            if(idx == 1):
                actor = Actor(env=env, action_sz=3, default_cash=actor_cash, enable_cuda=True)
                critic = Critic(input_sz=actor.input_sz + 1, enable_cuda=True)  # input size + 1 for the action
            else:
                actor.update_env(env)

            env.reset()
            actor.reset()
            state = actor.get_state()
            action = 0
            for step in range(env.row_sz):
                next_state, reward = actor.step(action)
                next_action = actor.choose_action(next_state)
                td_error = critic.learn(state, action, reward, next_state, next_action)
                actor.learn(td_error, 0.8)
                actor.record()
                state, action = next_state, next_action

            counter = Counter(actor.history["ACTION"])
            print("Stock {} | Episode {} | Reward {}".format(idx, episode, actor.get_reward()))
            print("HOLD {} | BUY {} | SELL {} \n".format(counter[1], counter[0], counter[2]))
            if(actor.portfolio_value() > (actor.default_cash * 1.5)):
                actor.save_model(str(episode) + '.pkl')
            print('-' * 25)