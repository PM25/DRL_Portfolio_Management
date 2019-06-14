from mylib.stock import Stock
from mylib.actor import Actor
from mylib.critic import Critic


# Start from here!
if __name__ == "__main__":
    stock = Stock("data")
    files_name = stock.get_files_name()
    train_files_df, val_files_df = stock.read_files(files_name[0:1], [.8, .2])
    