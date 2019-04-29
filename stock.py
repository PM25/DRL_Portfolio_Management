from yahoo_fin import stock_info as si
from pandas_datareader import data as web
import datetime as dt
import fix_yahoo_finance as yf
import os

yf.pdr_override()


# Input date format (year, month, day)
# Return Pandas format
def get_stock_data(start_date, end_date):
    year, month, day = start_date
    start = dt.datetime(year, month, day)
    year, month, day = end_date
    end = dt.datetime(year, month, day)
    return web.get_data_yahoo(['2002.TW'], start, end)


if (__name__ == "__main__"):
    # Check if Directory exist or not
    base_path = "data"
    train_path = os.path.join(base_path, "train")
    if(not os.path.exists(train_path)):
        os.mkdir(train_path)
    validation_path = os.path.join(base_path, "validation")
    if (not os.path.exists(validation_path)):
        os.mkdir(validation_path)
    test_path = os.path.join(base_path, "test")
    if (not os.path.exists(test_path)):
        os.mkdir(test_path)

    # Get stock data
    train_df = get_stock_data((2000, 1, 1), (2017, 1, 1))
    validation_df = get_stock_data((2017, 1, 2), (2018, 1, 1))
    test_df = get_stock_data((2018, 1, 2), (2019, 4, 28))

    # Save stock data as csv file
    train_df.to_csv(os.path.join(train_path, "2002.TW.csv"))
    validation_df.to_csv(os.path.join(validation_path, "2002.TW.csv"))
    test_df.to_csv(os.path.join(test_path, "2002.TW.csv"))