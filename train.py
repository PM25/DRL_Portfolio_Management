from mylib.agent import Agent
from mylib import data

import sys
import numpy as np


if(len(sys.argv) == 1) :
	stock_path = input("Stock File Path:")
	window_size = int(input("Window Size:"))
	episode_count = int(input("Episodes:"))
elif(len(sys.argv) == 4):
	stock_path = sys.argv[1]
	window_size = int(sys.argv[2])
	episode_count = int(sys.argv[3])
else:
	sys.exit("Error: Wrong number of parameter.")



agent = Agent(window_size)

stockdata = data.StockData(stock_path)
stock_data = stockdata.stock_data
data = stockdata.data