import numpy as np
import math
import csv
import sys


class StockData:
    def __init__(self, path):
        self.stock_data = self.csv_to_list(path)
        self.data = self.format(self.stock_data)
        self.sample_size, self.feature_size = self.data.shape

    def csv_to_list(self, path):
        out_list = []

        with open(path) as csv_file:
            rows = csv.reader(csv_file)
            for row in rows:
                out_list.append(row)

        return out_list[1:]  # Remove the header

    def normalize(self, data):
        if(is_number(data)):
            return float(data)
        else:
            combine_num = ""
            for num in data.split('-'):
                if(not is_number(num)): sys.exit("Error: Wrong Format.")
                combine_num += num

            return int(combine_num)

    def format(self, data_list):
        out = np.zeros((len(data_list), len(data_list[0])))
        for i in range(len(data_list)):
            for j in range(len(data_list[0])):
                out[i, j] = self.normalize(data_list[i][j])

        return out


def is_number(data):
    try:
        if ('.' in data):
            data = float(data)
        else:
            data = int(data)
        return True

    except ValueError:
        return False


def get_state(stock_list, end_time, win_size):
    start_time = end_time - win_size + 1

    if(start_time >= 0):
        block = stock_list[start_time:end_time+1]
    else:
        front = abs(start_time) * [stock_list[0]]
        rear = stock_list[:end_time+1]
        block = np.concatenate((front, rear), axis=0)

    return np.array(block)