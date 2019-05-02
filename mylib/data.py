from mylib import functions

import numpy as np
import csv


class StockData:
    # {raw_data}: stock data without any process
    # {normalize_data}: stock data with normalization
    def __init__(self, path, mean=None, std=None):
        self.raw_data = self.csv_to_list(path)
        self.clean_data = self.clean(self.raw_data)
        self.mean, self.std = mean , std
        if(mean is None):
            self.mean = self.clean_data.mean(axis=0)
        if(std is None):
            self.std = self.clean_data.std(axis=0)
        self.normalize_data = self.normalize(self.clean_data, self.mean, self.std)
        self.sample_size, self.feature_size = self.normalize_data.shape


    # Convert CSV file to python list
    def csv_to_list(self, path):
        out_list = []

        with open(path) as csv_file:
            rows = csv.reader(csv_file)
            for row in rows:
                out_list.append(row)

        return out_list[1:]  # Remove the header


    def clean(self, raw_data):
        out = np.zeros((len(raw_data), len(raw_data[0])))
        for i in range(len(raw_data)):
            for j in range(len(raw_data[0])):
                if(functions.is_number(raw_data[i][j])):
                    out[i, j] = raw_data[i][j]
                elif(functions.is_date(raw_data[i][j])):
                    out[i, j] = functions.date_to_number(raw_data[i][j])
                else:
                    out[i, j] = 0

        return out


    def normalize(self, data, mean=None, std=None):
        if(mean is None): mean = data.mean(axis=0)
        if(std is None): std = data.std(axis=0)

        data -= mean
        data /= std

        return data


    def get_state(self, end_time, win_size):
        start_time = end_time - win_size + 1

        if(start_time >= 0):
            data_piece = self.normalize_data[start_time:end_time+1]
        else:
            front_part = abs(start_time) * [self.normalize_data[0]]
            rear_part = self.normalize_data[:end_time+1]
            data_piece = np.concatenate((front_part, rear_part), axis=0)

        return data_piece