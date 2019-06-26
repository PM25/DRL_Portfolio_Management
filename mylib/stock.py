from pathlib import Path
import pandas as pd
import math


class Stock:

    def __init__(self, path="data"):
        self.base_dir = Path(path)
        self.train_dir = self.base_dir/Path("train")
        self.val_dir = self.base_dir/Path("validation")
        self.test_dir = self.base_dir/Path("test")
        self.files_name = self.get_files_name()


    def get_files_name(self):
        return [fname for fname in self.base_dir.glob("*.csv")]


    # {files_name}: a list that contain files' name.
    def read_files(self, files_name=None, split_ratio=None):
        if(files_name == None):
            files_name = self.files_name
        if(split_ratio == None):
            split_ratio = [.8, .2]

        assert(sum(split_ratio) == 1)

        files_df = [pd.read_csv(fname).drop(["CODE"], axis=1) for fname in files_name]
        start_ratio = 0
        split_files_df = []
        for ratio in split_ratio:
            end_ratio = start_ratio + ratio
            split_files_df.append([self.split_df(df, start_ratio, end_ratio) for df in files_df])
            start_ratio = end_ratio

        return split_files_df


    def split_df(self, df, start_ratio, end_ratio):
        assert(start_ratio >= 0 and start_ratio <= 1 and start_ratio <= end_ratio)
        assert (end_ratio >= 0 and end_ratio <= 1)

        row_count = len(df)
        start_idx = math.floor(row_count*start_ratio)
        end_idx = math.floor(row_count*end_ratio)

        return df[start_idx:end_idx]