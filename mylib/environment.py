import pandas as pd


class Environment:
    def __init__(self, data):
        assert (isinstance(data, pd.DataFrame))

        self.data = data
        self.row_sz = self.data.shape[0]
        self.features_sz = self.data.shape[1]
        self.cur_state_idx = self.reset()


    def get_state(self, offset=0, sz=1):
        start_idx = self.cur_state_idx - sz + 1 + offset
        end_idx = self.cur_state_idx + 1 + offset

        if(start_idx < 0):
            first_row_df = self.data[0:1]
            dup_row_df = pd.concat([first_row_df] * abs(start_idx))
            state = dup_row_df.append(self.data[0:end_idx])
        elif(end_idx > self.row_sz):
            state = self.data[start_idx:self.row_sz]
            last_row_df = self.data[self.row_sz - 1:self.row_sz]
            dup_row_df = pd.concat([last_row_df] * abs(end_idx - self.row_sz))
            state = state.append(dup_row_df)
        else:
            state = self.data[start_idx:end_idx]

        return state


    def get_close_price(self):
        df = self.get_state()

        return df["CLOSE"].values


    def get_date(self, offset=0):
        df = self.get_state(offset=offset)

        return df["DATE"].values


    def step(self, sz=1):
        success = False if(self.cur_state_idx >= (self.row_sz - 1)) else True

        if(success):
            self.cur_state_idx += sz

        return success


    def reset(self):
        self.cur_state_idx = 0

        return self.cur_state_idx