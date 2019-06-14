import pandas as pd


class Environment:
    def __init__(self, data):
        assert (isinstance(data, pd.DataFrame))

        self.data = data
        self.sz = len(self.data)
        self.cur_state_idx = self.reset()


    def get_state(self, offset=0, sz=1):
        start_idx = self.cur_state_idx - sz + 1 + offset
        end_idx = self.cur_state_idx + 1 + offset

        if(start_idx < 0):
            first_row_df = self.data[0:1]
            dup_row_df = pd.concat([first_row_df] * abs(start_idx))
            state = dup_row_df.append(self.data[0:end_idx])
        elif(end_idx > self.sz):
            state = self.data[start_idx:self.sz]
            last_row_df = self.data[self.sz-1:self.sz]
            dup_row_df = pd.concat([last_row_df] * abs(end_idx - self.sz))
            state = state.append(dup_row_df)
        else:
            state = self.data[start_idx:end_idx]

        return state


    def step(self, sz=1):
        success = False if(self.cur_state_idx >= (self.sz-1)) else True

        if(success):
            self.cur_state_idx += sz

        return (success, self.get_state(sz))


    def reset(self):
        self.cur_state_idx = 0

        return self.cur_state_idx