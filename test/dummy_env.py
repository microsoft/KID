import time


class DummyEnv(object):

    def __init__(self, size):
        self.size = size
        self.action_space = 2
        self.done = False
        self.cur_index = 0
        self.gen_ids = []

    def reset(self):
        self.done = False
        self.cur_index = 0
        self.gen_ids = []
        return self.cur_index

    def step(self, action):
        if self.done:
            raise ValueError('step after done !!!')

        # simulating lag
        time.sleep(0.1)

        # if done
        if self.cur_index == self.size:
            self.done = True
            return 0, 0, 0, [], 0, True, {}

        # action forward pass
        if action == 0:
            self.cur_index = max(self.cur_index - 1, 0)
            self.gen_ids.append(self.cur_index)
            self.done = self.cur_index == self.size
            return 0, self.cur_index, self.cur_index, self.gen_ids, int(
                self.done
            ), self.done, {}
        elif action == 1:
            self.cur_index += 1
            self.gen_ids.append(self.cur_index)
            self.done = self.cur_index == self.size
            return 1, self.cur_index, self.cur_index, self.gen_ids, int(
                self.done
            ), self.done, {}
