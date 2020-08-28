

class Queue(object):

    def __init__(self):
        self.storage = {}


    def __index__(self, state):
        s_string = self.stringify(state)
        return self.storage[s_string]


    def stringify(self, arr):
        return ",".join(arr.astype(str))

    def add(self, state):
        s_string = self.stringify(state)

        self.storage[s_string] = []

    def check_val(self, state):
        s_string = self.stringify(state)
        if s_string in self.storage:
            return True
        else:
            return False