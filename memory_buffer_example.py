import pickle


store_name = "store-2"
batch_size = 8


with open(f"./data/{store_name}-buffer.p", 'rb') as f:
    buffer = pickle.load(f)


for i in range(10):
    obses_t, actions, rewards, obses_tp1, dones = buffer.sample(batch_size)

    print(rewards)