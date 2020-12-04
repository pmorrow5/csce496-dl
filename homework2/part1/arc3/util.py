# Helper Functions
import numpy as np

# found from this stack overlow post: https://stackoverflow.com/questions/4601373/better-way-to-shuffle-two-numpy-arrays-in-unison
# x, y = shuffler(x, y)
def shuffler(images, label):
    randomize = np.arange(len(images))
    np.random.shuffle(randomize)
    images = images[randomize]
    label = label[randomize]
    return images, label

# function to split data off
# x, y = split_data(y, .1)
# Split off 10% from y and put it into x while remvoing
# the 10% split off into x from y 
def split_data(data, proportion):
    size = data.shape[0]
    split_idx = int(proportion * size)
    return data[:split_idx], data[split_idx:]