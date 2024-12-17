
import datagen
import data
from tensorflow.keras.utils import Sequence
import numpy as np

def normalize(s):
    return np.array(tuple((s[i] - data.get_mean()[i]) / data.get_stdev()[i] 
        for i in range(len(s))))

def to_output(t):
    return np.array(tuple(1 if i == t else 0 for i in range(4)))

class TrainDataSeq(Sequence):
    def __len__(self):
        return datagen.train_tot

    def __getitem__(self, idx):
        sample = datagen.train_sample(idx)
        output = (normalize(sample[0]).reshape((1, 7)), to_output(sample[1]).reshape((1, 4)))
        return output

class TestDataSeq(Sequence):
    def __len__(self):
        return datagen.test_tot
    def __getitem__(self, idx):
        sample = datagen.test_sample(idx)
        return (normalize(sample[0]).reshape((1, 7)), to_output(sample[1]).reshape((1, 4)))

