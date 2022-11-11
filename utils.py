import numpy as np

def train_test_split(df, fraction=0.8):
    msk = np.random.rand(len(df)) < fraction
    train = df[msk]
    test = df[~msk]

    return train, test
