class Normalizer:
    """
    normalize
    """

    def fit(self, X):
        self.x_max = X.max(axis=0)
        return self

    def transform(self, X):
        X = X / self.x_max
        return X
