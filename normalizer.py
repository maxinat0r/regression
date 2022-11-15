class MinMaxilizer:
    """
    Simple min MAX normalizer ;)
    """

    def fit(self, X):
        self.min = X.min(axis=0)
        self.max = X.max(axis=0)
        return self

    def transform(self, X):
        X_normed = (X - self.min) / (self.max - self.min)
        return X_normed
