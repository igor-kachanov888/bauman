import numpy as np

class Scaler:
    def __init__(self, limit_min = 0.015, limit_max = 0.985):
        self.limit_min = limit_min
        self.limit_max = limit_max
        self.min = None
        self.max = None

    def fit(self, X: np.array):
        self.min = np.quantile(X, self.limit_min, axis=0)
        self.max = np.quantile(X, self.limit_max, axis=0)

    def transform(self, X: np.array, isBlowout:bool = False):
        d = (self.max - self.min)
        d[ d == 0] = 1
        result = (X - self.min) / d
        if isBlowout:
            result[result > 1.] = 1.
            result[result < 0.] = 0
        return result

    def fit_transform(self, X: np.array, isBlowout:bool = False):
        self.fit(X)
        return self.transform(X, isBlowout)


