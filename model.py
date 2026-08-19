class Model:
    """Base model wrapper."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def fit(self, X, y):
        raise NotImplementedError("fit() must be implemented in a subclass.")

    def predict(self, X):
        raise NotImplementedError("predict() must be implemented in a subclass.")
