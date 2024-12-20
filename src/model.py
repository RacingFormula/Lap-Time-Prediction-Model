import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

class LapTimePredictionModel:
    def __init__(self):
        self.model = RandomForestRegressor(random_state=42)

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def save_model(self, path):
        import joblib
        joblib.dump(self.model, path)

    def load_model(self, path):
        import joblib
        self.model = joblib.load(path)