import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

import unittest
from src.model import LapTimePredictionModel
from src.data_processing import preprocess_data
import pandas as pd

class TestLapTimePredictionModel(unittest.TestCase):
    def test_training_and_prediction(self):
        data = {
            'Speed': [120, 118, 122, 121, 119],
            'Distance': [2000, 2100, 2050, 2080, 2020],
            'LapTime': [90.5, 91.2, 89.8, 90.0, 90.7]
        }
        df = pd.DataFrame(data)

        features = df.drop(columns=['LapTime'])
        labels = df['LapTime']

        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

        model = LapTimePredictionModel()
        model.train(X_train, y_train)

        predictions = model.predict(X_test)
        self.assertEqual(len(predictions), len(y_test))

if __name__ == '__main__':
    unittest.main()
