import numpy as np

from dqo.estimator.neo.encoder import encode_query
from dqo.estimator.query_estimator import QueryEstimator


class NeoQueryEstimator(QueryEstimator):
    converters = {"input": lambda x: np.array(eval(x)), "runtime": float}

    def encode_query(self, query):
        if self.db is None:
            raise RuntimeError('Missing db scheme')
        return encode_query(self.db, query)

    def compile(self):
        from tensorflow.keras.layers import Dense
        from tensorflow.keras.models import Sequential

        model = Sequential()
        model.add(Dense(64, input_dim=318, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(1, activation='linear'))
        model.compile(loss='mse', optimizer='adam', metrics=['accuracy', 'mae', 'mse'])

        self.model = model
