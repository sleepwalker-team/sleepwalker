"""Native LightGBM fitting with training-only bins and validation selection."""

import numpy as np

import lightgbm as lgb


class LightGBMModel:
    def __init__(self, classes, parameters=None, num_boost_round=100, early_stopping_rounds=20):
        self.classes = list(classes)
        if len(self.classes) < 2 or len(set(self.classes)) != len(self.classes):
            raise ValueError("classes must contain at least two distinct labels.")
        self.parameters = dict(parameters or {})
        if self.parameters.get("feature_pre_filter", False):
            raise ValueError("Use feature_pre_filter=False so memory and Sequence construction apply the same feature policy.")
        if any(key in self.parameters for key in ["objective", "num_class", "num_iterations", "num_iteration", "n_estimators", "num_boost_round"]):
            raise ValueError("Classification objective and round count are owned by the wrapper; use classes and num_boost_round.")
        self.num_boost_round = int(num_boost_round)
        self.early_stopping_rounds = int(early_stopping_rounds)
        if self.num_boost_round < 1 or self.early_stopping_rounds < 0:
            raise ValueError("num_boost_round must be positive and early_stopping_rounds nonnegative.")
        self.booster = None
        self.validation_used = False

    def fit(self, X, y, validation_data=None):
        # Sequence bin construction omits min_data_in_leaf from Dataset params.
        # Disable early feature filtering in both modes; tree fitting still
        # applies the configured leaf constraints.
        parameters = {"verbosity": -1, "feature_pre_filter": False, **self.parameters, "objective": "binary" if len(self.classes) == 2 else "multiclass"}
        if len(self.classes) > 2:
            parameters["num_class"] = len(self.classes)
        # Sequence casts bounded construction batches, rather than upcasting the
        # whole disk-backed matrix. Native bins/histograms still consume RAM.
        data = FeatureSequence(X) if isinstance(X, np.memmap) else X
        training = lgb.Dataset(data, label=y, params=parameters, free_raw_data=True)
        validation = None
        callbacks = []
        if validation_data is not None:
            X_val, y_val = validation_data
            values = FeatureSequence(X_val) if isinstance(X_val, np.memmap) else X_val
            validation = [lgb.Dataset(values, label=y_val, reference=training, free_raw_data=True)]
            if self.early_stopping_rounds:
                callbacks.append(lgb.early_stopping(self.early_stopping_rounds, verbose=False))
        self.booster = lgb.train(parameters, training, num_boost_round=self.num_boost_round, valid_sets=validation, callbacks=callbacks)
        self.validation_used = validation_data is not None
        return self

    def is_fitted(self):
        return self.booster is not None

    def probability_classes(self):
        return list(self.classes)

    def predict_proba(self, X):
        if not self.is_fitted():
            raise RuntimeError("LightGBMModel has not been fitted.")
        probabilities = np.asarray(self.booster.predict(X))
        return np.column_stack([1 - probabilities, probabilities]) if len(self.classes) == 2 else probabilities


class FeatureSequence(lgb.Sequence):
    batch_size = 4096

    def __init__(self, values):
        self.values = values

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        return np.asarray(self.values[index], dtype=np.float64)
