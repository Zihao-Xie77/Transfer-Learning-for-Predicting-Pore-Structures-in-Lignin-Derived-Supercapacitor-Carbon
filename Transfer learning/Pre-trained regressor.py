import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
import xgboost as xgb


class DataPreprocessor:
    def __init__(self, file_path, strata_columns, target_columns):
        self.file_path = file_path
        self.strata_columns = strata_columns
        self.target_columns = target_columns

    def load_and_encode(self):
        df = pd.read_excel(self.file_path)
        column_to_encode = 'Agent'
        df_encoded = pd.get_dummies(df[column_to_encode], prefix=column_to_encode)
        df = pd.concat([df, df_encoded], axis=1)
        df.drop(columns=[column_to_encode], inplace=True)
        df.to_csv(self.file_path.replace(".xlsx", ".csv"), index=False)
        self.df = pd.read_csv(self.file_path.replace(".xlsx", ".csv"))

    def stratified_split(self, test_size=0.2, train_size=0.8, random_state=77):
        s = StratifiedShuffleSplit(n_splits=1, train_size=train_size, test_size=test_size, random_state=random_state)
        strata_variable = self.df[self.strata_columns]
        for train_index, test_index in s.split(self.df, strata_variable):
            self.strat_train_set = self.df.loc[train_index]
            self.strat_test_set = self.df.loc[test_index]

    def preprocess(self):
        my_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy="mean")),
            ('std_scaler', StandardScaler())
        ])

        self.features_train = self.strat_train_set.drop(self.target_columns, axis=1)
        self.label_train = self.strat_train_set[self.target_columns]
        self.features_test = self.strat_test_set.drop(self.target_columns, axis=1)
        self.label_test = self.strat_test_set[self.target_columns]

        self.features_prepared_train = my_pipeline.fit_transform(self.features_train)
        self.features_prepared_test = my_pipeline.transform(self.features_test)
        self.label_train = my_pipeline.fit_transform(self.label_train)
        self.label_test = my_pipeline.transform(self.label_test)

        return self.features_prepared_train, self.features_prepared_test, self.label_train, self.label_test

class Evaluation:
    def __init__(self, model, features, labels):
        self.model = model
        self.features = features
        self.labels = labels

    def rmse(self):
        prediction = self.model.predict(self.features)
        mse = mean_squared_error(self.labels, prediction)
        rmse = np.sqrt(mse)
        print("-- RMSE --")
        print(f"Root Mean Square Error is: {rmse}")

    def cross_validation(self):
        scores = cross_val_score(self.model, self.features, self.labels, scoring="neg_mean_squared_error", cv=10)
        rmse_scores = np.sqrt(-scores)
        print("-- Cross Validation --")
        print("Mean: ", rmse_scores.mean())
        print("Standard deviation: ", rmse_scores.std())

    def r2_score(self):
        label_prediction = self.model.predict(self.features)
        r2 = r2_score(self.labels, label_prediction)
        print("-- R2 Score --")
        print("R2: ", r2)


file_path = r"Dataset/Original LDPC dataset.xlsx"
strata_columns = ['Agent_H3PO4', 'Agent_K2CO3', 'Agent_KOH', 'Agent_Na2CO3', 'Agent_NaOH', 'Agent_ZnCl2']
target_columns = ['SBET', 'TPV']

data_processor = DataPreprocessor(file_path, strata_columns, target_columns)
data_processor.load_and_encode()
data_processor.stratified_split()
features_train, features_test, label_train, label_test = data_processor.preprocess()

model1GBR_416 = xgb.XGBRegressor(n_estimators=150, learning_rate=0.2, max_depth=7, subsample=0.8, colsample_bytree=0.6,
                                 gamma=0.01, min_child_weight=2)
model2GBR_416 = xgb.XGBRegressor(n_estimators=170, learning_rate=0.18, max_depth=7, subsample=0.8, colsample_bytree=0.7,
                                 gamma=0.01, min_child_weight=2)

model1GBR_416.fit(features_train, label_train[:, 0])
model2GBR_416.fit(features_train, label_train[:, 1])

print("\n\n--- SBET ----")

print("\n\n-- TRAIN EVALUATION ---")
e1 = Evaluation(model1GBR_416, features_train, label_train[:, 0])
e1.rmse()
e1.cross_validation()
e1.r2_score()

print("\n\n--- TEST EVALUATION ---")
e2 = Evaluation(model1GBR_416, features_test, label_test[:, 0])
e2.rmse()
e2.r2_score()

print("\n\n--- TPV ----")

print("\n\n-- TRAIN EVALUATION ---")
e3 = Evaluation(model2GBR_416, features_train, label_train[:, 1])
e3.rmse()
e3.cross_validation()
e3.r2_score()

print("\n\n--- TEST EVALUATION ---")
e4 = Evaluation(model2GBR_416, features_test, label_test[:, 1])
e4.rmse()
e4.r2_score()
