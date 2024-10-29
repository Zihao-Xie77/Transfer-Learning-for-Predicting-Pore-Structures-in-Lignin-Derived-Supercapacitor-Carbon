import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from xgboost import XGBRegressor
import shap


file_path = r'Dataset\Cleaned original samples.xlsx'
date = pd.read_excel(file_path)

s = StratifiedShuffleSplit(n_splits=1, train_size=0.8, test_size=0.2, random_state=54)
strata_variable = date[['Agent_H3PO4', 'Agent_K2CO3', 'Agent_KOH', 'Agent_Na2CO3', 'Agent_NaOH',
                        'Agent_ZnCl2']]

for train_index, test_index in s.split(date, strata_variable):
    strat_train_set = date.loc[train_index]
    strat_test_set = date.loc[test_index]

my_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="mean")),
    ('std_scaler', StandardScaler())
])

features_train = strat_train_set.drop(['SBET', 'TPV'], axis=1)
features_train_label = strat_train_set[['Agent_H3PO4', 'Agent_K2CO3', 'Agent_KOH', 'Agent_Na2CO3', 'Agent_NaOH',
                                        'Agent_ZnCl2']]
label_train = strat_train_set[['SBET', 'TPV']]

features_test = strat_test_set.drop(['SBET', 'TPV'], axis=1)
feature_test_label = strat_test_set[['Agent_H3PO4', 'Agent_K2CO3', 'Agent_KOH', 'Agent_Na2CO3', 'Agent_NaOH',
                                     'Agent_ZnCl2']]
label_test = strat_test_set[['SBET', 'TPV']]

features_prepared_train_real = features_train
features_prepared_test = features_test

label_prepared_train_real = label_train
label_prepared_test = label_test.to_numpy()

df_fake = pd.read_excel(r"Dataset\Generated samples.xlsx")
df_fake_features = df_fake.drop(['SBET', 'TPV'], axis=1)
df_fake_labels = df_fake[['SBET', 'TPV']]

df_fake_features = df_fake_features.to_numpy()
df_fake_labels = df_fake_labels.to_numpy()

features_prepared_train = np.vstack((features_prepared_train_real, df_fake_features))
label_prepared_train = np.vstack((label_prepared_train_real, df_fake_labels))

model1GBR_416 = XGBRegressor(n_estimators=130, learning_rate=0.18, max_depth=7, subsample=0.8, colsample_bytree=0.6,
                             gamma=0.03, min_child_weight=2)

model2GBR_416 = XGBRegressor(n_estimators=130, learning_rate=0.18, max_depth=7, subsample=0.7, colsample_bytree=0.6,
                             gamma=0.03, min_child_weight=2)

model1GBR_416.fit(features_prepared_train, label_prepared_train[:, 0])
model2GBR_416.fit(features_prepared_train, label_prepared_train[:, 1])

class Evaluate:
    def __init__(self, model, features, labels):
        self.model = model
        self.features = features
        self.labels = labels

    def rmse(self):
        from sklearn.metrics import mean_squared_error
        prediction = self.model.predict(self.features)
        mse = mean_squared_error(self.labels, prediction)
        rmse = np.sqrt(mse)
        print("-- RMSE --")
        print(f"Root Mean Square Error is: {rmse}")

    def cross_validation(self):
        from sklearn.model_selection import cross_val_score
        scores = cross_val_score(self.model, self.features, self.labels, scoring="neg_mean_squared_error", cv=10)
        rmse_scores = np.sqrt(-scores)
        print("-- Cross Validation --")
        print("Mean: ", rmse_scores.mean())
        print("Standard deviation: ", rmse_scores.std())

    def r2_score(self):
        from sklearn.metrics import r2_score
        label_prediction = self.model.predict(self.features)
        r2 = r2_score(self.labels, label_prediction)
        print("-- R2 Score --")
        print("R2: ", r2)

print("\n\n--- SBET ----")

print("\n\n-- TRAIN EVALUATION ---")
e1 = Evaluate(model1GBR_416, features_prepared_train, label_prepared_train[:, 0])
e1.rmse()
e1.cross_validation()
e1.r2_score()

print("\n\n--- TEST EVALUATION ---")
e2 = Evaluate(model1GBR_416, features_prepared_test, label_prepared_test[:, 0])
e2.rmse()
e2.r2_score()

print("\n\n--- VPT ----")

print("\n\n-- TRAIN EVALUATION ---")
e3 = Evaluate(model2GBR_416, features_prepared_train, label_prepared_train[:, 1])
e3.rmse()
e3.cross_validation()
e3.r2_score()

print("\n\n--- TEST EVALUATION ---")
e4 = Evaluate(model2GBR_416, features_prepared_test, label_prepared_test[:, 1])
e4.rmse()
e4.r2_score()

# feature importance
feature_importances1 = (model1GBR_416.feature_importances_)
feature_importances2 = (model2GBR_416.feature_importances_)

custom_labels1 = ['C', 'H', 'O', 'N', 'S', 'VM', 'Ash', 'FC',
                 'A/S', 'HR', 'Temp', 'Time',
                 '$H_{\mathrm{3}}$P$O_{\mathrm{4}}$',
                 '$K_{\mathrm{2}}$C$O_{\mathrm{3}}$', 'KOH',
                 '$Na_{\mathrm{2}}$C$O_{\mathrm{3}}$', 'NaOH', 'Zn$Cl_{\mathrm{2}}$']

dd1 = pd.DataFrame({
    'Feature': custom_labels1,
    'ImportanceSSA': feature_importances1,
    'ImportanceTPV': feature_importances2
})

# SHAP analysis
explainer1 = shap.Explainer(model1GBR_416)
shap_values1 = explainer1(features_prepared_train)
shap_values_df1 = pd.DataFrame(shap_values1.values, columns=features_train.columns)
feature_train_shap = pd.DataFrame(features_prepared_train, columns=features_train.columns)
shap.summary_plot(shap_values1, features_prepared_train, feature_names=features_train.columns)

# feature_X1 = 'VM'
# feature_X2 = 'FC'
feature_X1 = 'Activation_temp'
feature_X2 = 'Activation_time'
shap_values_X1 = shap_values_df1[feature_X1]
shap_values_X2 = shap_values_df1[feature_X2]
features_X1 = feature_train_shap[feature_X1]
features_X2 = feature_train_shap[feature_X2]
dependency_df = pd.DataFrame({
    feature_X1: features_X1,
    feature_X2: features_X2,
    'shap_value': shap_values_X1 + shap_values_X2
})
pdp_df_SSA = dependency_df.groupby([feature_X1, feature_X2]).mean().reset_index()


explainer2 = shap.Explainer(model2GBR_416)
shap_values2 = explainer2(features_prepared_train)

shap_values_df2 = pd.DataFrame(shap_values2.values, columns=features_train.columns)
shap_values_df2.to_csv(r'D:\python machine learning\GAN\Project1\Data\Final\shap_values_TPV.csv', index=False)

shap.summary_plot(shap_values2, features_prepared_train, feature_names=features_train.columns)

feature_X1 = 'Activation_temp'
feature_X2 = 'Activation_time'

shap_values_X1 = shap_values_df2[feature_X1]
shap_values_X2 = shap_values_df2[feature_X2]
features_X1 = feature_train_shap[feature_X1]
features_X2 = feature_train_shap[feature_X2]

dependency_df = pd.DataFrame({
    feature_X1: features_X1,
    feature_X2: features_X2,
    'shap_value': shap_values_X1 + shap_values_X2
})

pdp_df_TPV = dependency_df.groupby([feature_X1, feature_X2]).mean().reset_index()
