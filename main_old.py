import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error

from sklearn.model_selection import cross_val_score

# 01. Load the dataset
housing = pd.read_csv("housing.csv")    

# 02. Create a new feature 'income_cat' by categorizing 'median_income'
housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]    
    
# 03. Remove the 'income_cat' attribute so the data is back to its original state
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)
    
# 04. We will work on the copy of the training set to avoid any changes to the original data
housing = strat_train_set.copy()

# 05. Separate features and labels
housing_labels = housing["median_house_value"].copy()
housing_features = housing.drop("median_house_value", axis=1)

# print(housing_features.head())
# print(housing_labels.head())
# print(housing_features.info())

# 06. Separate numerical and categorical columns
num_features = housing_features.select_dtypes(include=[np.number]).columns.tolist()
cat_features = housing_features.select_dtypes(include=[object]).columns.tolist()

# print("Numerical features:", num_features)
# print("Categorical features:", cat_features)

# 07. Lets build a pipeline for numerical and categorical features

# Numerical pipeline
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),  # Fill missing values with median
    ("scaler", StandardScaler())  # Standardize the features
])

# Categorical pipeline
cat_pipeline = Pipeline([  
    ("one_hot", OneHotEncoder(handle_unknown="ignore"))  # One-hot encode the categories
])

# Construct the full pipeline using ColumnTransformer
full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_features),  # Apply num_pipeline to numerical features
    ("cat", cat_pipeline, cat_features)   # Apply cat_pipeline to categorical features
])

# 08. Transform the data
housing_prepared = full_pipeline.fit_transform(housing)
print(housing_prepared.shape)


# 09. Train a model

# Linear regression model
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)
lin_predictions = lin_reg.predict(housing_prepared)
# lin_rmse = root_mean_squared_error(housing_labels, lin_predictions)
lin_rmses = -cross_val_score(lin_reg, housing_prepared, housing_labels, scoring="neg_root_mean_squared_error", cv=10)

# print("Linear Regression RMSE:", lin_rmse)
print(pd.Series(lin_rmses).describe() )

# Decision tree model
dt_reg = DecisionTreeRegressor()
dt_reg.fit(housing_prepared, housing_labels)
dt_predictions = dt_reg.predict(housing_prepared)
# dt_rmse = root_mean_squared_error(housing_labels, dt_predictions)
dt_rmses = -cross_val_score(dt_reg, housing_prepared, housing_labels, scoring="neg_root_mean_squared_error", cv=10)

# print("Decision Tree RMSE:", dt_rmse)
print(pd.Series(dt_rmses).describe() )


# Random Forest model
rf_reg = RandomForestRegressor()
rf_reg.fit(housing_prepared, housing_labels)
rf_predictions = rf_reg.predict(housing_prepared)
# rf_rmse = root_mean_squared_error(housing_labels, rf_predictions)
rf_rmses = -cross_val_score(rf_reg, housing_prepared, housing_labels, scoring="neg_root_mean_squared_error", cv=10)

# print("Random Forest RMSE:", rf_rmse)
print(pd.Series(rf_rmses).describe() )





