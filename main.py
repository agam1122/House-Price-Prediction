import os
import joblib

import numpy as np
import pandas as pd

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

MODEL_FILE = "model.pkl"
PIPELINE_FILE = "pipeline.pkl" 

def build_pipeline(num_features, cat_features):
    # Numerical pipeline
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),  # Fill missing values with median
        ("scaler", StandardScaler())  # Standardize the features
    ])

    # Categorical pipeline
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),  # Fill missing values with most frequent value
        ("onehot", OneHotEncoder(handle_unknown="ignore"))  # One-hot encode categorical features
    ])

    # Combine numerical and categorical pipelines into a full pipeline
    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_features),
        ("cat", cat_pipeline, cat_features)
    ])

    return full_pipeline    


if not os.path.exists(MODEL_FILE):
    # 01. Load the data
    housing = pd.read_csv("housing.csv")

    # 02. Create a new feature 'income_cat' by categorizing 'median_income'
    housing["income_cat"] = pd.cut(housing["median_income"],
                                bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                                labels=[1, 2, 3, 4, 5])

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing["income_cat"]):
        housing.loc[test_index].drop(["income_cat", "median_house_value"], axis=1).to_csv("input_data.csv", index=False)
        housing = housing.loc[train_index].drop("income_cat", axis=1 )
        
    # 03. Separate features and labels
    housing_labels = housing["median_house_value"].copy()
    housing_features = housing.drop("median_house_value", axis=1)
    
    # 06. Separate numerical and categorical columns
    num_features = housing_features.select_dtypes(include=[np.number]).columns.tolist()
    cat_features = housing_features.select_dtypes(exclude=[np.number]).columns.tolist()
    
    # 07. Construct the full pipeline
    full_pipeline = build_pipeline(num_features, cat_features)
    
    housing_prepared = full_pipeline.fit_transform(housing_features)
    # print(housing_prepared)
    
    # 08. Train a model
    model = RandomForestRegressor(random_state=42)
    model.fit(housing_prepared, housing_labels)
    
    joblib.dump(model, MODEL_FILE)
    joblib.dump(full_pipeline, PIPELINE_FILE)
    print("Model and pipeline saved to disk.")
else:
    model = joblib.load(MODEL_FILE)
    full_pipeline = joblib.load(PIPELINE_FILE)
        
    
input_data = pd.read_csv("input_data.csv")  # Load input data from CSV file
input_prepared = full_pipeline.transform(input_data)  # Preprocess the input data using the saved pipeline
predictions = model.predict(input_prepared)  # Make predictions using the loaded model
input_data["median_house_value"] = predictions  # Add predictions to the input data
input_data.to_csv("output.csv", index=False)  # Save the predictions to a new CSV file
print("Predictions saved to output.csv")










