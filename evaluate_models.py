# import os
# import pickle
# import numpy as np
# import pandas as pd
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# # Define paths
# models_dir = "artifacts/models"
# test_data_path = "artifacts/test.csv"  # Path to the test CSV file

# # Load test data
# test_data = pd.read_csv(test_data_path)

# # Preprocess test data
# # Assuming the target column is "Time_taken (min)" and the rest are features
# X_test = test_data.drop(columns=["Time_taken (min)"])
# y_test = test_data["Time_taken (min)"]

# # Handle missing values if any
# X_test = X_test.fillna(0)  # Replace missing values with 0 (adjust as needed)

# # Initialize results dictionary
# results = {}

# # Iterate through all models in the directory
# for model_file in os.listdir(models_dir):
#     model_path = os.path.join(models_dir, model_file)
    
#     # Load the model
#     with open(model_path, "rb") as f:
#         model = pickle.load(f)
    
#     # Make predictions
#     y_pred = model.predict(X_test)
    
#     # Calculate metrics
#     mae = mean_absolute_error(y_test, y_pred)
#     mse = mean_squared_error(y_test, y_pred)
#     rmse = np.sqrt(mse)
#     r2 = r2_score(y_test, y_pred)
    
#     # Store results
#     results[model_file] = {
#         "MAE": mae,
#         "MSE": mse,
#         "RMSE": rmse,
#         "R2": r2
#     }

# # Print results
# for model_name, metrics in results.items():
#     print(f"Model: {model_name}")
#     for metric_name, value in metrics.items():
#         print(f"  {metric_name}: {value}")
#     print()


import os
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# Define paths
models_dir = "artifacts/models"
test_data_path = "artifacts/test.csv"  # Path to the test CSV file

# Load test data
test_data = pd.read_csv(test_data_path)

# Preprocess test data
# Assuming the target column is "Time_taken (min)" and the rest are features
X_test = test_data.drop(columns=["Time_taken (min)"])
y_test = test_data["Time_taken (min)"]

# Handle missing values if any
X_test = X_test.fillna(0)  # Replace missing values with 0 (adjust as needed)

# Encode categorical features
categorical_columns = X_test.select_dtypes(include=["object"]).columns
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    X_test[col] = le.fit_transform(X_test[col].astype(str))  # Convert to string before encoding
    label_encoders[col] = le

# Initialize results dictionary
results = {}

# Iterate through all models in the directory
for model_file in os.listdir(models_dir):
    model_path = os.path.join(models_dir, model_file)
    
    # Load the model
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    # Store results
    results[model_file] = {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2
    }

# Print results
for model_name, metrics in results.items():
    print(f"Model: {model_name}")
    for metric_name, value in metrics.items():
        print(f"  {metric_name}: {value}")
    print()