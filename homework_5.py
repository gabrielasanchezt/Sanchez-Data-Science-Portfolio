import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Part 1: Data Loading and Exploration
# Load the California Housing dataset
housing = fetch_california_housing()

# Create DataFrame for features and Series for target
X = pd.DataFrame(housing.data, columns=housing.feature_names)
y = pd.Series(housing.target, name='med_house_value')

# Display first five rows
print("First five rows of the dataset:")
print(X.head())

# Print feature names and check for missing values
print("\nFeature Names:", X.columns.tolist())
print("\nMissing Values:")
print(X.isnull().sum())

# Generate summary statistics
print("\nSummary Statistics:")
print(X.describe())

# Part 2: Linear Regression on Unscaled Data
# Split dataset into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model performance
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance on Unscaled Data:")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R² Score: {r2:.4f}")

# Interpretation Questions (to be answered in Markdown format):
# 1. The R² score indicates how well the independent variables explain the variability of the target variable.
# 2. The features with the largest coefficients in absolute value have the strongest impact on predictions.
# 3. If RMSE is low and R² is high, the predictions match actual values well.

# Part 3: Feature Scaling and Impact Analysis
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model_scaled = LinearRegression()
model_scaled.fit(X_train_scaled, y_train)

y_pred_scaled = model_scaled.predict(X_test_scaled)

mse_scaled = mean_squared_error(y_test, y_pred_scaled)
rmse_scaled = np.sqrt(mse_scaled)
r2_scaled = r2_score(y_test, y_pred_scaled)

print("\nModel Performance on Scaled Data:")
print(f"Mean Squared Error (MSE): {mse_scaled:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse_scaled:.4f}")
print(f"R² Score: {r2_scaled:.4f}")

# Part 4: Feature Selection and Simplified Model
# Select three key features
selected_features = ['MedInc', 'AveRooms', 'HouseAge']

X_train_simple = X_train[selected_features]
X_test_simple = X_test[selected_features]

model_simple = LinearRegression()
model_simple.fit(X_train_simple, y_train)

y_pred_simple = model_simple.predict(X_test_simple)

mse_simple = mean_squared_error(y_test, y_pred_simple)
rmse_simple = np.sqrt(mse_simple)
r2_simple = r2_score(y_test, y_pred_simple)

print("\nSimplified Model Performance:")
print(f"Mean Squared Error (MSE): {mse_simple:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse_simple:.4f}")
print(f"R² Score: {r2_simple:.4f}")

# Answer interpretation questions in Markdown.
