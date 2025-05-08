# Supervised Machine Learning Streamlit App

A professional and interactive web application built with **Streamlit** that allows users to upload or select datasets, train supervised learning models, tune hyperparameters, and visualize performance metrics — all without writing code.

## Project Overview

This tool supports both **classification** and **regression** tasks, enabling users to:

- Upload their own dataset or choose from built-in options (Iris, Wine, Breast Cancer)
- Select a model: **Logistic Regression**, **Decision Tree**, **K-Nearest Neighbors**, or their regression counterparts
- Automatically detect whether the problem is classification or regression
- Interactively tune model hyperparameters using sliders and dropdowns
- View detailed performance metrics and visualizations

## Features

### Data Handling
- Load your own CSV file or choose a sample dataset
- Auto-encodes categorical variables and fills missing values

### Automatic Task Detection
- Classifies problems as either classification or regression based on the target variable

### Model Selection and Hyperparameter Tuning
- **Classification Models**:
  - Logistic Regression
  - Decision Tree
  - K-Nearest Neighbors
- **Regression Models**:
  - Linear Regression
  - Decision Tree Regressor
  - K-Nearest Regressor
- Adjustable sliders for key hyperparameters (e.g., max depth, number of neighbors, regularization)

### Model Evaluation Metrics
- **Classification**:
  - Accuracy
  - Confusion Matrix
  - ROC Curve (for binary classification)
  - Classification Report
- **Regression**:
  - Mean Squared Error (MSE)
  - Mean Absolute Error (MAE)
  - R² Score

### Cross-Validation
- Displays 5-fold cross-validation scores for each model

### Visual Outputs
- Feature importance chart (Decision Trees)
- Confusion matrix heatmap
- ROC curve plot

## Visual Examples

### Confusion Matrix
![Confusion Matrix](screenshots/confusion_matrix.png)

### ROC Curve
![ROC Curve](screenshots/roc_curve.png)

### User Interface
![Interface](screenshots/interface.png)

## How to Run Locally

1. **Clone the repository**
```bash
git clone https://github.com/gabrielasanchezt/MLStreamlitapp.git
cd MLStreamlitapp
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the app**
```bash
streamlit run app.py
```

## References

- [Scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Logistic Regression](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression)
- [Decision Trees](https://scikit-learn.org/stable/modules/tree.html)
- [K-Nearest Neighbors](https://scikit-learn.org/stable/modules/neighbors.html)
- [Seaborn for Visualizations](https://seaborn.pydata.org/)

## Author
Gabriela Sanchez  
University of Notre Dame  
Email: msanch25@nd.edu  
LinkedIn: [Gabriela Sanchez](https://www.linkedin.com/in/gabriela-sanchez-1b0476225/)

---
This project is part of my data science portfolio. It demonstrates full-stack model deployment and visualization for supervised learning.
