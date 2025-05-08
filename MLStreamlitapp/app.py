# app.py
# ------------------------------
# Supervised Machine Learning Explorer
# Developed with Streamlit
# ------------------------------

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report, roc_curve, auc,
    mean_squared_error, mean_absolute_error, r2_score
)
import matplotlib.pyplot as plt
import seaborn as sns

# Page setup
st.set_page_config(page_title="Supervised ML App", layout="wide")
st.title("\U0001F52C Supervised Machine Learning Explorer")

# Load Google Fonts
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Titillium+Web:wght@300;400;600;700&display=swap" rel="stylesheet">
<style>
html, body, [class*="css"] { font-family: 'Titillium Web', sans-serif; }
</style>
""", unsafe_allow_html=True)

# ------------------------------
# Dataset Loader
# ------------------------------
@st.cache_data
def load_sample_dataset(name):
    if name == "Iris Dataset":
        from sklearn.datasets import load_iris
        iris = load_iris(as_frame=True)
        return pd.concat([iris.data, pd.Series(iris.target, name='target')], axis=1)
    elif name == "Wine Dataset":
        from sklearn.datasets import load_wine
        wine = load_wine(as_frame=True)
        return pd.concat([wine.data, pd.Series(wine.target, name='target')], axis=1)
    elif name == "Breast Cancer Dataset":
        from sklearn.datasets import load_breast_cancer
        cancer = load_breast_cancer(as_frame=True)
        return pd.concat([cancer.data, pd.Series(cancer.target, name='target')], axis=1)

# ------------------------------
# Sidebar: Data Selection
# ------------------------------
st.sidebar.header("1. Load a Dataset")
sample = st.sidebar.selectbox("Choose a sample dataset or upload your own:", ("None", "Iris Dataset", "Wine Dataset", "Breast Cancer Dataset"))
file = st.sidebar.file_uploader("...or upload a CSV", type=["csv"])

# Load data
data = None
if sample != "None":
    data = load_sample_dataset(sample)
elif file:
    try:
        data = pd.read_csv(file)
    except Exception as e:
        st.error(f"Error loading file: {e}")

# ------------------------------
# Main Interface: Display and Preprocessing
# ------------------------------
if data is not None:
    st.write("## Dataset Preview")
    st.dataframe(data.head())

    st.sidebar.header("2. Select Target Variable")
    target_column = st.sidebar.selectbox("Target column:", data.columns)

    # Preprocessing input features
    X = data.drop(columns=[target_column])
    y = data[target_column]
    X = pd.get_dummies(X)
    X = X.fillna(X.mean())

    # Smart problem type detection
    is_classification = y.nunique() <= 20 and all(y.dropna().astype(int) == y.dropna())

    # Show classification/regression info
    if is_classification:
        st.sidebar.success("Classification detected: Using classification models.")
    else:
        st.sidebar.warning("Regression detected: Using regression models.")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    st.subheader("Target Class Distribution")
    st.bar_chart(y.value_counts())

    # ------------------------------
    # Sidebar: Model Selection and Hyperparameters
    # ------------------------------
    st.sidebar.header("3. Select Model")
    if is_classification:
        model_type = st.sidebar.selectbox("Model", ("Logistic Regression", "Decision Tree", "K-Nearest Neighbors"))
    else:
        model_type = st.sidebar.selectbox("Model", ("Linear Regression", "Decision Tree Regressor", "K-Nearest Regressor"))

    st.sidebar.header("4. Set Hyperparameters")

    # Model initialization
    if model_type == "Logistic Regression":
        C = st.sidebar.slider("Inverse regularization strength (C)", 0.01, 10.0, 1.0)
        solver = st.sidebar.selectbox("Solver", ("lbfgs", "liblinear"))
        model = LogisticRegression(C=C, solver=solver, max_iter=1000)

    elif model_type == "Decision Tree":
        max_depth = st.sidebar.slider("Max Depth", 1, 20, 5)
        criterion = st.sidebar.selectbox("Criterion", ("gini", "entropy"))
        model = DecisionTreeClassifier(max_depth=max_depth, criterion=criterion)

    elif model_type == "K-Nearest Neighbors":
        n_neighbors = st.sidebar.slider("n_neighbors", 1, 20, 5)
        weights = st.sidebar.selectbox("Weights", ("uniform", "distance"))
        p = st.sidebar.selectbox("Distance metric (p)", [1, 2], format_func=lambda x: "Manhattan" if x == 1 else "Euclidean")
        model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, p=p)

    elif model_type == "Linear Regression":
        model = LinearRegression()

    elif model_type == "Decision Tree Regressor":
        max_depth = st.sidebar.slider("Max Depth", 1, 20, 5)
        model = DecisionTreeRegressor(max_depth=max_depth)

    elif model_type == "K-Nearest Regressor":
        n_neighbors = st.sidebar.slider("n_neighbors", 1, 20, 5)
        weights = st.sidebar.selectbox("Weights", ("uniform", "distance"))
        p = st.sidebar.selectbox("Distance metric (p)", [1, 2], format_func=lambda x: "Manhattan" if x == 1 else "Euclidean")
        model = KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights, p=p)

    # ------------------------------
    # Model Training and Evaluation
    # ------------------------------
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.write("## Model Performance")

    if is_classification:
        # 1. Accuracy
        acc = accuracy_score(y_test, y_pred)
        st.write(f"**Accuracy:** {acc:.2f}")

        # 2. Classification Report
        st.text("Classification Report")
        st.code(classification_report(y_test, y_pred))

        # 3. Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        st.write("### Confusion Matrix")
        fig1, ax1 = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        st.pyplot(fig1)

        # 4. Feature Importance
        if model_type == "Decision Tree":
            st.write("### Feature Importance")
            importance_df = pd.DataFrame({
                'Feature': X.columns,
                'Importance': model.feature_importances_
            }).sort_values(by='Importance', ascending=False)
            fig3, ax3 = plt.subplots()
            sns.barplot(x='Importance', y='Feature', data=importance_df, ax=ax3)
            st.pyplot(fig3)

        # 5. ROC Curve (Binary only)
        if y.nunique() == 2:
            try:
                y_prob = model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_prob)
                roc_auc = auc(fpr, tpr)
                fig2, ax2 = plt.subplots()
                ax2.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
                ax2.plot([0, 1], [0, 1], linestyle='--')
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('ROC Curve')
                plt.legend()
                st.pyplot(fig2)
            except Exception as e:
                st.warning(f"ROC curve couldn't be displayed: {e}")
    else:
        # Regression metrics
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        st.write(f"**MSE:** {mse:.2f}, **MAE:** {mae:.2f}, **R² Score:** {r2:.2f}")

    # ------------------------------
    # Cross-validation
    # ------------------------------
    st.write("### Cross-Validation")
    cv_scores = cross_val_score(model, X, y, cv=5)
    st.write(f"Scores: {cv_scores}")
    st.write(f"Mean CV Score: {cv_scores.mean():.2f}")
else:
    st.info("Please load a dataset to get started!")
