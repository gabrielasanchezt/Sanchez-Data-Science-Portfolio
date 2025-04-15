import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

st.markdown(
    """
    <link href="https://fonts.googleapis.com/css2?family=Titillium+Web:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
    html, body, [class*="css"] {
        font-family: 'Titillium Web', sans-serif;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Supervised Machine Learning App")

st.sidebar.header("1. Upload Your Dataset")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

st.sidebar.header("Or choose a sample dataset")
sample_dataset = st.sidebar.selectbox(
    "Select a sample dataset",
    ("None", "Iris Dataset", "Wine Dataset", "Breast Cancer Dataset")
)

if sample_dataset == "Iris Dataset":
    from sklearn.datasets import load_iris
    iris = load_iris(as_frame=True)
    data = pd.concat([iris.data, pd.Series(iris.target, name='target')], axis=1)
    st.write("### Iris Dataset")
    st.markdown("""
    - **What is it?** Famous dataset used in pattern recognition.
    - **What's in it?** 150 samples of iris flowers.
    - **Goal:** Predict the flower species.
    """)

elif sample_dataset == "Wine Dataset":
    from sklearn.datasets import load_wine
    wine = load_wine(as_frame=True)
    data = pd.concat([wine.data, pd.Series(wine.target, name='target')], axis=1)
    st.write("### Wine Dataset")
    st.markdown("""
    - **What is it?** Wine classification task dataset.
    - **Goal:** Predict wine type based on chemical features.
    """)

elif sample_dataset == "Breast Cancer Dataset":
    from sklearn.datasets import load_breast_cancer
    cancer = load_breast_cancer(as_frame=True)
    data = pd.concat([cancer.data, pd.Series(cancer.target, name='target')], axis=1)
    st.write("### Breast Cancer Dataset")
    st.markdown("""
    - **What is it?** Breast cancer diagnostic dataset.
    - **Goal:** Predict malignancy.
    """)

elif uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("### Uploaded Dataset")

else:
    data = None

if data is not None:
    st.write("## Dataset Preview")
    st.write(data.head())

    if st.checkbox("\U0001F50D Show Pairplot (for small datasets only)", value=False):
        st.write("Generating pairplot... (can take a moment)")
        try:
            fig4 = sns.pairplot(data, hue='target')
            st.pyplot(fig4)
        except:
            st.warning("Could not generate pairplot. May be too many features or unsupported target labels.")

    if data.shape[1] < 2:
        st.error("Dataset doesn't have enough columns.")
    else:
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        st.subheader("Class Distribution in Target")
        st.write(y.value_counts())

        st.sidebar.header("2. Choose a Model")
        model_name = st.sidebar.selectbox("Select Model", ("Logistic Regression", "Decision Tree", "K-Nearest Neighbors"))

        st.sidebar.header("3. Set Hyperparameters")

        if model_name == "Logistic Regression":
            penalty = st.sidebar.selectbox("Penalty", ("l2", "none"))
            C = st.sidebar.slider("Inverse of regularization strength (C)", 0.01, 10.0, 1.0)
            solver = st.sidebar.selectbox("Solver", ("lbfgs", "saga"))
            model = LogisticRegression(penalty=penalty, C=C, solver=solver, max_iter=1000)

        elif model_name == "Decision Tree":
            max_depth = st.sidebar.slider("Max Depth", 1, 20, 5)
            criterion = st.sidebar.selectbox("Criterion", ("gini", "entropy"))
            splitter = st.sidebar.selectbox("Splitter", ("best", "random"))
            min_samples_split = st.sidebar.slider("Min Samples Split", 2, 10, 2)
            model = DecisionTreeClassifier(
                max_depth=max_depth, criterion=criterion, splitter=splitter, min_samples_split=min_samples_split
            )

        elif model_name == "K-Nearest Neighbors":
            n_neighbors = st.sidebar.slider("Number of Neighbors (k)", 1, 20, 5)
            weights = st.sidebar.selectbox("Weights", ("uniform", "distance"))
            p = st.sidebar.selectbox("Distance Metric (p)", [1, 2], format_func=lambda x: "Manhattan" if x == 1 else "Euclidean")
            model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, p=p)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        st.write("## Model Performance")
        accuracy = accuracy_score(y_test, y_pred)
        st.write(f"**Accuracy:** {accuracy:.2f}")

        st.write("### Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig1, ax1 = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        st.pyplot(fig1)

        st.subheader("Results Interpretation")
        st.markdown(f"""
        **Accuracy Interpretation:**
        - Your model predicts correctly about **{accuracy*100:.2f}%** of the time.

        **Confusion Matrix Interpretation:**
        - Diagonal = correct predictions; off-diagonal = misclassifications.
        """)

        if model_name == "Decision Tree":
            st.write("### Feature Importance")
            importance_df = pd.DataFrame({
                'feature': X.columns,
                'importance': model.feature_importances_
            }).sort_values(by='importance', ascending=False)

            fig3, ax3 = plt.subplots()
            sns.barplot(x='importance', y='feature', data=importance_df, ax=ax3)
            st.pyplot(fig3)

        if len(y.unique()) == 2:
            st.write("### ROC Curve")
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            roc_auc = auc(fpr, tpr)

            fig2, ax2 = plt.subplots()
            ax2.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
            ax2.plot([0, 1], [0, 1], linestyle='--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.legend()
            st.pyplot(fig2)

            st.markdown(f"""
            **ROC/AUC Interpretation:**
            - AUC close to **1** is ideal. **0.5** is random.
            - Your model's AUC is **{roc_auc:.2f}**.
            """)
else:
    st.info("Please upload a dataset or select a sample dataset to get started!")
