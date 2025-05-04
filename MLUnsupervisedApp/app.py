# app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
import seaborn as sns

# Set Streamlit page configuration
st.set_page_config(page_title="Unsupervised ML Explorer", layout="wide")

# Title and Description
st.title(" Unsupervised Machine Learning Explorer")
st.markdown("Explore clustering and dimensionality reduction techniques with your own data or the Iris dataset.")

# Upload dataset or use fallback
uploaded_file = st.file_uploader("📁 Upload your CSV file", type=["csv"])

# If no file is uploaded, use Iris dataset
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success(" File uploaded successfully!")
else:
    st.info("ℹ No file uploaded. Using sample Iris dataset.")
    from sklearn.datasets import load_iris
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)

# Show preview of dataset
st.write("### Dataset Preview")
st.dataframe(df.head())

# Select features for analysis
features = st.multiselect(" Select features for analysis", df.columns.tolist(), default=df.columns.tolist())

# Require at least 2 features
if len(features) < 2:
    st.warning(" Please select at least two features to continue.")
    st.stop()

X = df[features]

# Sidebar: Model Controls
st.sidebar.title(" Model Parameters")
num_clusters = st.sidebar.slider("Number of Clusters", min_value=2, max_value=10, value=3)

# --- K-MEANS CLUSTERING ---
st.subheader(" K-Means Clustering")

# Train K-Means model
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans_labels = kmeans.fit_predict(X)

# Evaluate performance using silhouette score
score = silhouette_score(X, kmeans_labels)
st.write(f"Silhouette Score: **{score:.3f}**")

# Scatter plot of clustering results
fig_kmeans, ax_kmeans = plt.subplots()
sns.scatterplot(x=X.iloc[:, 0], y=X.iloc[:, 1], hue=kmeans_labels, palette="Set2", ax=ax_kmeans)
ax_kmeans.set_title("K-Means Clustering Visualization")
st.pyplot(fig_kmeans)
plt.clf()  # Clear figure memory

# --- HIERARCHICAL CLUSTERING ---
st.subheader("🔸 Hierarchical Clustering")

try:
    # Generate linkage matrix for dendrogram
    linkage_matrix = linkage(X, method="ward")

    fig_dendro, ax_dendro = plt.subplots(figsize=(10, 4))
    dendrogram(linkage_matrix, ax=ax_dendro)
    ax_dendro.set_title("Dendrogram (Ward Linkage)")
    st.pyplot(fig_dendro)
    plt.clf()
except Exception as e:
    st.error(f" Error in Hierarchical Clustering: {e}")

# --- PRINCIPAL COMPONENT ANALYSIS (PCA) ---
st.subheader("Principal Component Analysis (PCA)")

# Reduce to 2 principal components
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Plot PCA result with K-Means labels
fig_pca, ax_pca = plt.subplots()
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=kmeans_labels, palette="Set2", ax=ax_pca)
ax_pca.set_title("PCA (Colored by K-Means Clusters)")
st.pyplot(fig_pca)
plt.clf()

# --- End of App ---
