
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.impute import SimpleImputer
from scipy.cluster.hierarchy import dendrogram, linkage

# Streamlit page setup
st.set_page_config(page_title="Unsupervised ML Explorer", layout="wide")
st.title("🔍 Unsupervised Machine Learning Explorer")

# File upload and data loading
st.sidebar.header("📁 Upload your dataset")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type="csv")

@st.cache_data
def load_data(file):
    if file is not None:
        return pd.read_csv(file)
    else:
        st.info("No file uploaded. Using default Iris dataset.")
        from sklearn.datasets import load_iris
        iris = load_iris()
        return pd.DataFrame(data=iris.data, columns=iris.feature_names)

df = load_data(uploaded_file)

st.subheader("📊 Original Dataset Overview")
st.write(df.head())

# Feature selection
# For this version, I will consider only numeric columns. Future releases can use OneHotEncoder to
# transform labels to numeric representatios.
st.sidebar.header("⚙️ Feature Selection")
numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
features = st.sidebar.multiselect("Select features for clustering:", options=numeric_cols, default=numeric_cols)

if len(features) < 2:
    st.warning("Please select at least two features.")
    st.stop()

# Missing value handling choice
st.sidebar.header("🧹 Missing Data Handling")
missing_strategy = st.sidebar.radio(
    "Choose how to handle missing values:",
    ("Mean Imputation", "Drop Rows")
)

if df[features].isnull().values.any():
    if missing_strategy == "Mean Imputation":
        st.warning("Missing values detected. Applying mean imputation.")
        imputer = SimpleImputer(strategy='mean')
        df[features] = imputer.fit_transform(df[features])
    elif missing_strategy == "Drop Rows":
        st.warning("Missing values detected. Dropping rows with missing values.")
        df = df.dropna(subset=features)

# Standardize selected features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])

# K-Means clustering
st.sidebar.header("📌 K-Means Parameters")
k = st.sidebar.slider("Number of Clusters (k)", min_value=2, max_value=10, value=3)
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans_labels = kmeans.fit_predict(X_scaled)

silhouette = silhouette_score(X_scaled, kmeans_labels)

# PCA transformation
st.sidebar.header("🎛️ PCA Settings")
max_pca_components = min(3, len(features))
if max_pca_components > 2:
    pca_components = st.sidebar.slider("Number of PCA Components", min_value=2, max_value=max_pca_components, value=2)
else:
    pca_components = 2
    st.sidebar.info(f"PCA will use 2 components since only {len(features)} features are selected.")

pca = PCA(n_components=pca_components)
X_pca = pca.fit_transform(X_scaled)

# Dendrogram with method selection
st.sidebar.header("🧬 Hierarchical Clustering")
max_dendro_rows = 1000
linkage_method = st.sidebar.selectbox("Linkage Method", ["ward", "single", "complete", "average"])

st.subheader("🔎 K-Means Clustering Results")
st.markdown(f"**Silhouette Score:** {silhouette:.2f} (Higher is better)")

pca_df = pd.DataFrame(X_pca, columns=[f"PC{i+1}" for i in range(pca_components)])
pca_df['KMeans Cluster'] = kmeans_labels.astype(str)
fig_kmeans = px.scatter(pca_df, x="PC1", y="PC2", color="KMeans Cluster",
                        title="K-Means Clusters Visualized with PCA",
                        labels={"KMeans Cluster": "Cluster"},
                        color_discrete_sequence=px.colors.qualitative.Safe)
st.plotly_chart(fig_kmeans, use_container_width=True)

# Hierarchical Clustering
if X_scaled.shape[0] <= max_dendro_rows:
    st.subheader("🧬 Dendrogram")
    linked = linkage(X_scaled, method=linkage_method)
    fig, ax = plt.subplots(figsize=(10, 5))
    dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=False)
    st.pyplot(fig)

    # Visualizing Agglomerative Clustering results
    hier_cluster = AgglomerativeClustering(n_clusters=k, linkage=linkage_method)
    hier_labels = hier_cluster.fit_predict(X_scaled)
    pca_df['Hierarchical Cluster'] = hier_labels.astype(str)
    fig_hier = px.scatter(pca_df, x="PC1", y="PC2", color="Hierarchical Cluster",
                          title="Hierarchical Clusters Visualized with PCA",
                          labels={"Hierarchical Cluster": "Cluster"},
                          color_discrete_sequence=px.colors.qualitative.Pastel)
    st.plotly_chart(fig_hier, use_container_width=True)
else:
    st.info(f"Dataset has more than {max_dendro_rows} rows. Dendrogram and hierarchical clustering visualization are disabled for performance reasons.")

# Elbow method
st.subheader("📈 Elbow Method (Inertia vs. K)")
elbow_range = range(1, 11)
inertias = []
for i in elbow_range:
    km = KMeans(n_clusters=i, random_state=42)
    km.fit(X_scaled)
    inertias.append(km.inertia_)
fig_elbow = px.line(x=list(elbow_range), y=inertias, labels={"x": "Number of Clusters", "y": "Inertia"}, title="Elbow Method")
st.plotly_chart(fig_elbow, use_container_width=True)

# Explanations and guidance
with st.expander("❓ How to interpret the results"):
    st.markdown("""
    - **K-Means Clustering** groups data based on similarity. The *Silhouette Score* shows how well points fit within their clusters (range -1 to 1).
    - **PCA** reduces the dataset's dimensions to help visualize clusters in 2D.
    - **Hierarchical Clustering** shows how data points group in a tree-like structure. You can experiment with different linkage methods: *ward*, *single*, *complete*, *average*.
    - **Elbow Method** helps choose a good value for *k* by finding a point where adding more clusters doesn’t significantly reduce inertia.
    """)

# Optional download of results
st.sidebar.header("⬇️ Download Results")
download_df = df.copy()
download_df['KMeans Cluster'] = kmeans_labels
if X_scaled.shape[0] <= max_dendro_rows:
    download_df['Hierarchical Cluster'] = hier_labels

download_csv = download_df.to_csv(index=False)
st.sidebar.download_button(label="Download Clustered Data", data=download_csv, file_name="clustered_data.csv", mime="text/csv")
