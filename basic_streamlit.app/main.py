#!/usr/bin/env/python

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Set title
st.title("Palmer's Penguins Explorer")

# Description
st.write("""
This Streamlit app allows you to explore the Palmer Penguins dataset.  
Use the interactive filters in the sidebar to refine your analysis.
""")

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("data/penguins.csv")

df = load_data()

# Sidebar filters
st.sidebar.header("Filter Options")

species = st.sidebar.multiselect("Select Species", df["species"].dropna().unique(), default=df["species"].dropna().unique())
island = st.sidebar.multiselect("Select Island", df["island"].dropna().unique(), default=df["island"].dropna().unique())
sex = st.sidebar.multiselect("Select Sex", df["sex"].dropna().unique(), default=df["sex"].dropna().unique())
min_mass, max_mass = st.sidebar.slider("Body Mass (g)", 
                                       int(df["body_mass_g"].min()), 
                                       int(df["body_mass_g"].max()), 
                                       (int(df["body_mass_g"].min()), int(df["body_mass_g"].max())))

# Apply filters
filtered_df = df[
    (df["species"].isin(species)) & 
    (df["island"].isin(island)) & 
    (df["sex"].isin(sex)) &
    (df["body_mass_g"] >= min_mass) & 
    (df["body_mass_g"] <= max_mass)
]

# Display filtered data
st.write("### Filtered Data")
st.dataframe(filtered_df)

# Summary Statistics
st.write("### Summary Statistics")
st.write(filtered_df.describe())

# Data Visualization
st.write("### Data Visualizations")

# Histogram of Body Mass
st.write("#### Body Mass Distribution")
fig, ax = plt.subplots()
sns.histplot(filtered_df["body_mass_g"], bins=20, kde=True, ax=ax)
st.pyplot(fig)

# Scatter Plot: Flipper Length vs. Body Mass
st.write("#### Flipper Length vs. Body Mass")
fig, ax = plt.subplots()
sns.scatterplot(data=filtered_df, x="flipper_length_mm", y="body_mass_g", hue="species", style="sex", ax=ax)
st.pyplot(fig)