# app.py

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt

# Title
st.title("Supervised Machine Learning App 🚀")

# Sidebar
st.sidebar.header("1. Upload Your Dataset 📄")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file")

