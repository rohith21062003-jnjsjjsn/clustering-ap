import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Original Data", df.head())
else:
    st.stop()

# Select only numeric columns
df_numeric = df.select_dtypes(include=[np.number])

# Handle missing values
df_numeric = df_numeric.fillna(df_numeric.mean())

st.write("Numeric Data Used for Clustering", df_numeric.head())

scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_numeric)

dbscan = DBSCAN(eps=0.5, min_samples=5)
clusters = dbscan.fit_predict(scaled_data)

df_numeric["Cluster"] = clusters
st.write("Clustered Output", df_numeric)

