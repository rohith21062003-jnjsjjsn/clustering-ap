import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx"])

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)
    st.write("Original Data", df.head())
else:
    st.stop()

# Select numeric columns
df_numeric = df.select_dtypes(include=[np.number])

# Fill missing values
df_numeric = df_numeric.fillna(df_numeric.mean())

# Scaling
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_numeric)

# DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
clusters = dbscan.fit_predict(scaled_data)

# Add cluster column
df_numeric["Cluster"] = clusters

# Show clustered data
st.write("Clustered Output", df_numeric)

# Show cluster counts (NOW it will work)
st.write("Cluster Counts:")
st.write(df_numeric["Cluster"].value_counts())

st.write("Cluster counts:")
st.write(df_numeric["Cluster"].value_counts())

# Show number of noise points
noise_points = (df_numeric["Cluster"] == -1).sum()
st.write("Number of Noise Points (-1):", noise_points)





