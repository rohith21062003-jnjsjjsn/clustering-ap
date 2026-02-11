import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

st.title("DBSCAN Clustering App")

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Dataset Preview")
    st.write(df.head())

    eps = st.slider("Select EPS value", 0.1, 5.0, 0.5)
    min_samples = st.slider("Select Min Samples", 1, 20, 5)

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)

    model = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = model.fit_predict(scaled_data)

    df["Cluster"] = clusters

    st.write("Clustered Data")
    st.write(df)

    fig, ax = plt.subplots()
    ax.scatter(df.iloc[:, 0], df.iloc[:, 1], c=clusters)
    st.pyplot(fig)

streamlit run app.py

