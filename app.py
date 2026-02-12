import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

st.title("DBSCAN Clustering App")

uploaded_file = st.file_uploader("Upload File", type=["csv","xlsx"])

if uploaded_file is not None:
    if uploaded_file.name.endswith('.csv'):
        df=pd.read_csv(uploaded_file)
    else:
        df=pd.read_excel(uploaded_file)
    st.write("Dataset Preview")
    st.write(df.head())

    eps = st.slider("Select EPS value", 0.1, 5.0, 0.5)
    min_samples = st.slider("Select Min Samples", 1, 20, 5)

    # create a version of the data with only numbers for the model
    numeric_df=df.select_dtypes(include=['number'])
    

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)
    #select only numeric columns for the math
    numeric_df=df.select_dtypes(include=['float64','int64'])
    scaled_data=scaler.fit_transform(numeric_df)

    model = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = model.fit_predict(scaled_data)

    df["Cluster"] = clusters

    st.write("Clustered Data")
    st.write(df)

    fig, ax = plt.subplots()
    ax.scatter(numeric_df.iloc[:, 0], numeric_df.iloc[:, 1], c=clusters)
    st.pyplot(fig)





