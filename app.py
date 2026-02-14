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
    st.write(df.dtypes)
    st.write(df.isnull().sum())

    eps = st.slider("Select EPS value", 0.1, 5.0, 0.5)
    min_samples = st.slider("Select Min Samples", 1, 20, 5)

    # create a version of the data with only numbers for the model
    numeric_df=df.select_dtypes(include=['number'])
    

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)
    import numpy as np
    # select only numeric columns
    df_numeric=df.select_dtypes(include=[np.number])

    # fill the missing values
    df_numeric=df_numeric.fillna(df_numeric.mean())

    scaled_data=scaler.fit_transform(df.numeric)

    from sklearn.cluster import DBSCAN

    dbscan=DBSCAN(eps=0.5,min_samples=5)
    clusters = model.fit_predict(scaled_data)

    df_numeric["Cluster"] = clusters

    st.write(df_numeric)
    

    fig, ax = plt.subplots()
    ax.scatter(numeric_df.iloc[:, 0], numeric_df.iloc[:, 1], c=clusters)
    st.pyplot(fig)







