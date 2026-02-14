import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="DBSCAN Clustering App",
    layout="wide"
)

st.title("📊 DBSCAN Clustering Application")
st.markdown("Upload an Excel file and perform DBSCAN clustering with noise detection.")

st.divider()

# -----------------------------
# File Upload
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload your Excel file (.xlsx)",
    type=["xlsx"]
)

if uploaded_file is not None:

    # -----------------------------
    # Load Data
    # -----------------------------
    df = pd.read_excel(uploaded_file)

    st.subheader("🔍 Original Dataset")
    st.dataframe(df, use_container_width=True)

    # -----------------------------
    # Select Numeric Columns
    # -----------------------------
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

    if len(numeric_cols) < 2:
        st.error("Dataset must contain at least two numeric columns for DBSCAN.")
    else:
        st.subheader("🧮 Select Features for Clustering")

        selected_cols = st.multiselect(
            "Choose numeric columns:",
            numeric_cols,
            default=numeric_cols
        )

        if len(selected_cols) >= 2:

            # -----------------------------
            # Preprocessing
            # -----------------------------
            df_numeric = df[selected_cols].copy()
            df_numeric = df_numeric.fillna(df_numeric.mean())

            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(df_numeric)

            st.success("Data preprocessing and scaling completed successfully.")

            # -----------------------------
            # DBSCAN Parameters
            # -----------------------------
            st.subheader("⚙️ DBSCAN Parameters")

            col1, col2 = st.columns(2)
            with col1:
                eps = st.slider("Epsilon (eps)", 0.1, 5.0, 0.5)
            with col2:
                min_samples = st.slider("Min Samples", 1, 20, 5)

            # -----------------------------
            # Model Training
            # -----------------------------
            if st.button("🚀 Run DBSCAN Clustering"):

                dbscan = DBSCAN(
                    eps=eps,
                    min_samples=min_samples
                )

                clusters = dbscan.fit_predict(scaled_data)

                df_numeric["Cluster"] = clusters

                st.subheader("📌 Cluster Results")

                # -----------------------------
                # Cluster Counts
                # -----------------------------
                st.markdown("### 📊 Cluster Counts")
                cluster_counts = df_numeric["Cluster"].value_counts().sort_index()
                st.write(cluster_counts)

                # -----------------------------
                # Noise Points
                # -----------------------------
                noise_points = (df_numeric["Cluster"] == -1).sum()
                st.markdown(f"### 🚨 Noise Points (-1): **{noise_points}**")

                # -----------------------------
                # -----------------------------
# Cluster Visualization
# -----------------------------
st.subheader("📈 Cluster Visualization")

if len(selected_cols) >= 2:

    x_axis = st.selectbox("Select X-axis", selected_cols, index=0)
    y_axis = st.selectbox("Select Y-axis", selected_cols, index=1)

    fig, ax = plt.subplots()

    scatter = ax.scatter(
        df_numeric[x_axis],
        df_numeric[y_axis],
        c=df_numeric["Cluster"]
    )

    ax.set_xlabel(x_axis)
    ax.set_ylabel(y_axis)
    ax.set_title("DBSCAN Cluster Scatter Plot")

    st.pyplot(fig)

        else:
            st.warning("Please select at least two numeric columns.")
else:
    st.info("👆 Upload an Excel file to get started.")
