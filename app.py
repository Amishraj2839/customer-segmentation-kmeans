import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from model import predict_segment, kmeans


# -------------------------------
# Page Configuration
# -------------------------------

st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="👥",
    layout="wide"
)


# -------------------------------
# Load Dataset
# -------------------------------

df = pd.read_csv("data/Mall_Customers.csv")


# -------------------------------
# Segment Information
# -------------------------------

segment_info = {
    "Premium Customers": "High income and high spending customers.",
    "Potential Customers": "Lower income with high spending.",
    "Careful Customers": "High income with low spending.",
    "Low-Value Customers": "Lower income and low spending.",
    "Average Customers": "Moderate income and spending."
}


# -------------------------------
# Cluster Colors
# -------------------------------

cluster_colors = {
    0: "#2ca02c",   # Green
    1: "#ff7f0e",   # Orange
    2: "#d62728",   # Red
    3: "#9467bd",   # Purple
    4: "#1f77b4"    # Blue
}


# -------------------------------
# Cluster Names
# -------------------------------

cluster_names = {
    0: "Average Customers",
    1: "Premium Customers",
    2: "Potential Customers",
    3: "Careful Customers",
    4: "Low-Value Customers"
}


# -------------------------------
# Page Title
# -------------------------------

st.title("👥 Customer Segmentation System")

st.write(
    "This application uses **K-Means Clustering** to group customers "
    "based on their annual income and spending score."
)


# -------------------------------
# Dataset Overview
# -------------------------------

st.subheader("📊 Dataset Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Customers", len(df))

with col2:
    st.metric("Number of Features", 2)

with col3:
    st.metric("Number of Segments", 5)

st.divider()


# -------------------------------
# Analyze Customer
# -------------------------------

st.subheader("🔍 Analyze a Customer")

col1, col2 = st.columns(2)

with col1:
    income = st.number_input(
        "Annual Income (k$)",
        min_value=0.0,
        max_value=200.0,
        value=50.0
    )

with col2:
    spending_score = st.number_input(
        "Spending Score (1-100)",
        min_value=1.0,
        max_value=100.0,
        value=50.0
    )


if st.button("Analyze Customer", type="primary"):

    segment = predict_segment(
        income,
        spending_score
    )

    st.success(
        f"Customer Segment: **{segment}**"
    )

    st.info(
        segment_info[segment]
    )


st.divider()


# -------------------------------
# Prepare Clusters
# -------------------------------

X = df[
    ["Annual Income (k$)", "Spending Score (1-100)"]
]

df["Cluster"] = kmeans.predict(X)


# -------------------------------
# Customer Segments
# -------------------------------

st.subheader("📊 Customer Segments")


# Two columns
graph_col, guide_col = st.columns([1.5, 1])


# ==================================================
# LEFT SIDE - GRAPH
# ==================================================

with graph_col:

    fig, ax = plt.subplots(
        figsize=(5, 3.5)
    )

    # Plot each cluster
    for cluster in range(5):

        cluster_data = df[
            df["Cluster"] == cluster
        ]

        ax.scatter(
            cluster_data["Annual Income (k$)"],
            cluster_data["Spending Score (1-100)"],
            s=25,
            color=cluster_colors[cluster],
            alpha=0.85
        )

    # Plot centroids
    ax.scatter(
        kmeans.cluster_centers_[:, 0],
        kmeans.cluster_centers_[:, 1],
        s=100,
        color="black",
        marker="X"
    )

    # Labels
    ax.set_xlabel(
        "Annual Income (k$)",
        fontsize=8
    )

    ax.set_ylabel(
        "Spending Score (1-100)",
        fontsize=8
    )

    ax.set_title(
        "Customer Segmentation using K-Means",
        fontsize=10
    )

    ax.tick_params(
        axis="both",
        labelsize=7
    )

    plt.tight_layout()

    st.pyplot(
        fig,
        use_container_width=False
    )


# -------------------------------
# Segment Guide
# -------------------------------

with guide_col:

    st.markdown("### 📋 Segment Guide")

    st.markdown("""
🟢 **Average Customers**  
<small>Moderate income and spending.</small>

🟠 **Premium Customers**  
<small>High income and high spending.</small>

🔴 **Potential Customers**  
<small>Lower income with high spending.</small>

🟣 **Careful Customers**  
<small>High income with low spending.</small>

🔵 **Low-Value Customers**  
<small>Lower income and low spending.</small>

✖️ **Cluster Center**  
<small>Black X marks represent the center of each group.</small>
""", unsafe_allow_html=True)


# -------------------------------
# Footer
# -------------------------------

st.divider()

st.caption(
    "Machine Learning Project | "
    "K-Means Customer Segmentation | "
    "Python + Streamlit"
)