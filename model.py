import pandas as pd
from sklearn.cluster import KMeans


# Load dataset
df = pd.read_csv("data/Mall_Customers.csv")


# Select features
X = df[["Annual Income (k$)", "Spending Score (1-100)"]]


# Create and train K-Means model
kmeans = KMeans(
    n_clusters=5,
    random_state=42,
    n_init=10
)

kmeans.fit(X)


# Cluster names
cluster_names = {
    0: "Average Customers",
    1: "Premium Customers",
    2: "Potential Customers",
    3: "Careful Customers",
    4: "Low-Value Customers"
}


def predict_segment(income, spending_score):
    """
    Predict the customer segment based on
    annual income and spending score.
    """

    customer = [[income, spending_score]]

    cluster = kmeans.predict(customer)[0]

    segment = cluster_names[cluster]

    return segment