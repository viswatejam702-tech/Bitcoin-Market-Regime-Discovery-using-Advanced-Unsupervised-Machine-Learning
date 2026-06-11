import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

from sklearn.metrics import silhouette_score

st.set_page_config(
    page_title="Bitcoin Clustering Intelligence",
    page_icon="₿",
    layout="wide"
)

# ==========================
# LOAD FILES
# ==========================

model = joblib.load("kmeans_model.pkl")
scaler = joblib.load("scaler.pkl")
pca = joblib.load("pca.pkl")

df = pd.read_csv(r"clustered_bitcoin.csv")

# ==========================
# SIDEBAR
# ==========================

st.sidebar.image(
    "https://cdn-icons-png.flaticon.com/512/5968/5968260.png",
    width=120
)

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        "Overview",
        "Visualization",
        "Clusters",
        "Predict Cluster"
    ]
)

# ==========================
# OVERVIEW
# ==========================

if page == "Overview":

    st.title(
        "₿ Bitcoin Market Regime Discovery"
    )

    c1,c2,c3 = st.columns(3)

    c1.metric(
        "Rows",
        df.shape[0]
    )

    c2.metric(
        "Columns",
        df.shape[1]
    )

    c3.metric(
        "Clusters",
        df["Cluster"].nunique()
    )

    st.subheader("Dataset")

    st.dataframe(df.head())

    st.subheader("Statistics")

    st.dataframe(
        df.describe()
    )

# ==========================
# VISUALIZATION
# ==========================

elif page == "Visualization":

    st.title("Market Analytics")

    fig = px.scatter(
        df,
        x="Close",
        y="Volume",
        color="Cluster",
        size="Volume",
        title="Volume vs Close"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    fig2 = px.histogram(
        df,
        x="Cluster",
        color="Cluster",
        title="Cluster Distribution"
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

# ==========================
# CLUSTERS
# ==========================

elif page == "Clusters":

    st.title(
        "Cluster Insights"
    )

    cluster_counts = (
        df["Cluster"]
        .value_counts()
        .reset_index()
    )

    cluster_counts.columns = [
        "Cluster",
        "Count"
    ]

    st.dataframe(
        cluster_counts
    )

    fig = px.pie(
        cluster_counts,
        names="Cluster",
        values="Count",
        title="Cluster Distribution"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# ==========================
# PREDICTION
# ==========================

elif page == "Predict Cluster":

    st.title(
        "Cluster Prediction"
    )

    open_price = st.number_input(
        "Open",
        value=10000.0
    )

    high_price = st.number_input(
        "High",
        value=11000.0
    )

    low_price = st.number_input(
        "Low",
        value=9000.0
    )

    close_price = st.number_input(
        "Close",
        value=10500.0
    )

    volume = st.number_input(
        "Volume",
        value=1000000.0
    )

    if st.button(
        "Predict Cluster"
    ):

        daily_return = (
            (close_price-open_price)
            / open_price
        )

        price_range = (
            (high_price-low_price)
            / low_price
        )

        volatility = 0.05

        volume_change = 0.02

        ma7 = close_price

        ma30 = close_price

        momentum = (
            close_price-open_price
        )

        data = np.array([[
            open_price,
            high_price,
            low_price,
            close_price,
            volume,
            daily_return,
            price_range,
            volatility,
            volume_change,
            ma7,
            ma30,
            momentum
        ]])

        scaled = scaler.transform(
            data
        )

        prediction = model.predict(
            scaled
        )[0]

        st.success(
            f"Predicted Cluster: {prediction}"
        )

        st.balloons()