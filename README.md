# Bitcoin Market Regime Discovery using Advanced Clustering Analytics

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Machine Learning](https://img.shields.io/badge/Machine-Learning-green)
![Unsupervised Learning](https://img.shields.io/badge/Unsupervised-Learning-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)
![Status](https://img.shields.io/badge/Status-Production-success)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

# Project Overview

Bitcoin markets exhibit complex and dynamic behavior driven by volatility, investor sentiment, trading volume, and macroeconomic factors. Traditional supervised learning approaches require labeled data, which is often unavailable for market regime identification.

This project leverages **Advanced Unsupervised Machine Learning Techniques** to automatically discover hidden market structures and behavioral patterns within Bitcoin historical data.

The system performs extensive feature engineering, dimensionality reduction, clustering analysis, cluster validation, and interactive visualization through a professional Streamlit dashboard.

---

# Business Problem

Financial markets move through multiple hidden regimes such as:

* Bull Markets
* Bear Markets
* Sideways Consolidation
* High Volatility Periods
* Low Volatility Accumulation Zones
* Extreme Trading Activity

Identifying these regimes manually is difficult and subjective.

The objective of this project is to use clustering algorithms to automatically detect hidden market segments and provide actionable insights into Bitcoin market behavior.

---

# Project Architecture

```text
Bitcoin Historical Dataset
            │
            ▼
Data Cleaning & Validation
            │
            ▼
Feature Engineering
            │
            ▼
Data Scaling
            │
            ▼
Dimensionality Reduction (PCA)
            │
            ▼
Multiple Clustering Algorithms
            │
            ▼
Cluster Evaluation Metrics
            │
            ▼
Market Regime Discovery
            │
            ▼
Interactive Streamlit Dashboard
```

---

# Dataset Information

Dataset contains historical Bitcoin trading information including:

| Feature | Description    |
| ------- | -------------- |
| Date    | Trading Date   |
| Open    | Opening Price  |
| High    | Highest Price  |
| Low     | Lowest Price   |
| Close   | Closing Price  |
| Volume  | Trading Volume |

---

# Advanced Feature Engineering

To capture market behavior more effectively, additional financial indicators were created.

### Daily Return

```python
(Close - Open) / Open
```

Measures daily profitability.

---

### Price Range

```python
(High - Low) / Low
```

Measures daily price movement.

---

### Volatility

```python
Rolling Standard Deviation
```

Measures market uncertainty.

---

### Volume Change

```python
Percentage Change in Trading Volume
```

Captures unusual trading activity.

---

### Moving Averages

```python
MA_7
MA_30
```

Short-term and long-term trends.

---

### Momentum

```python
Current Price - Previous Price
```

Captures directional strength.

---

# Machine Learning Pipeline

## Data Cleaning

* Missing Value Handling
* Duplicate Removal
* Feature Validation
* Data Type Correction

---

## Feature Scaling

Standardization using:

```python
StandardScaler()
```

Ensures equal contribution of all variables.

---

## Dimensionality Reduction

Principal Component Analysis (PCA)

Benefits:

* Noise Reduction
* Faster Training
* Better Visualization
* Improved Cluster Separation

---

# Clustering Algorithms Implemented

## K-Means Clustering

Used for centroid-based market segmentation.

### Hyperparameters

```python
KMeans(
    n_clusters=4,
    n_init=20,
    random_state=42
)
```

---

## Agglomerative Clustering

Hierarchical clustering approach.

Benefits:

* Captures nested structures
* Interpretable hierarchy

---

## DBSCAN

Density-based clustering.

Advantages:

* Detects anomalies
* No predefined cluster count

---

## OPTICS

Advanced density-based clustering.

Benefits:

* Handles varying densities
* Robust against noise

---

## Gaussian Mixture Models

Probabilistic clustering approach.

Advantages:

* Soft clustering
* Captures complex distributions

---

## Birch Clustering

Suitable for large-scale datasets.

Advantages:

* Memory efficient
* Fast training

---

# Distance Metrics

The project incorporates important similarity measurements.

## Euclidean Distance

```python
sqrt(sum((x-y)^2))
```

Most common clustering distance metric.

---

## Manhattan Distance

```python
sum(abs(x-y))
```

Useful for high-dimensional spaces.

---

## Cosine Similarity

```python
(A.B)/(||A|| ||B||)
```

Measures directional similarity.

---

# Cluster Validation Metrics

To ensure cluster quality, multiple evaluation methods are used.

---

## Silhouette Score

Measures cluster separation quality.

### Range

```text
-1 to +1
```

Higher is better.

---

## Davies-Bouldin Index

Measures cluster compactness.

Lower values indicate better clustering.

---

## Calinski-Harabasz Score

Measures cluster density and separation.

Higher values indicate better performance.

---

# Visual Analytics

### Elbow Method

Optimal cluster selection.

### PCA Cluster Visualization

2D market regime visualization.

### Cluster Distribution

Cluster population analysis.

### Market Behavior Analysis

Price and volume exploration.

### Interactive Dashboard

Real-time cluster insights.

---

# Streamlit Dashboard Features

## Overview Dashboard

* Dataset Statistics
* Data Quality Report
* Missing Values Analysis

---

## Market Analytics

* Volume Trends
* Price Movements
* Correlation Heatmaps

---

## Clustering Dashboard

* Select Clustering Algorithm
* Visualize Clusters
* Compare Results

---

## Cluster Insights

* Cluster Distribution
* Cluster Centers
* Market Regime Analysis

---

## Prediction Module

Predicts the most likely market regime for new observations.

---

# Project Structure

```text
Bitcoin_Clustering_Project
│
├── bitcoin_dataset.csv
│
├── train_model.py
│
├── app.py
│
├── clustered_bitcoin.csv
│
├── scaler.pkl
│
├── pca.pkl
│
├── kmeans_model.pkl
│
├── requirements.txt
│
└── README.md
```

---

# Installation

Clone Repository

```bash
git clone <repository-url>
```

Move into project folder

```bash
cd Bitcoin_Clustering_Project
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# Training

Train the clustering pipeline.

```bash
python train_model.py
```

Generated files:

```text
kmeans_model.pkl
scaler.pkl
pca.pkl
clustered_bitcoin.csv
```

---

# Launch Dashboard

```bash
streamlit run app.py
```

---

# Technology Stack

* Python
* Pandas
* NumPy
* Scikit-Learn
* Streamlit
* Plotly
* Matplotlib
* Seaborn
* Joblib
* PCA
* Clustering Algorithms

---

# Key Results

✔ Identified hidden Bitcoin market regimes

✔ Reduced dimensionality using PCA

✔ Compared multiple clustering algorithms

✔ Evaluated cluster quality using industry-standard metrics

✔ Built a production-ready analytics dashboard

✔ Implemented scalable machine learning workflow

✔ Created deployable clustering intelligence system

---

# Future Improvements

* HDBSCAN Integration
* AutoML Cluster Selection
* Real-Time Bitcoin API Integration
* Deep Clustering Models
* Market Anomaly Detection Engine
* Explainable AI (XAI) Layer
* Cloud Deployment (AWS/Azure/GCP)
* Docker Containerization

---

# Resume Impact

### Advanced Bitcoin Market Regime Discovery System

Designed and deployed an end-to-end unsupervised machine learning solution for Bitcoin market segmentation using K-Means, DBSCAN, OPTICS, Agglomerative Clustering, Birch, and Gaussian Mixture Models. Engineered financial indicators, applied PCA for dimensionality reduction, evaluated clustering quality using multiple metrics, and developed a production-ready Streamlit analytics platform.

---

# Author
m viswa teja reddy
**Machine Learning & Data Science Portfolio Project**

Focused on:

* Unsupervised Learning
* Financial Analytics
* Data Science
* Machine Learning Research
* Interactive Dashboard Development

⭐ If you found this project useful, consider giving the repository a star.
