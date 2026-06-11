import pandas as pd
import numpy as np
import joblib

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# =========================
# LOAD DATA
# =========================

df = pd.read_csv(r"C:\ds and AI\ALL DATASETS\bitcoin_dataset.csv")

# =========================
# CLEAN DATA
# =========================

numeric_cols = df.select_dtypes(include=np.number).columns

imputer = SimpleImputer(strategy="median")

df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

# =========================
# FEATURE ENGINEERING
# =========================

df['Daily_Return'] = (
    (df['Close'] - df['Open']) / df['Open']
)

df['Price_Range'] = (
    (df['High'] - df['Low']) / df['Low']
)

df['Volatility'] = (
    df['Close'].rolling(7).std()
)

df['Volume_Change'] = (
    df['Volume'].pct_change()
)

df['MA_7'] = (
    df['Close'].rolling(7).mean()
)

df['MA_30'] = (
    df['Close'].rolling(30).mean()
)

df['Momentum'] = (
    df['Close'] - df['Close'].shift(10)
)

df = df.bfill()
df = df.ffill()

# =========================
# FEATURES
# =========================

features = [
    'Open',
    'High',
    'Low',
    'Close',
    'Volume',
    'Daily_Return',
    'Price_Range',
    'Volatility',
    'Volume_Change',
    'MA_7',
    'MA_30',
    'Momentum'
]

X = df[features]

# =========================
# SCALE
# =========================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# =========================
# PCA
# =========================

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)

# =========================
# KMEANS
# =========================

model = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=20
)

clusters = model.fit_predict(X_scaled)

df["Cluster"] = clusters

# =========================
# SAVE FILES
# =========================

joblib.dump(model,"kmeans_model.pkl")
joblib.dump(scaler,"scaler.pkl")
joblib.dump(pca,"pca.pkl")

df.to_csv(
    "clustered_bitcoin.csv",
    index=False
)

print("Training Complete")
print("Files Saved Successfully")