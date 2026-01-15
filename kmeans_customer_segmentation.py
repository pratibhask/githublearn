"""
K-Means Customer Segmentation for Global Superstore

This script loads the Global Superstore Excel workbook, engineers customer-level features,
scales them, determines optimal clusters via the Elbow method and Silhouette Score,
fits a final KMeans model, visualizes clusters via PCA, and saves outputs.
"""
from __future__ import annotations

import os
import sys
import glob
import warnings
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8")
sns.set_palette("viridis")


@dataclass
class Paths:
    data_root: str
    excel_path: str
    output_dir: str


def resolve_paths() -> Paths:
    data_root = os.path.join("/workspace", "data", "GLOBAL SUPER STORE _SALES DATA ANALYSIS")
    # Try to find the Excel workbook
    candidates = [
        os.path.join(data_root, "Global_Superstore_Analysis_Excel.xlsx"),
        os.path.join(data_root, "Global Superstore.xlsx"),
    ]
    excel_path = next((p for p in candidates if os.path.exists(p)), "")
    if not excel_path:
        # Fallback: search for any xlsx under data_root
        found = glob.glob(os.path.join(data_root, "*.xlsx"))
        if found:
            excel_path = found[0]
    if not excel_path or not os.path.exists(excel_path):
        raise FileNotFoundError(
            f"Excel workbook not found under {data_root}. Found candidates: {found if 'found' in locals() else []}"
        )
    output_dir = os.path.join("/workspace", "outputs")
    os.makedirs(output_dir, exist_ok=True)
    return Paths(data_root=data_root, excel_path=excel_path, output_dir=output_dir)


def load_orders_sheet(excel_path: str) -> pd.DataFrame:
    """Load the raw orders sheet from the workbook.

    Preference order:
    1) A known sheet name like 'Global_Superstore_Cleaned' (case-insensitive)
    2) A common name like 'Orders'/'Sales'
    3) Scan all sheets and pick the first that contains required columns
    """
    # Required columns (before standardization) to detect a valid orders sheet
    required_cols_raw = {
        "Order ID", "Order Date", "Customer ID", "Sales", "Quantity", "Profit"
    }

    xls = pd.ExcelFile(excel_path)
    lower_to_orig = {s.lower(): s for s in xls.sheet_names}

    # 1) Prefer explicit cleaned/raw orders sheet
    for preferred in ["Global_Superstore_Cleaned", "Global Superstore", "Orders"]:
        key = preferred.lower()
        if key in lower_to_orig:
            try:
                df_try = pd.read_excel(excel_path, sheet_name=lower_to_orig[key], engine="openpyxl")
                cols = set(map(str, df_try.columns))
                if required_cols_raw.issubset(cols):
                    return df_try
            except Exception:
                pass

    # 2) Try other common names quickly
    for name in ["Order", "Sales", "Sheet1", "Orders_2014"]:
        key = name.lower()
        if key in lower_to_orig:
            try:
                df_try = pd.read_excel(excel_path, sheet_name=lower_to_orig[key], engine="openpyxl")
                cols = set(map(str, df_try.columns))
                if required_cols_raw.issubset(cols):
                    return df_try
            except Exception:
                pass

    # 3) Scan all sheets for required columns
    for sheet in xls.sheet_names:
        try:
            df_try = pd.read_excel(excel_path, sheet_name=sheet, engine="openpyxl")
            cols = set(map(str, df_try.columns))
            if required_cols_raw.issubset(cols):
                return df_try
        except Exception:
            continue

    # If none matched, fallback to first sheet but warn
    print(f"Warning: defaulting to first sheet: {xls.sheet_names[0]}")
    return pd.read_excel(excel_path, sheet_name=xls.sheet_names[0], engine="openpyxl")


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Strip whitespace and lower case
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    return df


def engineer_customer_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    df = standardize_columns(df)

    # Expected columns (common in Global Superstore dataset)
    candidate_cols = {
        "order_id": ["order_id"],
        "order_date": ["order_date", "date"],
        "ship_date": ["ship_date"],
        "customer_id": ["customer_id", "customerid", "cust_id"],
        "customer_name": ["customer_name", "customer"],
        "segment": ["segment"],
        "country": ["country"],
        "city": ["city"],
        "state": ["state"],
        "region": ["region"],
        "sales": ["sales", "sale"],
        "quantity": ["quantity", "qty"],
        "discount": ["discount"],
        "profit": ["profit"],
        "product_id": ["product_id", "productid"],
        "category": ["category"],
        "sub_category": ["sub-category", "sub_category"],
    }

    # Map existing columns
    present = {k: next((c for c in v if c in df.columns), None) for k, v in candidate_cols.items()}

    required_for_features = ["customer_id", "order_id", "order_date", "sales", "quantity", "profit"]
    missing_required = [k for k in required_for_features if not present.get(k)]
    if missing_required:
        raise KeyError(f"Missing required columns for feature engineering: {missing_required}\nAvailable columns: {list(df.columns)}")

    # Parse dates
    for dcol in ["order_date", "ship_date"]:
        colname = present.get(dcol)
        if colname and not np.issubdtype(df[colname].dtype, np.datetime64):
            df[colname] = pd.to_datetime(df[colname], errors="coerce")

    # Compute RFM-like and behavioral features per customer
    df["revenue"] = df[present["sales"]].astype(float)
    df["profit_val"] = df[present["profit"]].astype(float)
    df["order_count_flag"] = 1

    orders_by_customer = df.groupby(present["customer_id"]).agg(
        total_revenue=("revenue", "sum"),
        total_profit=("profit_val", "sum"),
        num_orders=("order_count_flag", "sum"),
        total_quantity=(present["quantity"], "sum"),
        avg_discount=(present["discount"], "mean") if present.get("discount") else ("revenue", lambda x: 0.0),
        first_purchase=(present["order_date"], "min"),
        last_purchase=(present["order_date"], "max"),
        n_unique_products=(present["product_id"], pd.Series.nunique) if present.get("product_id") else ("order_count_flag", lambda x: 0),
    ).reset_index().rename(columns={present["customer_id"]: "customer_id"})

    # Recency in days relative to dataset max date
    max_date = pd.to_datetime(df[present["order_date"]]).max()
    orders_by_customer["recency_days"] = (max_date - orders_by_customer["last_purchase"]).dt.days

    # Monetary and frequency normalized features
    orders_by_customer["avg_order_value"] = orders_by_customer["total_revenue"] / orders_by_customer["num_orders"].replace(0, np.nan)
    orders_by_customer["avg_items_per_order"] = orders_by_customer["total_quantity"] / orders_by_customer["num_orders"].replace(0, np.nan)
    orders_by_customer["profit_margin"] = np.where(
        orders_by_customer["total_revenue"] > 0,
        orders_by_customer["total_profit"] / orders_by_customer["total_revenue"],
        0.0,
    )

    # Fill any NaNs from division by zero
    orders_by_customer = orders_by_customer.fillna(0)

    # Select features for clustering
    feature_cols = [
        "recency_days",
        "num_orders",
        "total_revenue",
        "avg_order_value",
        "avg_items_per_order",
        "profit_margin",
        "total_profit",
        "n_unique_products",
        "avg_discount",
    ]

    features = orders_by_customer[["customer_id"] + feature_cols].copy()
    return features, feature_cols


def scale_features(df: pd.DataFrame, feature_cols: List[str]) -> Tuple[np.ndarray, StandardScaler]:
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(df[feature_cols])
    return data_scaled, scaler


def find_optimal_k(data_scaled: np.ndarray, k_min: int = 2, k_max: int = 10, random_state: int = 42) -> Tuple[List[int], List[float], List[float]]:
    ks = list(range(k_min, k_max + 1))
    inertia_values: List[float] = []
    silhouette_values: List[float] = []

    for k in ks:
        km = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
        labels = km.fit_predict(data_scaled)
        inertia_values.append(km.inertia_)
        # Silhouette is undefined for k=1; here k>=2
        sil = silhouette_score(data_scaled, labels)
        silhouette_values.append(sil)
    return ks, inertia_values, silhouette_values


def plot_k_selection(ks: List[int], inertia_values: List[float], silhouette_values: List[float], output_dir: str) -> Tuple[str, str]:
    elbow_path = os.path.join(output_dir, "k_elbow_inertia.png")
    sil_path = os.path.join(output_dir, "k_silhouette.png")

    plt.figure(figsize=(7, 5))
    plt.plot(ks, inertia_values, marker="o")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Inertia")
    plt.title("Elbow Method: Inertia vs k")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(elbow_path, dpi=150)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.plot(ks, silhouette_values, marker="o", color="red")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Silhouette Score")
    plt.title("Silhouette Score vs k")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(sil_path, dpi=150)
    plt.close()

    return elbow_path, sil_path


def fit_kmeans(data_scaled: np.ndarray, k: int, random_state: int = 42) -> Tuple[np.ndarray, KMeans]:
    model = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
    labels = model.fit_predict(data_scaled)
    return labels, model


def visualize_pca_scatter(data_scaled: np.ndarray, labels: np.ndarray, output_dir: str, title: str = "Customer Segments (PCA)") -> str:
    pca = PCA(n_components=2, random_state=42)
    pcs = pca.fit_transform(data_scaled)

    plt.figure(figsize=(7, 6))
    scatter = plt.scatter(pcs[:, 0], pcs[:, 1], c=labels, cmap="viridis", alpha=0.8, s=35)
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.title(title)
    cbar = plt.colorbar(scatter)
    cbar.set_label("Cluster")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(output_dir, "clusters_pca_scatter.png")
    plt.savefig(out_path, dpi=180)
    plt.close()
    return out_path


def summarize_clusters(df_features: pd.DataFrame, feature_cols: List[str], labels: np.ndarray) -> pd.DataFrame:
    df = df_features.copy()
    df["Cluster"] = labels
    summary = df.groupby("Cluster")[feature_cols].mean().reset_index().sort_values("Cluster")
    return summary


def main():
    paths = resolve_paths()
    print(f"Using workbook: {paths.excel_path}")

    orders_df = load_orders_sheet(paths.excel_path)
    features_df, feature_cols = engineer_customer_features(orders_df)

    # Keep original ids for final output merge
    customer_ids = features_df["customer_id"].values

    data_scaled, scaler = scale_features(features_df, feature_cols)

    # Find optimal k
    ks, inertia_values, silhouette_values = find_optimal_k(data_scaled, 2, 10)
    elbow_path, sil_path = plot_k_selection(ks, inertia_values, silhouette_values, paths.output_dir)

    # Heuristic choice: best silhouette (ties -> smaller k)
    best_idx = int(np.argmax(silhouette_values))
    best_k = ks[best_idx]
    print(f"Best k by silhouette: {best_k} (score={silhouette_values[best_idx]:.3f})")

    # Fit final model
    labels, model = fit_kmeans(data_scaled, best_k)

    # Save labeled customers and centroids
    labeled_df = features_df.copy()
    labeled_df["Cluster"] = labels
    labeled_path = os.path.join(paths.output_dir, "customer_clusters.csv")
    labeled_df.to_csv(labeled_path, index=False)

    # PCA visualization
    pca_path = visualize_pca_scatter(data_scaled, labels, paths.output_dir)

    # Cluster summary
    summary_df = summarize_clusters(features_df, feature_cols, labels)
    summary_path = os.path.join(paths.output_dir, "cluster_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    print("Outputs saved:")
    print(" - Elbow plot:", elbow_path)
    print(" - Silhouette plot:", sil_path)
    print(" - PCA scatter:", pca_path)
    print(" - Labeled customers:", labeled_path)
    print(" - Cluster summary:", summary_path)


if __name__ == "__main__":
    main()
