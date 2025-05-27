# Step 1: Import all required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import AgglomerativeClustering
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
import warnings
import logging
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth
from mlxtend.preprocessing import TransactionEncoder
import networkx as nx
import os
from datetime import datetime
import sys
from io import StringIO

# Suppress warnings
warnings.filterwarnings('ignore')
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)

# Create output directories
output_dir = "plots"
logs_dir = "logs"
for directory in [output_dir, logs_dir]:
    if not os.path.exists(directory):
        os.makedirs(directory)

# Set up logging
log_filename = os.path.join(logs_dir, f"retail_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler(sys.stdout)
    ]
)

# Create a custom print function that logs to file
def log_print(*args, **kwargs):
    output = StringIO()
    print(*args, file=output, **kwargs)
    contents = output.getvalue()
    logging.info(contents.strip())
    print(*args, **kwargs)

# Set global style
plt.style.use('seaborn-v0_8')  # Using a specific seaborn style version
sns.set_theme(style="whitegrid")  # Setting seaborn theme
plt.rcParams['figure.figsize'] = [12, 6]
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

# Function to save plots
def save_plot(filename, dpi=300, bbox_inches='tight'):
    plt.savefig(os.path.join(output_dir, filename), dpi=dpi, bbox_inches=bbox_inches)
    plt.close()

# Function to enhance plot appearance
def enhance_plot(title, xlabel=None, ylabel=None, grid=True):
    plt.title(title, pad=20, fontsize=14, fontweight='bold')
    if xlabel:
        plt.xlabel(xlabel, fontsize=12)
    if ylabel:
        plt.ylabel(ylabel, fontsize=12)
    if grid:
        plt.grid(True, alpha=0.3)
    plt.tight_layout()

# Step 2: Data Loading and Initial Exploration
def load_and_explore_data():
    # Set file path
    file_path = r'C:\Users\milit\OneDrive\Desktop\retail\online_retail_II.csv'
    
    # Read CSV file
    df = pd.read_csv(file_path, low_memory=False)

    # Show basic structure
    log_print("Dataset shape:", df.shape)
    log_print("\nFirst few rows:")
    log_print(df.head())
    
    # Basic info
    log_print("\nDataset info:")
    df.info(buf=StringIO())

    # Check for missing values
    missing_values = df.isnull().sum()
    log_print("\nMissing values per column:\n", missing_values)

    # Check number of unique countries
    log_print("\nUnique countries:", df['Country'].nunique())
    log_print("\nTop countries by transaction count:")
    log_print(df['Country'].value_counts().head())
    
    return df

# Step 3: Data Cleaning
def clean_data(df):
    # Remove cancelled transactions
    df_clean = df[~df['Invoice'].astype(str).str.startswith('C')]

    # Drop rows with missing Customer ID or Description
    df_clean = df_clean.dropna(subset=['Customer ID', 'Description'])

    # Remove duplicates
    df_clean = df_clean.drop_duplicates()

    # Convert InvoiceDate to datetime
    df_clean['InvoiceDate'] = pd.to_datetime(df_clean['InvoiceDate'])

    # Reset index
    df_clean.reset_index(drop=True, inplace=True)

    log_print("Cleaned dataset shape:", df_clean.shape)
    log_print("\nRemaining missing values:\n", df_clean.isnull().sum())
    
    return df_clean

# Step 4: Feature Engineering
def engineer_features(df_clean):
    # Create TotalPrice
    df_clean['TotalPrice'] = df_clean['Quantity'] * df_clean['Price']

    # Create date-based features
    df_clean['Year'] = df_clean['InvoiceDate'].dt.year
    df_clean['Month'] = df_clean['InvoiceDate'].dt.month
    df_clean['Day'] = df_clean['InvoiceDate'].dt.day
    df_clean['Hour'] = df_clean['InvoiceDate'].dt.hour
    df_clean['Weekday'] = df_clean['InvoiceDate'].dt.dayofweek

    # RFM Analysis Preparation
    snapshot_date = df_clean['InvoiceDate'].max() + pd.Timedelta(days=1)

    rfm = df_clean.groupby('Customer ID').agg({
        'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
        'Invoice': 'nunique',
        'TotalPrice': 'sum'
    }).reset_index()

    rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']

    log_print("\nRFM sample:\n", rfm.head())
    
    return df_clean, rfm

# Step 5: Data Visualization
def visualize_data(df_clean, rfm):
    # 1. Top 10 countries (excluding UK)
    plt.figure(figsize=(12, 6))
    top_countries = df_clean[df_clean['Country'] != 'United Kingdom']['Country'].value_counts().head(10)
    sns.barplot(x=top_countries.values, y=top_countries.index, palette='viridis')
    enhance_plot('Top 10 Countries by Transactions (excluding UK)', 'Number of Transactions', 'Country')
    save_plot('01_top_countries.png')

    # 2. Top 10 selling products
    plt.figure(figsize=(12, 6))
    top_products = df_clean['Description'].value_counts().head(10)
    sns.barplot(x=top_products.values, y=top_products.index, palette='magma')
    enhance_plot('Top 10 Selling Products', 'Quantity Sold', 'Product')
    save_plot('02_top_products.png')

    # 3. Monthly sales trend
    plt.figure(figsize=(15, 6))
    monthly_sales = df_clean.set_index('InvoiceDate').resample('M')['TotalPrice'].sum()
    plt.plot(monthly_sales.index, monthly_sales.values, linewidth=2, color='#2ecc71')
    enhance_plot('Monthly Sales Trend', 'Month', 'Total Sales (£)')
    save_plot('03_monthly_sales.png')

    # 4. RFM Histograms
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    sns.histplot(rfm['Recency'], bins=30, ax=axs[0], kde=True, color='#3498db')
    axs[0].set_title('Recency Distribution', pad=20, fontsize=12, fontweight='bold')

    sns.histplot(rfm['Frequency'], bins=30, ax=axs[1], kde=True, color='#e74c3c')
    axs[1].set_title('Frequency Distribution', pad=20, fontsize=12, fontweight='bold')

    sns.histplot(rfm['Monetary'], bins=30, ax=axs[2], kde=True, color='#2ecc71')
    axs[2].set_title('Monetary Distribution', pad=20, fontsize=12, fontweight='bold')

    plt.tight_layout()
    save_plot('04_rfm_distributions.png')

# Step 6: Enhanced Customer Segmentation
def perform_customer_segmentation(rfm):
    # Standardize RFM
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])

    # 1. K-Means Clustering
    sse = {}
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(rfm_scaled)
        sse[k] = kmeans.inertia_

    # Plot Elbow Curve
    plt.figure(figsize=(10, 6))
    plt.plot(list(sse.keys()), list(sse.values()), marker='o', linewidth=2, color='#3498db')
    enhance_plot('Elbow Method For Optimal k', 'Number of Clusters', 'SSE (Inertia)')
    save_plot('05_elbow_curve.png')

    # Apply K-Means with 3 clusters
    kmeans = KMeans(n_clusters=3, random_state=42)
    rfm['KMeans_Cluster'] = kmeans.fit_predict(rfm_scaled)
    
    # 2. Hierarchical Clustering
    linked = linkage(rfm_scaled, method='ward')
    plt.figure(figsize=(12, 8))
    dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=False)
    enhance_plot('Customer Dendrogram (Hierarchical Clustering)', 'Customer Index', 'Distance')
    save_plot('06_dendrogram.png')
    
    # Apply Agglomerative clustering
    hc = AgglomerativeClustering(n_clusters=3)
    rfm['Hierarchical_Cluster'] = hc.fit_predict(rfm_scaled)
    
    # 3. DBSCAN (Density-Based)
    db = DBSCAN(eps=1.5, min_samples=5)
    rfm['DBSCAN_Cluster'] = db.fit_predict(rfm_scaled)
    
    # Count noise points
    noise_count = sum(rfm['DBSCAN_Cluster'] == -1)
    log_print(f"DBSCAN noise points (outliers): {noise_count}")
    
    # 4. PCA for Visualization
    pca = PCA(n_components=2)
    components = pca.fit_transform(rfm_scaled)
    rfm['PCA1'] = components[:, 0]
    rfm['PCA2'] = components[:, 1]
    
    # Plot clusters
    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=rfm, x='PCA1', y='PCA2', hue='KMeans_Cluster', 
                   palette='viridis', s=100, alpha=0.6)
    enhance_plot('Customer Segments by K-Means (PCA Projection)', 'PCA1', 'PCA2')
    save_plot('07_customer_segments.png')
    
    # Cluster Interpretation
    cluster_summary = rfm.groupby('KMeans_Cluster').agg({
        'Recency': 'mean',
        'Frequency': 'mean',
        'Monetary': 'mean',
        'CustomerID': 'count'
    }).rename(columns={'CustomerID': 'Count'}).round(2)

    log_print("\nCluster Summary:")
    log_print(cluster_summary)
    
    # Label clusters
    def interpret_cluster(row):
        if row['KMeans_Cluster'] == 0:
            return 'High-Value Loyal'
        elif row['KMeans_Cluster'] == 1:
            return 'Low-Spend Recent'
        elif row['KMeans_Cluster'] == 2:
            return 'Churned'
        else:
            return 'Other'
    
    rfm['Segment'] = rfm.apply(interpret_cluster, axis=1)
    
    return rfm

# Step 7: Classification Analysis
def perform_classification_analysis(df_clean, rfm):
    # Join original df_clean with RFM to include Country
    rfm_with_country = df_clean[['Customer ID', 'Country']].drop_duplicates()
    rfm_full = rfm.merge(rfm_with_country, left_on='CustomerID', right_on='Customer ID', how='left').drop(columns=['Customer ID'])

    # Label target variable (High-value customers)
    threshold = rfm_full['Monetary'].quantile(0.75)
    rfm_full['HighValueCustomer'] = (rfm_full['Monetary'] >= threshold).astype(int)

    # One-hot encode Country
    rfm_encoded = pd.get_dummies(rfm_full, columns=['Country'], drop_first=True)

    # Select only numeric features for classification
    numeric_features = ['Recency', 'Frequency', 'Monetary']
    X = rfm_encoded[numeric_features]
    y = rfm_encoded['HighValueCustomer']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Define models
    models = {
        'Naive Bayes': GaussianNB(),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
        'Support Vector Machine': SVC(kernel='rbf', probability=True, random_state=42)
    }
    
    # Evaluate each model
    results = []
    for name, model in models.items():
        # Train
        model.fit(X_train_scaled, y_train)
        
        # Predict
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else None
        
        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc = roc_auc_score(y_test, y_prob) if y_prob is not None else None
        
        results.append({
            'Model': name,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1 Score': f1,
            'ROC AUC': roc
        })
        
        # Confusion matrix
        log_print(f"\n{name} - Classification Report:")
        log_print(classification_report(y_test, y_pred))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'{name} - Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        save_plot(f'classification_{name.lower().replace(" ", "_")}_confusion_matrix.png')
    
    # ROC Curve
    if y_prob is not None:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {name}')
        plt.legend(loc='lower right')
        save_plot(f'roc_curve_{name.lower().replace(" ", "_")}.png')
    
    results_df = pd.DataFrame(results)
    log_print("\nModel Evaluation Summary:")
    log_print(results_df)
    
    return results_df

# Step 8: Business Insights and Recommendations
def generate_business_insights(df_clean, rfm, classification_results):
    log_print("\n=== Business Insights and Recommendations ===\n")
    
    # 1. Customer Segment Analysis
    segment_summary = rfm.groupby('Segment').agg({
        'CustomerID': 'count',
        'Monetary': 'mean',
        'Frequency': 'mean',
        'Recency': 'mean'
    }).round(2)
    
    log_print("1. Customer Segment Analysis:")
    log_print(segment_summary)
    log_print("\nRecommendations:")
    log_print("- High-Value Loyal: Focus on retention and upselling")
    log_print("- Low-Spend Recent: Encourage higher purchase frequency")
    log_print("- Churned: Implement re-engagement campaigns")
    
    # 2. Product Analysis
    top_products = df_clean.groupby('Description')['Quantity'].sum().sort_values(ascending=False).head(10)
    log_print("\n2. Top 10 Products by Quantity:")
    log_print(top_products)

    # 3. Seasonal Trends
    monthly_sales = df_clean.groupby(df_clean['InvoiceDate'].dt.month)['TotalPrice'].mean()
    peak_months = monthly_sales.nlargest(3)
    log_print("\n3. Peak Sales Months:")
    log_print(peak_months)
    log_print("\nRecommendation: Plan promotions and inventory accordingly")
    
    # 4. Classification Insights
    best_model = classification_results.loc[classification_results['F1 Score'].idxmax()]
    log_print("\n4. Best Performing Model for High-Value Customer Prediction:")
    log_print(f"Model: {best_model['Model']}")
    log_print(f"F1 Score: {best_model['F1 Score']:.4f}")
    log_print("\nRecommendation: Use this model for targeted marketing campaigns")
    
    # 5. Country-wise Analysis
    country_analysis = df_clean.groupby('Country').agg({
        'Customer ID': 'nunique',
        'TotalPrice': 'sum'
    }).sort_values('TotalPrice', ascending=False)
    
    log_print("\n5. Country-wise Analysis:")
    log_print(country_analysis.head())
    log_print("\nRecommendation: Focus marketing efforts on high-value countries")
    
    return {
        'segment_summary': segment_summary,
        'top_products': top_products,
        'peak_months': peak_months,
        'best_model': best_model,
        'country_analysis': country_analysis
    }

# Step 9: Time Series Analysis
def perform_time_series_analysis(df_clean):
    # Aggregate daily sales
    df_daily_sales = df_clean.groupby(df_clean['InvoiceDate'].dt.date)['TotalPrice'].sum()
    df_daily_sales = df_daily_sales.to_frame()
    df_daily_sales.index = pd.to_datetime(df_daily_sales.index)
    df_daily_sales = df_daily_sales.asfreq('D')

    # Plot daily sales
    plt.figure(figsize=(15, 6))
    plt.plot(df_daily_sales.index, df_daily_sales['TotalPrice'], 
             linewidth=2, color='#2ecc71', alpha=0.8)
    enhance_plot('Daily Sales Trend', 'Date', 'Total Sales (£)')
    save_plot('08_daily_sales.png')

    # Seasonal decomposition
    df_daily_sales['TotalPrice'] = df_daily_sales['TotalPrice'].fillna(method='ffill')
    decomposition = seasonal_decompose(df_daily_sales['TotalPrice'], model='multiplicative', period=30)
    decomposition.plot()
    plt.tight_layout()
    save_plot('09_seasonal_decomposition.png')

    # ARIMA Forecast
    model = ARIMA(df_daily_sales['TotalPrice'], order=(5, 1, 0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=30)

    plt.figure(figsize=(15, 6))
    plt.plot(df_daily_sales.index, df_daily_sales['TotalPrice'], 
             label='Historical Sales', color='#3498db', alpha=0.8)
    plt.plot(pd.date_range(df_daily_sales.index[-1] + pd.Timedelta(days=1), 
                          periods=30, freq='D'), forecast, 
             label='Forecast', color='#e74c3c', linewidth=2)
    enhance_plot('ARIMA Forecast of Daily Sales (Next 30 Days)', 'Date', 'Sales (£)')
    plt.legend(fontsize=12)
    save_plot('10_arima_forecast.png')
    
    # Prophet Forecast
    df_prophet = df_daily_sales.reset_index()
    df_prophet = df_prophet.rename(columns={'InvoiceDate': 'ds', 'TotalPrice': 'y'})
    df_prophet = df_prophet[['ds', 'y']].dropna()

    prophet_model = Prophet(daily_seasonality=True)
    prophet_model.fit(df_prophet)

    future = prophet_model.make_future_dataframe(periods=30)
    forecast_prophet = prophet_model.predict(future)

    fig = prophet_model.plot(forecast_prophet)
    plt.title('Prophet Forecast of Daily Sales (Next 30 Days)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '11_prophet_forecast.png'), dpi=300, bbox_inches='tight')
    plt.close()

# Step 6: Association Rules Analysis
def analyze_association_rules(df_clean):
    # Prepare data for France
    basket = (df_clean[df_clean['Country'] == 'France']
              .groupby(['Invoice', 'Description'])['Quantity']
              .sum().unstack().fillna(0))
    
    # Convert to boolean
    basket = basket > 0
    
    # Apply FP-Growth
    frequent_itemsets = fpgrowth(basket, min_support=0.02, use_colnames=True)
    
    # Generate association rules
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
    
    # Filter strong rules
    strong_rules = rules[(rules['lift'] > 1.2) & (rules['confidence'] > 0.5)]
    
    # Create node-label mapping (numeric)
    items = set()
    for _, row in strong_rules.iterrows():
        items.update(row['antecedents'])
        items.update(row['consequents'])
    item_to_id = {item: f'I{i+1}' for i, item in enumerate(sorted(items, key=str))}
    id_to_item = {v: k for k, v in item_to_id.items()}
    
    # Build graph with numeric labels
    plt.figure(figsize=(12, 8))
    G = nx.DiGraph()
    for _, row in strong_rules.iterrows():
        for antecedent in row['antecedents']:
            for consequent in row['consequents']:
                G.add_edge(item_to_id[antecedent], item_to_id[consequent], weight=row['lift'])
    pos = nx.spring_layout(G, k=0.5, iterations=20)
    edges = G.edges(data=True)
    weights = [edge[2]['weight'] for edge in edges]
    nx.draw(G, pos, with_labels=True, node_size=2000, node_color='lightblue',
            font_size=12, font_weight='bold', edge_color=weights, width=2.0, edge_cmap=plt.cm.Blues)
    plt.title("Association Rules Network Graph (France) - Item IDs", fontsize=16)
    plt.savefig(os.path.join(output_dir, '12_association_rules_network.png'), dpi=300, bbox_inches='tight')
    plt.close()
    # Save top 10 rules as CSV
    strong_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10).to_csv(os.path.join(output_dir, '13_top10_association_rules.csv'), index=False)
    # Print mapping for interpretation
    log_print("\nItem ID Mapping:")
    for item, label in item_to_id.items():
        log_print(f"{label}: {item}")
    # Print top rules
    log_print("\nTop Association Rules:")
    log_print(strong_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head())
    return strong_rules

# --- ENHANCED EDA ---
def enhanced_eda(df_clean, rfm):
    # 1. Top customers
    plt.figure(figsize=(12, 6))
    top_customers = df_clean.groupby('Customer ID')['TotalPrice'].sum().sort_values(ascending=False).head(10)
    sns.barplot(x=top_customers.values, y=top_customers.index.astype(int), palette='viridis')
    enhance_plot('Top 10 Customers by Total Spend', 'Total Spend (£)', 'Customer ID')
    save_plot('11_top_customers.png')
    
    # 2. Order sizes
    plt.figure(figsize=(12, 6))
    order_sizes = df_clean.groupby('Invoice')['Quantity'].sum()
    sns.histplot(order_sizes, bins=50, kde=True, color='#e74c3c')
    enhance_plot('Distribution of Order Sizes', 'Number of Items', 'Number of Orders')
    save_plot('12_order_sizes.png')
    
    # 3. Heatmap of sales by weekday and hour (fixed)
    df_clean['Weekday'] = df_clean['InvoiceDate'].dt.day_name()
    df_clean['Hour'] = df_clean['InvoiceDate'].dt.hour
    sales_pivot = df_clean.pivot_table(index='Hour', columns='Weekday', values='TotalPrice', aggfunc='sum').fillna(0)
    # Reorder columns to standard weekday order
    sales_pivot = sales_pivot[['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']]
    plt.figure(figsize=(12, 6))
    sns.heatmap(sales_pivot, cmap='YlGnBu')
    plt.title('Heatmap of Sales by Weekday and Hour')
    plt.xlabel('Weekday')
    plt.ylabel('Hour of Day')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '13_sales_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Country boxplot
    plt.figure(figsize=(12, 6))
    top_countries = df_clean['Country'].value_counts().head(5).index
    sns.boxplot(x='Country', y='TotalPrice', 
                data=df_clean[df_clean['Country'].isin(top_countries)], 
                showfliers=False, palette='Set3')
    enhance_plot('Order Value Distribution by Country (Top 5)', 'Country', 'Order Value (£)')
    save_plot('14_country_boxplot.png')
    
    # 5. RFM scatter
    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=rfm, x='Recency', y='Monetary', hue='Frequency', 
                   palette='viridis', s=100, alpha=0.6)
    enhance_plot('Customer Value Analysis', 'Recency (days)', 'Monetary Value (£)')
    save_plot('15_rfm_scatter.png')

# Main execution
if __name__ == "__main__":
    try:
        # Step 1: Load and explore data
        df = load_and_explore_data()
        
        # Step 2: Clean data
        df_clean = clean_data(df)
        
        # Step 3: Engineer features
        df_clean, rfm = engineer_features(df_clean)
        
        # Step 4: Visualize data
        visualize_data(df_clean, rfm)
        # Step 4b: Enhanced EDA
        enhanced_eda(df_clean, rfm)
        
        # Step 5: Perform customer segmentation
        rfm = perform_customer_segmentation(rfm)
        
        # Step 6: Analyze association rules
        analyze_association_rules(df_clean)
        
        # Step 7: Perform classification analysis
        classification_results = perform_classification_analysis(df_clean, rfm)
        
        # Step 8: Generate business insights
        insights = generate_business_insights(df_clean, rfm, classification_results)
        
        # Step 9: Perform time series analysis
        perform_time_series_analysis(df_clean)
        
        log_print("\nAnalysis completed successfully!")
        log_print(f"Plots saved in: {output_dir}")
        log_print(f"Log file saved in: {log_filename}")
        
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}", exc_info=True)
        raise

