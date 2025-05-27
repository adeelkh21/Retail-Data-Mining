import streamlit as st
import os
import pandas as pd
import re
import matplotlib.pyplot as plt

# Helper to load log sections
def extract_log_section(log_path, start_marker, end_marker=None):
    with open(log_path, "r", encoding="utf-8") as f:
        log = f.read()
    start = log.find(start_marker)
    if start == -1:
        return "Section not found in log."
    if end_marker:
        end = log.find(end_marker, start)
        return log[start:end] if end != -1 else log[start:]
    return log[start:]

# Plot mapping and interpretations
PLOTS = {
    "overview": [
        ("01_top_countries.png", "Top 10 Countries by Transactions", "This bar chart shows the countries with the most transactions, highlighting the primary markets for the business."),
        ("02_top_products.png", "Top 10 Selling Products", "This bar chart displays the most popular products, helping identify bestsellers and inventory priorities."),
    ],
    "preprocessing": [],
    "eda": [
        ("03_monthly_sales.png", "Monthly Sales Trend", "This line plot reveals seasonality and growth patterns in monthly sales."),
        ("04_rfm_distributions.png", "RFM Distributions", "Histograms for Recency, Frequency, and Monetary value, useful for customer segmentation."),
        ("TransactionByhourofDay.png", "Transactions by Hour of Day", "This plot shows the distribution of transactions across different hours of the day, highlighting peak shopping times."),
        ("11_top_customers.png", "Top 10 Customers by Spend", "Bar chart of the highest-spending customers, useful for loyalty programs."),
        ("13_sales_heatmap.png", "Sales Heatmap by Weekday and Hour", "Heatmap showing peak sales periods by day and hour."),
        ("14_country_boxplot.png", "Order Value by Country", "Boxplot comparing order values across top countries."),
        ("15_rfm_scatter.png", "Recency vs. Monetary Value", "Scatter plot showing customer value clusters."),
    ],
    "segmentation": [
        ("05_elbow_curve.png", "K-Means Elbow Curve", "Elbow method to determine optimal clusters for customer segmentation."),
        ("06_dendrogram.png", "Hierarchical Clustering Dendrogram", "Dendrogram for visualizing customer groupings."),
        ("07_customer_segments.png", "Customer Segments (PCA Projection)", "PCA scatter plot of customer clusters."),
    ],
    "classification": [
        ("classification_naive_bayes_confusion_matrix.png", "Naive Bayes Confusion Matrix", "Confusion matrix for Naive Bayes classifier."),
        ("classification_k-nearest_neighbors_confusion_matrix.png", "KNN Confusion Matrix", "Confusion matrix for K-Nearest Neighbors classifier."),
        ("classification_support_vector_machine_confusion_matrix.png", "SVM Confusion Matrix", "Confusion matrix for Support Vector Machine classifier."),
    ],
    "association": [
        ("12_association_rules_network.png", "Association Rules Network", "Network graph showing strong product associations."),
        ("13_top10_association_rules.csv", "Top 10 Association Rules", "CSV table of the top 10 association rules (see below)."),
    ],
    "timeseries": [
        ("08_daily_sales.png", "Daily Sales Trend", "Line plot of daily sales, showing trends and anomalies."),
        ("09_seasonal_decomposition.png", "Seasonal Decomposition", "Decomposition of sales into trend, seasonality, and residuals."),
        ("10_arima_forecast.png", "ARIMA Forecast", "ARIMA model forecast for future sales."),
        ("11_prophet_forecast.png", "Prophet Forecast", "Prophet model forecast for future sales."),
    ],
}

# Sidebar
st.set_page_config(page_title="Retail Analysis Dashboard", layout="wide")
st.markdown(
    """
    <style>
    .sidebar .sidebar-content {background: #22223b;}
    .css-1d391kg {padding: 2rem 1rem;}
    .stButton>button {background-color: #4A4E69; color: white;}
    .stButton>button:hover {background-color: #22223b;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.title("🛍️ Retail Analysis")
section = st.sidebar.radio(
    "Select Analysis Step",
    [
        "Project Overview",
        "Data Overview",
        "Preprocessing",
        "EDA",
        "Segmentation",
        "Classification",
        "Association Rules",
        "Time Series",
        "Business Insights",
        "Logs"
    ],
)

# Project Summary
st.sidebar.markdown("---")
st.sidebar.markdown("""
### 📘 Project Summary
This data mining project was conducted as a semester-long academic endeavor by Muhammad Adeel (Reg. No. 2022331) and Nauman Ali Murad (Reg. No. 2022479) for the course Data Mining. The objective was to extract actionable business insights from a large e-commerce transactional dataset comprising over 1 million records.""")

# Main content
st.title("Retail Analysis Dashboard")
st.caption("A data scientist's summary of your retail data, using pre-generated analysis and visualizations.")

plots_dir = "plots"
logs_dir = "logs"
log_file = os.path.join(logs_dir, "latest.log")
if not os.path.exists(log_file):
    # fallback: use any .log file
    logs = [f for f in os.listdir(logs_dir) if f.endswith(".log")]
    log_file = os.path.join(logs_dir, logs[0]) if logs else None

def show_plots(plot_list):
    for fname, title, interp in plot_list:
        st.subheader(title)
        if fname.endswith(".png"):
            st.image(os.path.join(plots_dir, fname), use_container_width=True)
        elif fname.endswith(".csv"):
            st.write("Top 10 Association Rules (CSV):")
            st.dataframe(pd.read_csv(os.path.join(plots_dir, fname)))
        st.caption(f"**Interpretation:** {interp}")

def extract_df_from_log(log, start_marker, end_marker=None):
    """Extracts a DataFrame from a log section."""
    start = log.find(start_marker)
    if start == -1:
        return None
    if end_marker:
        end = log.find(end_marker, start)
        section = log[start:end] if end != -1 else log[start:]
    else:
        section = log[start:]
    # Find the DataFrame block (pandas prints with ... and [n rows x m columns])
    df_block = re.findall(r'((?:.*\\n)+?\\[\\d+ rows x \\d+ columns\\])', section)
    if df_block:
        # Try to read as CSV
        from io import StringIO
        try:
            # Remove the "[n rows x m columns]" line
            csv_text = re.sub(r'\\[\\d+ rows x \\d+ columns\\]', '', df_block[0])
            df = pd.read_csv(StringIO(csv_text), delim_whitespace=True)
            return df
        except Exception:
            return None
    return None

def extract_missing_values(log):
    # Extract the missing values section
    match = re.search(r"Missing values per column:[\\s\\S]*?Country\\s+\\d+", log)
    if match:
        lines = match.group(0).splitlines()[1:]
        data = [line.split() for line in lines if line.strip()]
        df = pd.DataFrame(data, columns=["Column", "Missing"])
        return df
    return None

def extract_top_countries(log):
    # Extract the top countries section
    match = re.search(r"Top countries by transaction count:[\\s\\S]*?Name: count, dtype: int64", log)
    if match:
        lines = match.group(0).splitlines()[2:-1]
        data = [line.split() for line in lines if line.strip()]
        df = pd.DataFrame(data, columns=["Country", "Count"])
        return df
    return None

if section == "Project Overview":
    st.markdown("""
    # 🛍️ Retail Customer Analytics & Segmentation Project

    ## 📌 Overview
    This project aimed to perform comprehensive customer behavior analysis, product sales insights, and segmentation using advanced data science techniques. The dataset used was the Online Retail II dataset, which contains transactional data for a UK-based online retailer.

    The goal was to clean, explore, and analyze the data to extract meaningful insights that can help the business in customer relationship management, targeted marketing, and strategic planning.

    ## ✅ Objectives
    - Clean and preprocess retail transaction data.
    - Perform Exploratory Data Analysis (EDA) and visualize key trends.
    - Conduct RFM analysis (Recency, Frequency, Monetary).
    - Apply clustering techniques (K-Means, Hierarchical, DBSCAN) to segment customers.
    - Visualize clusters using PCA.
    - Discover association rules for product recommendations.
    - Forecast future sales using Prophet and ARIMA.

    ## 🧰 Tools & Technologies Used
    - **Languages/Libraries**: Python, Pandas, NumPy, Scikit-learn, Seaborn, Matplotlib, Prophet, Statsmodels, Mlxtend, NetworkX
    - **Clustering Algorithms**: K-Means, Agglomerative Clustering, DBSCAN
    - **Forecasting Models**: ARIMA, Facebook Prophet
    - **Visualization Tools**: Matplotlib, Seaborn
    - **Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC

    ## 🔍 Step-by-Step Workflow
    ### 📥 1. Data Import and Logging Setup
    All necessary Python libraries were imported and customized logging functions were set up to track the flow and errors during runtime. Output directories for visualizations and logs were created dynamically.

    ### 📊 2. Data Loading & Exploration
    - Loaded a large CSV dataset (online_retail_II.csv).
    - Displayed key statistics and missing value counts.
    - Identified top countries by transaction volume and key product details.

    ### 🧹 3. Data Cleaning
    - Removed cancelled invoices (marked with 'C').
    - Dropped rows with missing Customer ID and Description.
    - Removed duplicates.
    - Parsed InvoiceDate into Python datetime format.

    ### 🧠 4. Feature Engineering
    - Created a new feature: TotalPrice = Quantity × Price.
    - Extracted temporal features (Year, Month, Day, Weekday, Hour).
    - Built an RFM Table for each customer:
      - Recency: Days since last purchase
      - Frequency: Number of invoices
      - Monetary: Total money spent

    ### 📈 5. Exploratory Data Visualization
    Visualizations were generated and saved to highlight:
    - Top 10 countries (excluding UK)
    - Top-selling products
    - Monthly sales trends
    - Distributions of RFM features

    ### 👥 6. Customer Segmentation
    Applied and compared multiple clustering algorithms:
    - **K-Means Clustering**
      - Optimal number of clusters determined using the Elbow Method
      - 3 clusters chosen and visualized with PCA
    - **Hierarchical Clustering**
      - Used dendrogram to visualize cluster linkage
    - **DBSCAN**
      - Identified noise points and density-based clusters
    - **PCA Projection**
      - Reduced RFM features to 2D for easy visualization

    ### 🔍 Key Observations:
    - RFM clustering showed clearly distinct customer groups.
    - Some customers had high frequency and monetary value, ideal for loyalty programs.
    - Seasonal sales peaks were identified around December, indicating holiday shopping patterns.

    ### 📦 Bonus: Association Rule Mining
    Used Apriori and FP-Growth algorithms to generate product association rules. These can be directly used to power:
    - Product bundling
    - "Customers who bought this also bought…" recommendations
    - Visualized frequent itemsets and created a network graph for interpretability.

    ### 📉 Forecasting Future Sales
    Time-series models were used to forecast future sales trends:
    - ARIMA for statistical forecasting based on lag patterns
    - Prophet (by Meta) for capturing seasonality, trend, and holidays
    - Resulting forecasts showed expected dips and rises with weekly seasonality

    ## 📌 Conclusion
    This project successfully demonstrated the power of data-driven retail analytics. From cleaning raw data to segmenting customers and forecasting trends, this end-to-end pipeline provides actionable insights for businesses looking to improve customer retention and sales.

    ## 🌟 Future Enhancements
    - Deploy this solution on a dashboard using Streamlit or Power BI
    - Integrate real-time transaction feeds
    - Apply deep learning for customer lifetime value (CLV) prediction
    - Perform A/B testing for marketing campaigns based on cluster behavior
    """)

elif section == "Data Overview":
    # Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Rows", "1,067,371")
    with col2:
        st.metric("Columns", "8")
    with col3:
        st.metric("Unique Countries", "43")

    # First Few Rows (hardcoded for clarity)
    st.markdown("### First Few Rows")
    df_preview = pd.DataFrame([
        ["489434", "85048", "15CM CHRISTMAS GLASS BALL 20 LIGHTS", 12, "2009-12-01 07:45:00", 6.95, 13085.0, "United Kingdom"],
        ["489434", "79323P", "PINK CHERRY LIGHTS", 12, "2009-12-01 07:45:00", 6.75, 13085.0, "United Kingdom"],
        ["489434", "79323W", "WHITE CHERRY LIGHTS", 12, "2009-12-01 07:45:00", 6.75, 13085.0, "United Kingdom"],
        ["489434", "22041", "RECORD FRAME 7\" SINGLE SIZE", 48, "2009-12-01 07:45:00", 2.10, 13085.0, "United Kingdom"],
        ["489434", "21232", "STRAWBERRY CERAMIC TRINKET BOX", 24, "2009-12-01 07:45:00", 1.25, 13085.0, "United Kingdom"],
    ], columns=["Invoice", "StockCode", "Description", "Quantity", "InvoiceDate", "Price", "Customer ID", "Country"])
    st.dataframe(df_preview, use_container_width=True)

    # Data Types and Non-Nulls (from pandas info)
    st.markdown("### Data Types & Non-Null Counts")
    df_info = pd.DataFrame({
        "Column": ["Invoice", "StockCode", "Description", "Quantity", "InvoiceDate", "Price", "Customer ID", "Country"],
        "Non-Null Count": [1067371, 1067371, 1062989, 1067371, 1067371, 1067371, 824364, 1067371],
        "Dtype": ["object", "object", "object", "int64", "object", "float64", "float64", "object"]
    })
    st.dataframe(df_info, use_container_width=True)

    # Data Quality & Country Distribution
    st.markdown("### Data Quality & Country Distribution")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Missing Values per Column")
        df_missing = pd.DataFrame({
            "Column": ["Invoice", "StockCode", "Description", "Quantity", "InvoiceDate", "Price", "Customer ID", "Country"],
            "Missing": [0, 0, 4382, 0, 0, 0, 243007, 0]
        })
        st.dataframe(df_missing, use_container_width=True)
    with col2:
        st.markdown("#### Top Countries by Transaction Count")
        df_countries = pd.DataFrame({
            "Country": ["United Kingdom", "EIRE", "Germany", "France", "Netherlands"],
            "Count": [981330, 17866, 17624, 14330, 5140]
        })
        st.dataframe(df_countries, use_container_width=True)

    # Now show the visualizations
    show_plots(PLOTS["overview"])

elif section == "Preprocessing":
    st.markdown("## Cleaned Data Overview")
    # Data types and non-null counts after cleaning
    df_clean_info = pd.DataFrame({
        "Column": ["Invoice", "StockCode", "Description", "Quantity", "InvoiceDate", "Price", "Customer ID", "Country"],
        "Non-Null Count": [779495]*8,
        "Dtype": ["object", "object", "object", "int64", "datetime64[ns]", "float64", "float64", "object"]
    })
    st.dataframe(df_clean_info, use_container_width=True)

    st.markdown("## First Few Rows of Cleaned Data")
    df_clean_preview = pd.DataFrame([
        ["489434", "85048", "15CM CHRISTMAS GLASS BALL 20 LIGHTS", 12, "2009-12-01 07:45:00", 6.95, 13085.0, "United Kingdom"],
        ["489434", "79323P", "PINK CHERRY LIGHTS", 12, "2009-12-01 07:45:00", 6.75, 13085.0, "United Kingdom"],
        ["489434", "79323W", "WHITE CHERRY LIGHTS", 12, "2009-12-01 07:45:00", 6.75, 13085.0, "United Kingdom"],
        ["489434", "22041", "RECORD FRAME 7\" SINGLE SIZE", 48, "2009-12-01 07:45:00", 2.10, 13085.0, "United Kingdom"],
        ["489434", "21232", "STRAWBERRY CERAMIC TRINKET BOX", 24, "2009-12-01 07:45:00", 1.25, 13085.0, "United Kingdom"],
    ], columns=["Invoice", "StockCode", "Description", "Quantity", "InvoiceDate", "Price", "Customer ID", "Country"])
    st.dataframe(df_clean_preview, use_container_width=True)

    st.markdown("## First Few Rows of Engineered Features")
    df_features_preview = pd.DataFrame([
        [83.4, 2009, 12, 1, 7, 1],
        [81.0, 2009, 12, 1, 7, 1],
        [81.0, 2009, 12, 1, 7, 1],
        [100.8, 2009, 12, 1, 7, 1],
        [30.0, 2009, 12, 1, 7, 1],
    ], columns=["TotalPrice", "Year", "Month", "Day", "Hour", "Weekday"])
    st.dataframe(df_features_preview, use_container_width=True)

    st.markdown("## RFM Sample")
    df_rfm = pd.DataFrame([
        [12346.0, 326, 12, 77556.46],
        [12347.0, 2, 8, 4921.53],
        [12348.0, 75, 5, 2019.40],
        [12349.0, 19, 4, 4428.69],
        [12350.0, 310, 1, 334.40],
    ], columns=["CustomerID", "Recency", "Frequency", "Monetary"])
    st.dataframe(df_rfm, use_container_width=True)

    # RFM Distributions Plot
    st.markdown("## RFM Distributions")
    st.image(os.path.join(plots_dir, "04_rfm_distributions.png"), use_container_width=True)
    st.caption("Histograms for Recency, Frequency, and Monetary value, useful for customer segmentation.")

elif section == "EDA":
    show_plots(PLOTS["eda"])

elif section == "Segmentation":
    show_plots(PLOTS["segmentation"])
    # Pretty cluster summary table
    st.markdown("## Cluster Summary")
    df_cluster_summary = pd.DataFrame({
        "Cluster": ["0", "1", "2"],
        "Recency": [66.33, 462.16, 23.09],
        "Frequency": [7.64, 2.20, 143.14],
        "Monetary": [3134.79, 746.41, 173123.58],
        "Count": [3849, 2010, 22]
    })
    st.dataframe(df_cluster_summary, use_container_width=True)
    st.caption(
        "Cluster 0: High-Value Loyal (low recency, high frequency, high monetary, large group)  \n"
        "Cluster 1: Low-Spend Recent (high recency, low frequency, low monetary, medium group)  \n"
        "Cluster 2: Churned (very low recency, very high frequency, very high monetary, very small group)"
    )

elif section == "Classification":
    # Add summary plot and table for classification metrics
    st.markdown("## Classification Metrics Summary")
    # Hardcoded summary table
    df_metrics = pd.DataFrame({
        "Model": ["Naive Bayes", "KNN", "SVM"],
        "Accuracy": [0.97, 0.98, 0.97],
        "Precision (0)": [0.97, 0.98, 0.97],
        "Recall (0)": [0.99, 0.99, 1.00],
        "F1-score (0)": [0.98, 0.98, 0.98],
        "Precision (1)": [0.96, 0.97, 0.99],
        "Recall (1)": [0.92, 0.93, 0.91],
        "F1-score (1)": [0.94, 0.95, 0.95],
    })
    st.dataframe(df_metrics, use_container_width=True)
    st.image(os.path.join(plots_dir, "classification_metrics_summary.png"), use_container_width=True)
    st.caption(
        "This summary table and bar plot compare the main classification metrics (accuracy, precision, recall, F1-score) for each model. All models perform very well, with KNN and SVM showing slightly higher precision and recall for the minority class (1). Naive Bayes is also strong, but SVM achieves the highest precision for class 1. Overall, all models are suitable, but KNN and SVM may be preferred for slightly better minority class performance."
    )
    
    # Custom layout for confusion matrix plots
    st.markdown("## Confusion Matrices")
    cols = st.columns(3)
    for i, (fname, title, interp) in enumerate(PLOTS["classification"]):
        with cols[i]:
            st.image(os.path.join(plots_dir, fname), width=400, use_container_width=False)
            st.caption(f"**{title}**\n{interp}")

elif section == "Association Rules":
    show_plots(PLOTS["association"])
    # Add the new directed graph at the bottom
    st.markdown("## Top 5 Association Rules (France) - Directed Graph")
    st.image(os.path.join(plots_dir, "top5associationrulesFracnce.png"), use_container_width=True)
    st.caption("Directed network graph of the top 5 association rules for France.")

elif section == "Time Series":
    show_plots(PLOTS["timeseries"])

elif section == "Business Insights":
    st.markdown("""
    ### 📊 **Business Insights & Strategic Recommendations**

    *Derived from Transactional Analysis of ~1.07M Rows (Cleaned to ~779K)*

    ---

    #### 1️⃣ **Customer Segmentation via RFM Analysis**

    * **Cluster 0: High-Value, Frequent Buyers**

      * **Recency**: 66 days
      * **Frequency**: 7.6 purchases
      * **Monetary**: £3,134 avg. spend
      * ✅ **Action**: Launch loyalty programs, exclusive early-bird sales, and bundled product deals to retain them.

    * **Cluster 1: Dormant, Low-Value Customers**

      * **Recency**: 462 days
      * **Frequency**: 2.2 purchases
      * **Monetary**: £746 avg. spend
      * ⚠️ **Action**: Run re-engagement campaigns (e.g., discount emails, feedback forms) and analyze reasons for churn.

    * **Cluster 2: VIP Customers**

      * **Recency**: 23 days
      * **Frequency**: 143 purchases
      * **Monetary**: £173,123 avg. spend
      * 💎 **Action**: Provide dedicated account managers, premium services, and early access to new collections.

    ---

    #### 2️⃣ **Geographic Concentration**

    * **Top 5 Countries by Transactions:**

      1. **United Kingdom** 🇬🇧 – 981,330 transactions
      2. **Ireland (EIRE)** 🇮🇪 – 17,866
      3. **Germany** 🇩🇪 – 17,624
      4. **France** 🇫🇷 – 14,330
      5. **Netherlands** 🇳🇱 – 5,140

    ✅ **Action**: Focus marketing and logistics infrastructure in the UK, while growing targeted promotions and partnerships in top EU countries.

    ---

    #### 3️⃣ **Product Portfolio Insights**

    * Popular product lines include:

      * 🎨 **Mini Cases & Polkadot Designs**
      * ⏰ **Bakelike Alarm Clocks (Various Colors)**
      * 🍽️ **Children's Breakfast & Cutlery Sets**
      * 🍰 **Cake Stands & Baking Sets**
      * 🎒 **Charlotte Bags & Lunch Boxes**

    📦 **Action**: Ensure these items are always in stock. Consider upselling related items through product bundling.

    ---

    #### 4️⃣ **Data Quality Improvements**

    * **Customer ID Missing Rate**: ~22.7%
      ✅ **Action**: Enforce login or sign-up before purchase to collect complete customer profiles for better segmentation and personalization.

    * **Description Nulls**: 4,382 rows
      ✅ **Action**: Clean product catalog to ensure every item has clear, marketable descriptions.

    ---

    #### 5️⃣ **Anomaly Detection (DBSCAN)**

    * **30 Outliers Detected**
      🧠 **Action**: Review these for potential fraud, bulk corporate orders, or pricing errors.

    ---

    #### 6️⃣ **Operational Opportunities**

    * Implement **dynamic pricing models** for frequent buyers.
    * Launch **seasonal campaigns** centered around top-selling categories (e.g., back-to-school for kids' products).
    * Set up **country-specific micro-sites** or delivery offers for Germany, France, and the Netherlands to boost non-UK sales.

    ---
    """)

elif section == "Logs":
    if log_file:
        st.subheader("Full Log Output")
        with open(log_file, "r", encoding="utf-8") as f:
            st.code(f.read())
