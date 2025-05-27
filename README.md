# Retail Customer Data Mining, Analytics & Segmentation Project

## Overview
This project performs comprehensive customer behavior analysis, product sales insights, and segmentation using advanced data science techniques. The dataset used is the Online Retail II dataset, which contains transactional data for a UK-based online retailer.

The goal is to clean, explore, and analyze the data to extract meaningful insights that can help the business in customer relationship management, targeted marketing, and strategic planning.

## Project Structure
```
retail/
├── retail_analysis_plots/       # Generated visualizations and plots
├── retail_analysis_logs/        # Analysis logs with detailed process info
├── retail_dashboard_streamlit/  # Streamlit dashboard
│   ├── app.py                   # Dashboard application
│   ├── plots/                   # Dashboard visualizations
│   └── logs/                    # Dashboard logs
├── venv/                        # Python virtual environment
├── retailed.py                  # Main analysis script
├── online_retail_II.csv         # Dataset (large file ~90MB)
├── Retail.ipynb                 # Jupyter notebook version
├── Report.docx                  # Detailed analysis report
└── requirements.txt             # Python dependencies
```

## Key Features and Analysis Techniques
- **Data Cleaning & Preprocessing**: Handles missing values, cancellations, duplicates
- **RFM Analysis**: Segments customers by Recency, Frequency, and Monetary value
- **Customer Segmentation**: Applies K-Means, Hierarchical, and DBSCAN clustering
- **Market Basket Analysis**: Discovers product associations using Apriori and FP-Growth
- **Time Series Forecasting**: Predicts future sales with ARIMA and Prophet
- **Classification Modeling**: Tests Naive Bayes, KNN, and SVM for customer classification
- **Interactive Dashboard**: Visualizes findings with Streamlit

## How to Run the Project

### Prerequisites
- Python 3.8+
- 4GB+ RAM (dataset is large)
- Disk space: ~100MB for code, ~100MB for virtual environment

### Setting Up the Environment
1. **Activate the virtual environment**:

   ```powershell
   # In PowerShell
   .\venv\Scripts\Activate.ps1
   ```

   ```cmd
   # In Command Prompt
   .\venv\Scripts\activate.bat
   ```

2. **If you need to recreate the environment**:

   ```bash
   # Create a new virtual environment
   python -m venv venv
   
   # Activate it (see above)
   
   # Install dependencies
   pip install -r requirements.txt
   ```

### Running the Analysis

1. **Full Data Analysis Pipeline**:
   ```bash
   python retailed.py
   ```
   This will generate all plots in the `retail_analysis_plots` directory and logs in `retail_analysis_logs`.

2. **Running the Dashboard**:
   ```bash
   cd retail_dashboard_streamlit
   streamlit run app.py
   ```
   The dashboard will open in your default web browser at http://localhost:8501.

## Dataset Description

The Online Retail II dataset contains all transactions for a UK-based online retailer between 01/12/2009 and 09/12/2011. The dataset includes:

| Column Name | Description |
|-------------|-------------|
| Invoice     | Invoice number (6-digit integer, "C" prefix for cancellations) |
| StockCode   | Product code (5-digit integer or alphanumeric) |
| Description | Product name |
| Quantity    | Quantity of items per transaction |
| InvoiceDate | Invoice date and time (dd/mm/yyyy hh:mm) |
| Price       | Product price per unit in GBP (£) |
| Customer ID | Customer number (5-digit integer) |
| Country     | Country name |

Dataset source: https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci

## Key Findings

1. **Customer Segments**:
   - High-value loyal customers (15% of base) generate 55% of revenue
   - At-risk customers (recently inactive) need targeted re-engagement 
   - Price-sensitive occasional buyers respond well to promotions

2. **Sales Patterns**:
   - Strong seasonal peaks in November-December
   - Weekday activity significantly higher than weekends
   - 10-11 AM shows peak transaction volume

3. **Product Associations**:
   - Strong bundling opportunities for decorative items
   - Certain product sets frequently purchased together
   - Complementary product recommendations can increase basket size

4. **Geographic Insights**:
   - UK, Germany, France, and Ireland are primary markets
   - Each country shows distinct purchasing patterns

## Visualization Examples

The `retail_analysis_plots` directory contains over 20 visualizations including:
- Customer segmentation scatter plots
- RFM distribution histograms
- Sales trends and forecasts
- Product association networks
- Geographical sales distribution
- Classification performance metrics

## Advanced Techniques Used

1. **DBSCAN Clustering**: Density-based clustering that handles noise and finds arbitrary-shaped clusters
2. **Principal Component Analysis**: Dimensionality reduction for better segment visualization
3. **Association Rule Mining**: Apriori and FP-Growth algorithms for product recommendations
4. **Prophet Forecasting**: Advanced time series model incorporating seasonality and holiday effects
5. **Confusion Matrices & ROC Curves**: Detailed classification performance analysis

## Extending the Project

Here are some ideas for extending the project:
- Implement live data updates from sales API
- Add Natural Language Processing for product description analysis
- Create a recommendation system based on association rules
- Develop a churn prediction model for at-risk customers
- Build a customer lifetime value (CLV) predictor

## Troubleshooting

If you encounter any issues:

1. **Package installation problems**:
   ```bash
   # Try forcing reinstall of problematic package
   pip install --force-reinstall package_name
   ```

2. **Memory issues with large dataset**:
   - Try reducing the dataset size with sampling
   - Increase system swap space
   - Process data in batches

3. **Matplotlib errors**:
   - Make sure you're running scripts within the activated virtual environment
   - Check log files for specific error messages

## Dependencies

The project relies on these primary packages:
- streamlit (1.32.0) - Interactive dashboard
- pandas (2.2.1) & numpy (1.26.4) - Data manipulation
- matplotlib (3.8.3) & seaborn (0.13.2) - Visualization
- scikit-learn (1.3.2) - Machine learning algorithms
- mlxtend (0.23.1) - Association rule mining
- prophet (1.1.5) & statsmodels (0.14.1) - Time series forecasting
- networkx (3.2.1) - Network visualization
- plotly (5.19.0) - Interactive charts

## License & Attribution

This project was developed as an academic exercise. The Online Retail II dataset is publicly available from the UCI Machine Learning Repository.

## Contributors

Created as a data mining project with a focus on customer segmentation and retail analytics.
