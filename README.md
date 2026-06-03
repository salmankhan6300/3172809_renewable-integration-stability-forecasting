# Distributed Renewable Integration Stability Forecasting

Predicting grid instability risk using NASA POWER hourly meteorological data for Northern Ireland (2015-2025).

## Overview

This project implements a distributed machine learning pipeline to forecast renewable energy integration stability. It uses 10 years of hourly weather data (~89,000 records, 15 variables) from the NASA POWER dataset to:

- Estimate **Wind Power Index (WPI)** and **Solar Irradiance Index (SII)**
- Compute a custom **Renewable Volatility Stress Index (RVSI)**
- Predict grid **instability risk** (Low / Medium / High / Critical)
- Provide an interactive **web dashboard** for visualization

## Architecture

```
NASA POWER CSV (10 years, hourly, 15 vars)
       |
       v
Data Ingestion (PySpark) -- Download, Load, HDFS-Sim Partition
       |
       v
Data Preprocessing (PySpark) -- Sentinel removal, Datetime, Validation
       |
       v
Exploratory Data Analysis -- 12 saved visualizations
       |
       v
Feature Engineering (PySpark) -- WPI, SII, Temporal, Interaction, Lags, RVSI
       |
       v
ML Modeling (PySpark MLlib) -- RF, GBT, Logistic Regression
       |
       v
Deep Learning (Keras) -- LSTM Regression, DNN Classification
       |
       v
Time Series (SARIMA) -- Daily RVSI forecasting
       |
       v
Flask REST API  +  Streamlit Dashboard
```

## Tech Stack

| Component | Technology |
|---|---|
| Distributed Processing | Apache Spark (PySpark), Spark SQL |
| Storage | HDFS simulation (partitioned Parquet) |
| ML Models (Spark) | PySpark MLlib (Random Forest, GBT, Logistic Regression) |
| Deep Learning | TensorFlow / Keras (LSTM, DNN) |
| Time Series | statsmodels (SARIMA) |
| Web API | Flask |
| Dashboard | Streamlit + Plotly |
| Testing | pytest |
| Visualization | Matplotlib, Seaborn, Plotly |

## Project Structure

```
Project/
|-- config/spark_config.py       # Spark session and project constants
|-- src/
|   |-- data_ingestion.py        # Download and load data
|   |-- data_preprocessing.py    # Clean / transform
|   |-- eda.py                   # Exploratory data analysis (12 plots)
|   |-- feature_engineering.py   # WPI, SII, temporal, interaction, RVSI
|   |-- modeling.py              # PySpark MLlib models (RF, GBT, LR)
|   |-- deep_learning.py         # Keras LSTM + DNN
|   |-- time_series.py           # SARIMA forecasting
|   |-- utils.py                 # Shared helpers
|-- notebooks/main.ipynb         # Full end-to-end Colab notebook
|-- api/app.py                   # Flask REST API
|-- dashboard/app.py             # Streamlit dashboard
|-- tests/                       # pytest test suite
|-- data/
|   |-- raw/                     # Raw CSV data
|   |-- clean/                   # Cleaned + feature Parquet data
|   |-- plots/                   # EDA and model plots (PNG)
|-- hdfs_sim/                    # Simulated HDFS storage
|-- models/                      # Saved ML / DL models
|-- PROJECT_REPORT.md            # Full project documentation
|-- requirements.txt             # Python dependencies
|-- .gitignore
```

## Environment Setup

### Prerequisites

- Python 3.10+
- Java 11+ (for PySpark)
- pip

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd day1

# Install Python dependencies
pip install -r requirements.txt

# Set JAVA_HOME (Linux/WSL example)
export JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
# Unset SPARK_HOME if using pip-installed PySpark
unset SPARK_HOME
```

### WSL-Specific Setup

If running on Windows with WSL:

```bash
# Install Java
sudo apt update && sudo apt install -y openjdk-21-jdk

# Install Python packages (system-wide in WSL)
pip3 install --break-system-packages -r requirements.txt
```

## Running the Pipeline

### Step-by-Step Commands

```bash
# 1. Data Ingestion -- download NASA POWER data and partition to HDFS-sim
python3 -m src.data_ingestion

# 2. Data Preprocessing -- handle sentinels, create datetime, validate schema
python3 -m src.data_preprocessing

# 3. Exploratory Data Analysis -- generate 12 plots to data/plots/
python3 -m src.eda

# 4. Feature Engineering -- compute WPI, SII, temporal, interaction, RVSI
python3 -m src.feature_engineering

# 5. ML Modeling (PySpark MLlib) -- train RF, GBT, LR and evaluate
python3 -m src.modeling

# 6. Time Series Analysis -- SARIMA forecasting on daily RVSI
python3 -m src.time_series

# 7. Deep Learning -- LSTM regression + DNN classification
python3 -m src.deep_learning
```

### Running Tests

```bash
# Run all tests
python3 -m pytest tests/ -v

# Run specific test modules
python3 -m pytest tests/test_preprocessing.py -v
python3 -m pytest tests/test_features.py -v
python3 -m pytest tests/test_modeling.py -v
python3 -m pytest tests/test_api.py -v
```

### Launch Dashboard

```bash
streamlit run dashboard/app.py
```

### Launch API

```bash
python3 -m api.app
```

### Run via Notebook (Google Colab)

Upload `notebooks/main.ipynb` to Google Colab. The notebook auto-installs all dependencies and runs the full pipeline end-to-end.

## Dataset

- **Source**: [NASA POWER 10-Year Hourly Datasets](https://www.kaggle.com/datasets/mdfahimbinamin/nasa-power-10-years-hourly-datasets-4-countries)
- **Region**: Northern Ireland
- **Period**: 2015-2025 (hourly resolution)
- **Records**: ~89,376 rows x 15 columns
- **Missing values**: Encoded as -999

## Key Metrics

| Metric | Description |
|---|---|
| WPI | Wind Power Index: 0.5 x rho x v^3 |
| SII | Solar Irradiance Index: adjusted for cloud cover and temperature efficiency |
| RVSI | Renewable Volatility Stress Index: combined wind and solar volatility |
| Risk Classes | Low, Medium, High, Critical (quantile-based thresholds) |

## EDA Plots Generated

All 12 plots are saved to `data/plots/`:

| # | Plot | File |
|---|---|---|
| 1 | Descriptive Statistics Table | 01_descriptive_statistics.png |
| 2 | Variable Distributions (Histogram + KDE) | 02_distributions.png |
| 3 | Pearson Correlation Heatmap | 03_correlation_heatmap.png |
| 4 | 10-Year Time Series Trends | 04_time_series_trends.png |
| 5 | Diurnal (Hourly) Patterns | 05_hourly_patterns.png |
| 6 | Seasonal Box Plots | 06_seasonal_boxplots.png |
| 7 | Wind Direction Distribution | 07_wind_direction.png |
| 8 | Missing Data (Sentinel) Heatmap | 08_missing_data_heatmap.png |
| 9 | STL Seasonal Decomposition | 09_seasonal_decomposition.png |
| 10 | ACF / PACF Analysis | 10_acf_pacf.png |
| 11 | RVSI Distribution + Risk Pie Chart | 11_rvsi_analysis.png |
| 12 | WPI vs SII Scatter by Risk Class | 12_wpi_vs_sii_scatter.png |

## References

Vijay Babu, A.R., Bharath Kumar, N., Narasipuram, R.P. and Periyannan, S., 2025. Solar Energy Forecasting using Machine Learning Techniques for Enhanced Grid Stability. IEEE Access, 13. doi:10.1109/ACCESS.2025.3574093.
