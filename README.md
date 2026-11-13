# cloud-based-ml-sales-forecasting

Cloud-Based Retail Sales Forecasting Using Azure Machine Learning, LightGBM, and Automated ML Pipelines

This project implements an end-to-end cloud-based machine learning workflow for 7-day retail sales forecasting using Microsoft Azure Machine Learning and the FreshRetailNet-50K dataset.

The solution uses Azure ML compute resources to execute data preparation, feature engineering, and LightGBM model training in the cloud. The trained model is registered in the Azure ML Model Registry and subsequently used for cloud-based batch inference and evaluation.

Historical retail sales data were processed to develop 23 forecasting features, including recent sales trends, weekly seasonality, stockout history, product-level demand across stores, and store-category sales patterns. Weather, discounts, holidays, and promotional activities were included as external forecasting inputs.

The workflow uses Python, Azure ML SDK v2, managed cloud compute, MLflow experiment tracking, model registration, and batch prediction jobs. Final predictions are evaluated against actual sales, with additional store-level performance analysis and visualization.

### Key Results

- Implemented an **end-to-end cloud-based ML forecasting workflow on Microsoft Azure**.
- Executed **data preparation, feature engineering, and LightGBM training on Azure ML compute**.
- Selected **100 products across 897 stores**, resulting in **25,539 store-product series**.
- Engineered **23 forecasting features** covering sales history, stockouts, store/product characteristics, and external variables.
- Trained a **LightGBM regression model** using **1,608,957 historical forecasting examples**.
- Registered the trained model in **Azure ML Model Registry** for versioned model management.
- Executed a separate **Azure ML batch inference job**, generating **178,773 daily sales predictions**.
- Achieved **32.11% daily WAPE** and **10.23% weekly product-level WAPE** on the held-out evaluation period (June 26 – July 2, 2024).
- Implemented **cloud-based model training, model registration, batch inference, evaluation, and MLflow experiment tracking**.

### Azure ML Workflow

Data Preparation → Feature Engineering → LightGBM Training → Model Registration → Batch Inference → Evaluation

- **Cloud Platform:** Microsoft Azure
- **ML Service:** Azure Machine Learning
- **Compute:** Azure ML CPU Compute Cluster
- **Machine Learning:** LightGBM Regression
- **Experiment Tracking:** MLflow
- **Model Management:** Azure ML Model Registry
- **Inference:** Azure ML Batch Jobs
- **Evaluation:** Daily and Weekly WAPE

### Result Snapshots

[Five-Store Forecast Performance](docs/results/five_store_final_summary.png): Comparison of weekly product-level forecasting accuracy across five selected stores. The figure includes Daily WAPE, Weekly WAPE, priority product counts, and sales coverage for each store.

[Store 300 - Actual vs. Predicted Weekly Sales](docs/results/store_300_top30_products.png): Comparison of actual and predicted 7-day sales totals for the 30 highest-selling products in Store 300, illustrating product-level forecasting performance.

**Keywords:** Python, Azure Machine Learning, Azure ML Pipelines, Batch Inference, Cloud Computing, Cloud-Based Machine Learning, Demand Forecasting, Feature Engineering, LightGBM, Machine Learning, MLflow, Model Registry, Retail Sales Forecasting, Sales Analytics, Time-Series Forecasting.

**Data Source:** FreshRetailNet-50K Retail Sales Dataset, March–July 2024.
