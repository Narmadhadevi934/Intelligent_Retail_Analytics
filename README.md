# Intelligent Retail Demand Forecasting and Product Recommendation

## Project Overview

The Intelligent Retail Analytics System is a Python-based machine learning project designed to analyze retail transaction data, forecast future product demand, discover frequently purchased product combinations, and recommend relevant products to customers.

The project uses data analytics, machine learning, and pattern mining techniques to support better retail business decisions.

## Objectives

- Analyze retail transaction data.
- Clean and preprocess the dataset.
- Identify best-selling products and customer purchasing patterns.
- Forecast future product demand using machine learning.
- Discover frequently purchased product combinations using FP-Growth.
- Recommend products using collaborative filtering.
- Evaluate forecasting and recommendation performance.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Mlxtend

## Machine Learning Algorithms

### Demand Forecasting
- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor

### Frequent Pattern Mining
- FP-Growth Algorithm
- Association Rule Mining

### Product Recommendation
- Popularity-Based Recommendation
- Item-Based Collaborative Filtering
- Cosine Similarity

## Project Structure

```text
Intelligent_Retail_Analytics/
│
├── make_synthetic_retail.py
├── retail_analytics.py
├── README.md
│
└── outputs/
    ├── synthetic_retail_dataset.csv
    ├── cleaned_retail_dataset.csv
    ├── monthly_sales_trend.png
    ├── top_10_products.png
    ├── sales_by_month.png
    ├── top_customers.png
    ├── sales_by_weekday.png
    ├── demand_forecasting_metrics.csv
    ├── demand_test_predictions.csv
    ├── next_4_weeks_demand_forecast.csv
    ├── frequent_itemsets.csv
    ├── association_rules.csv
    ├── customer_product_recommendations.csv
    ├── recommendation_evaluation.csv
    ├── final_project_summary.csv
    └── final_project_report.txt
```

Additional visualization files are generated during execution.

## Dataset

This project uses a synthetically generated retail transaction dataset.

The dataset contains:
- Invoice Number
- Stock Code
- Product Description
- Quantity
- Invoice Date
- Unit Price
- Customer ID
- Country

The synthetic dataset is generated using `make_synthetic_retail.py`.

## Installation

Install the required Python libraries:

```bash
pip install pandas numpy matplotlib scikit-learn mlxtend
```

## How to Run the Project

**Step 1:** Generate the dataset.

```bash
python make_synthetic_retail.py
```

**Step 2:** Run the complete retail analytics system.

```bash
python retail_analytics.py
```

**Step 3:** Check the generated results inside the `outputs` folder.

## Evaluation Metrics

### Demand Forecasting
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R-squared (R²)

### Product Recommendation
- Precision@5
- Recall@5

### Association Rule Mining
- Support
- Confidence
- Lift

## Expected Results

- Retail sales analysis and visualization.
- Identification of top-selling products.
- Future demand predictions for the next four weeks.
- Frequently purchased product combinations.
- Personalized product recommendations.
- Model evaluation reports.

## Conclusion

The Intelligent Retail Analytics System combines data preprocessing, exploratory data analysis, machine learning, frequent pattern mining, and recommendation techniques to provide useful insights into retail transactions.

The system helps retailers understand customer purchasing behavior, estimate future product demand, and improve product recommendations.

## Author

Student Project – Intelligent Retail Analytics

## License

This project is developed for educational and academic purposes.