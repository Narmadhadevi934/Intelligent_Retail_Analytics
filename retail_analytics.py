
import os
import pandas as pd
import numpy as np

# ==========================================
# 1. DATASET CONFIGURATION
# ==========================================

DATA_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "outputs",
    "synthetic_retail_dataset.csv"
)

# ==========================================
# 2. LOAD DATASET
# ==========================================

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}\n"
        "Please run make_synthetic_retail.py first."
    )

df = pd.read_csv(DATA_PATH)

# ==========================================
# 3. DATA TYPE CONVERSION
# ==========================================

df["InvoiceDate"] = pd.to_datetime(
    df["InvoiceDate"],
    errors="coerce"
)

df["Quantity"] = pd.to_numeric(
    df["Quantity"],
    errors="coerce"
)

df["UnitPrice"] = pd.to_numeric(
    df["UnitPrice"],
    errors="coerce"
)

# ==========================================
# 4. DATASET INFORMATION
# ==========================================

print("\n====================================")
print("   INTELLIGENT RETAIL ANALYTICS")
print("====================================")

print("\nDataset loaded successfully!")

print("\nTotal records:", len(df))

print("Total customers:", df["CustomerID"].nunique())

print("Total products:", df["StockCode"].nunique())

print("Total invoices:", df["InvoiceNo"].nunique())

print("\nDataset columns:")
print(df.columns.tolist())

print("\nFirst five records:")
print(df.head())

print("\nDataset information:")
df.info()

print("\nMissing values:")
print(df.isnull().sum())

# ==========================================
# 5. SAVE LOADED DATASET INFORMATION
# ==========================================

os.makedirs(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs"),
    exist_ok=True
)

print("\nDataset loading completed successfully!")


# ==========================================
# 6. DATA PREPROCESSING
# ==========================================

print("\n====================================")
print("       DATA PREPROCESSING")
print("====================================")

# Keep original dataset count
original_count = len(df)

# 1. Remove duplicate records
df = df.drop_duplicates()

# 2. Remove missing important values
df = df.dropna(
    subset=[
        "InvoiceNo",
        "StockCode",
        "InvoiceDate",
        "Quantity",
        "UnitPrice",
        "CustomerID"
    ]
)

# 3. Remove invalid quantities and prices
df = df[
    (df["Quantity"] > 0) &
    (df["UnitPrice"] > 0)
].copy()

# 4. Remove cancellation invoices
df = df[
    ~df["InvoiceNo"].astype(str).str.startswith("C")
].copy()

# 5. Create total sales column
df["TotalSales"] = df["Quantity"] * df["UnitPrice"]

# 6. Extract date features
df["Year"] = df["InvoiceDate"].dt.year
df["Month"] = df["InvoiceDate"].dt.month
df["Day"] = df["InvoiceDate"].dt.day
df["Week"] = df["InvoiceDate"].dt.isocalendar().week.astype(int)
df["DayOfWeek"] = df["InvoiceDate"].dt.dayofweek

# 7. Sort records by date
df = df.sort_values("InvoiceDate").reset_index(drop=True)

# 8. Display preprocessing results
print("\nOriginal records:", original_count)
print("Cleaned records:", len(df))
print("Removed records:", original_count - len(df))

print("\nMissing values after cleaning:")
print(df.isnull().sum())

print("\nTotal sales:", round(df["TotalSales"].sum(), 2))

print("\nCleaned dataset preview:")
print(df.head())

# 9. Save cleaned dataset
OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "outputs"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

cleaned_path = os.path.join(
    OUTPUT_DIR,
    "cleaned_retail_dataset.csv"
)

df.to_csv(cleaned_path, index=False)

print("\nCleaned dataset saved successfully!")
print("Location:", cleaned_path)


# ==========================================
# 7. EXPLORATORY DATA ANALYSIS (EDA)
# ==========================================

import matplotlib.pyplot as plt

print("\n====================================")
print(" EXPLORATORY DATA ANALYSIS")
print("====================================")

# Output directory
OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "outputs"
)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# 1. MONTHLY SALES TREND
monthly_sales = (
    df.groupby(df["InvoiceDate"].dt.to_period("M"))["TotalSales"]
    .sum()
)

plt.figure(figsize=(12, 5))
plt.plot(
    monthly_sales.index.astype(str),
    monthly_sales.values,
    marker="o"
)
plt.title("Monthly Retail Sales Trend")
plt.xlabel("Month")
plt.ylabel("Total Sales")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "monthly_sales_trend.png"))
plt.close()


# 2. TOP 10 BEST-SELLING PRODUCTS
top_products = (
    df.groupby("Description")["Quantity"]
    .sum()
    .nlargest(10)
)

plt.figure(figsize=(10, 6))
top_products.sort_values().plot(kind="barh")
plt.title("Top 10 Best-Selling Products")
plt.xlabel("Quantity Sold")
plt.ylabel("Product")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "top_10_products.png"))
plt.close()


# 3. SALES BY MONTH
monthly_pattern = (
    df.groupby("Month")["TotalSales"]
    .sum()
    .reindex(range(1, 13), fill_value=0)
)

plt.figure(figsize=(10, 5))
plt.bar(monthly_pattern.index, monthly_pattern.values)
plt.title("Total Sales by Month")
plt.xlabel("Month")
plt.ylabel("Total Sales")
plt.xticks(range(1, 13))
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "sales_by_month.png"))
plt.close()


# 4. TOP 10 CUSTOMERS BY SPENDING
top_customers = (
    df.groupby("CustomerID")["TotalSales"]
    .sum()
    .nlargest(10)
)

plt.figure(figsize=(10, 5))
top_customers.plot(kind="bar")
plt.title("Top 10 Customers by Spending")
plt.xlabel("Customer ID")
plt.ylabel("Total Spending")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "top_customers.png"))
plt.close()


# 5. SALES BY WEEKDAY
weekday_sales = (
    df.groupby("DayOfWeek")["TotalSales"]
    .sum()
    .reindex(range(7), fill_value=0)
)

weekday_names = [
    "Monday", "Tuesday", "Wednesday",
    "Thursday", "Friday", "Saturday", "Sunday"
]

plt.figure(figsize=(10, 5))
plt.bar(weekday_names, weekday_sales.values)
plt.title("Sales by Day of Week")
plt.xlabel("Day")
plt.ylabel("Total Sales")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "sales_by_weekday.png"))
plt.close()


# SAVE EDA SUMMARY FILES
monthly_sales.rename("TotalSales").to_csv(
    os.path.join(OUTPUT_DIR, "monthly_sales_summary.csv")
)

top_products.rename("QuantitySold").to_csv(
    os.path.join(OUTPUT_DIR, "top_products_summary.csv")
)

print("\nEDA completed successfully!")
print("Five visualizations generated.")
print("Charts and summaries saved in:", OUTPUT_DIR)


# ==========================================
# 8. MACHINE LEARNING - DEMAND FORECASTING
# ==========================================

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("\n====================================")
print(" DEMAND FORECASTING")
print("====================================")

# 1. SELECT TOP 10 PRODUCTS
top_stock_codes = (
    df.groupby("StockCode")["Quantity"]
    .sum()
    .nlargest(10)
    .index
)

forecast_df = df[df["StockCode"].isin(top_stock_codes)].copy()

# 2. AGGREGATE WEEKLY DEMAND
forecast_df["WeekStart"] = (
    forecast_df["InvoiceDate"].dt.to_period("W").dt.start_time
)

weekly_sales = (
    forecast_df.groupby(["StockCode", "WeekStart"])["Quantity"]
    .sum()
    .reset_index()
)

# 3. FILL MISSING WEEKS WITH ZERO DEMAND
all_weeks = pd.date_range(
    weekly_sales["WeekStart"].min(),
    weekly_sales["WeekStart"].max(),
    freq="W-MON"
)

# Align weekly dates to Monday
weekly_sales["WeekStart"] = (
    weekly_sales["WeekStart"] -
    pd.to_timedelta(weekly_sales["WeekStart"].dt.weekday, unit="D")
)

all_weeks = pd.date_range(
    weekly_sales["WeekStart"].min(),
    weekly_sales["WeekStart"].max(),
    freq="W-MON"
)

products = weekly_sales["StockCode"].unique()

full_index = pd.MultiIndex.from_product(
    [products, all_weeks],
    names=["StockCode", "WeekStart"]
)

weekly_sales = (
    weekly_sales.set_index(["StockCode", "WeekStart"])
    .reindex(full_index, fill_value=0)
    .reset_index()
)

weekly_sales = weekly_sales.sort_values(
    ["StockCode", "WeekStart"]
).reset_index(drop=True)

# 4. CREATE FEATURES
weekly_sales["Lag_1"] = (
    weekly_sales.groupby("StockCode")["Quantity"].shift(1)
)

weekly_sales["Lag_2"] = (
    weekly_sales.groupby("StockCode")["Quantity"].shift(2)
)

weekly_sales["Lag_4"] = (
    weekly_sales.groupby("StockCode")["Quantity"].shift(4)
)

weekly_sales["Rolling_Mean_4"] = (
    weekly_sales.groupby("StockCode")["Quantity"]
    .transform(lambda x: x.shift(1).rolling(4).mean())
)

weekly_sales["Month"] = weekly_sales["WeekStart"].dt.month
weekly_sales["WeekNumber"] = weekly_sales["WeekStart"].dt.isocalendar().week.astype(int)

model_data = weekly_sales.dropna().copy()

# Convert product codes into numerical columns
product_dummies = pd.get_dummies(
    model_data["StockCode"],
    prefix="Product",
    dtype=int
)

X = pd.concat(
    [
        model_data[
            ["Lag_1", "Lag_2", "Lag_4",
             "Rolling_Mean_4", "Month", "WeekNumber"]
        ].reset_index(drop=True),
        product_dummies.reset_index(drop=True)
    ],
    axis=1
)

y = model_data["Quantity"].reset_index(drop=True)


# 5. CHRONOLOGICAL TRAIN-TEST SPLIT

unique_weeks = sorted(model_data["WeekStart"].unique())

if len(unique_weeks) < 10:
    raise ValueError(
        "Not enough weekly data for forecasting. "
        "At least 10 weeks are required."
    )

test_weeks = unique_weeks[-8:]

train_mask = (
    ~model_data["WeekStart"].isin(test_weeks)
).to_numpy()

test_mask = (
    model_data["WeekStart"].isin(test_weeks)
).to_numpy()

# Reset all indexes before splitting
X = X.reset_index(drop=True)
y = y.reset_index(drop=True)

X_train = X.iloc[train_mask].reset_index(drop=True)
X_test = X.iloc[test_mask].reset_index(drop=True)

y_train = y.iloc[train_mask].reset_index(drop=True)
y_test = y.iloc[test_mask].reset_index(drop=True)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))


# 6. TRAIN MACHINE LEARNING MODELS
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=100,
        random_state=42
    )
}

results = []
predictions = {}

for model_name, model in models.items():

    model.fit(X_train, y_train)

    predicted = model.predict(X_test)
    predicted = np.maximum(predicted, 0)

    predictions[model_name] = predicted

    mae = mean_absolute_error(y_test, predicted)
    rmse = np.sqrt(mean_squared_error(y_test, predicted))
    r2 = r2_score(y_test, predicted)

    results.append({
        "Model": model_name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })

# 7. COMPARE MODEL PERFORMANCE
metrics_df = pd.DataFrame(results)
metrics_df = metrics_df.sort_values("MAE")

print("\nMODEL PERFORMANCE:")
print(metrics_df.to_string(index=False))

metrics_df.to_csv(
    os.path.join(OUTPUT_DIR, "demand_forecasting_metrics.csv"),
    index=False
)

# 8. SELECT BEST MODEL
best_model_name = metrics_df.iloc[0]["Model"]
best_model = models[best_model_name]

print("\nBest Model:", best_model_name)

# 9. SAVE TEST PREDICTIONS
test_results = model_data.loc[
    test_mask, ["StockCode", "WeekStart", "Quantity"]
].copy()

test_results["PredictedDemand"] = predictions[best_model_name]

test_results.rename(
    columns={"Quantity": "ActualDemand"},
    inplace=True
)

test_results.to_csv(
    os.path.join(OUTPUT_DIR, "demand_test_predictions.csv"),
    index=False
)

# 10. FORECAST NEXT FOUR WEEKS
future_results = []

for product in products:

    product_history = weekly_sales[
        weekly_sales["StockCode"] == product
    ].sort_values("WeekStart")

    history = product_history["Quantity"].tolist()
    last_date = product_history["WeekStart"].max()

    for step in range(1, 5):

        future_date = last_date + pd.Timedelta(weeks=step)

        feature_values = {
            "Lag_1": history[-1],
            "Lag_2": history[-2],
            "Lag_4": history[-4],
            "Rolling_Mean_4": np.mean(history[-4:]),
            "Month": future_date.month,
            "WeekNumber": int(future_date.isocalendar().week)
        }

        for code in products:
            feature_values[f"Product_{code}"] = int(code == product)

        future_input = pd.DataFrame(
            [feature_values],
            columns=X.columns
        )

        predicted_demand = max(
            0,
            float(best_model.predict(future_input)[0])
        )

        future_results.append({
            "StockCode": product,
            "ForecastWeek": future_date,
            "PredictedDemand": round(predicted_demand, 2)
        })

        history.append(predicted_demand)

future_df = pd.DataFrame(future_results)

future_df.to_csv(
    os.path.join(OUTPUT_DIR, "next_4_weeks_demand_forecast.csv"),
    index=False
)

# 11. PLOT MODEL PERFORMANCE
plt.figure(figsize=(9, 5))
plt.bar(metrics_df["Model"], metrics_df["MAE"])
plt.title("Demand Forecasting - Model Comparison")
plt.xlabel("Machine Learning Model")
plt.ylabel("Mean Absolute Error")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "forecast_model_comparison.png")
)
plt.close()

# 12. PLOT ACTUAL VS PREDICTED DEMAND
plt.figure(figsize=(12, 5))

plt.plot(
    test_results["ActualDemand"].values,
    label="Actual Demand"
)

plt.plot(
    test_results["PredictedDemand"].values,
    label="Predicted Demand"
)

plt.title("Actual vs Predicted Product Demand")
plt.xlabel("Test Observations")
plt.ylabel("Weekly Quantity")
plt.legend()
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "actual_vs_predicted_demand.png")
)
plt.close()

# 13. DISPLAY FUTURE FORECAST
print("\nNEXT FOUR WEEKS DEMAND FORECAST:")
print(future_df.head(20).to_string(index=False))

print("\nDemand forecasting completed successfully!")
print("Best Model:", best_model_name)
print("Forecast files and charts saved in:", OUTPUT_DIR)


# ==========================================
# 9. FREQUENT PATTERN MINING - FP-GROWTH
# ==========================================

from mlxtend.frequent_patterns import fpgrowth, association_rules

print("\n====================================")
print(" FREQUENT PATTERN MINING - FP-GROWTH")
print("====================================")

# 1. SELECT TOP 40 PRODUCTS
top_40_products = (
    df.groupby("StockCode")["Quantity"]
    .sum()
    .nlargest(40)
    .index
)

pattern_df = df[df["StockCode"].isin(top_40_products)].copy()

# 2. CREATE TRANSACTION BASKET
basket = pattern_df.pivot_table(
    index="InvoiceNo",
    columns="StockCode",
    values="Quantity",
    aggfunc="sum",
    fill_value=0
)

# Convert quantities into True/False values
basket = basket > 0

print("\nTransaction basket created successfully!")
print("Number of transactions:", len(basket))
print("Number of products:", len(basket.columns))

# 3. APPLY FP-GROWTH ALGORITHM
frequent_itemsets = fpgrowth(
    basket,
    min_support=0.005,
    use_colnames=True,
    max_len=3
)

print("\nFrequent itemsets discovered:", len(frequent_itemsets))

# 4. SAVE FREQUENT ITEMSETS
if not frequent_itemsets.empty:

    frequent_itemsets["Products"] = frequent_itemsets[
        "itemsets"
    ].apply(lambda items: ", ".join(sorted(items)))

    frequent_itemsets.to_csv(
        os.path.join(OUTPUT_DIR, "frequent_itemsets.csv"),
        index=False
    )

    print("\nTOP FREQUENT PRODUCT COMBINATIONS:")
    print(
        frequent_itemsets[
            ["Products", "support"]
        ]
        .sort_values("support", ascending=False)
        .head(15)
        .to_string(index=False)
    )

    # 5. GENERATE ASSOCIATION RULES
    if len(frequent_itemsets) >= 2:

        rules = association_rules(
            frequent_itemsets,
            metric="confidence",
            min_threshold=0.3
        )

        if not rules.empty:

            # Convert product combinations into readable text
            rules["Antecedents"] = rules["antecedents"].apply(
                lambda items: ", ".join(sorted(items))
            )

            rules["Consequents"] = rules["consequents"].apply(
                lambda items: ", ".join(sorted(items))
            )

            # Sort rules by lift
            rules = rules.sort_values(
                "lift",
                ascending=False
            )

            rules.to_csv(
                os.path.join(OUTPUT_DIR, "association_rules.csv"),
                index=False
            )

            print("\nTOP 10 ASSOCIATION RULES:")
            print(
                rules[
                    [
                        "Antecedents",
                        "Consequents",
                        "support",
                        "confidence",
                        "lift"
                    ]
                ]
                .head(10)
                .to_string(index=False)
            )

            # 6. VISUALIZE TOP 10 RULES
            top_rules = rules.head(10).copy()

            rule_labels = (
                top_rules["Antecedents"]
                + " -> "
                + top_rules["Consequents"]
            )

            plt.figure(figsize=(12, 6))

            plt.barh(
                rule_labels.iloc[::-1],
                top_rules["lift"].iloc[::-1]
            )

            plt.title("Top 10 Product Association Rules")
            plt.xlabel("Lift")
            plt.ylabel("Product Association")
            plt.tight_layout()

            plt.savefig(
                os.path.join(
                    OUTPUT_DIR,
                    "top_association_rules.png"
                )
            )

            plt.close()

            print("\nAssociation rules generated:", len(rules))

        else:
            print(
                "\nNo association rules found with "
                "the current confidence threshold."
            )

    else:
        print("\nNot enough frequent itemsets to generate rules.")

else:
    print(
        "\nNo frequent itemsets found. "
        "Try reducing min_support to 0.002."
    )

print("\nFP-Growth analysis completed!")
print("Results saved in:", OUTPUT_DIR)


# ==========================================
# 10. PRODUCT RECOMMENDATION SYSTEM
# ==========================================

from sklearn.metrics.pairwise import cosine_similarity

print("\n====================================")
print(" PRODUCT RECOMMENDATION SYSTEM")
print("====================================")

# 1. CREATE CUSTOMER-PRODUCT MATRIX
recommendation_df = df[
    ["CustomerID", "StockCode", "Quantity"]
].copy()

customer_product_matrix = recommendation_df.pivot_table(
    index="CustomerID",
    columns="StockCode",
    values="Quantity",
    aggfunc="sum",
    fill_value=0
)

print("\nCustomer-product matrix created!")
print("Customers:", customer_product_matrix.shape[0])
print("Products:", customer_product_matrix.shape[1])

# 2. POPULARITY-BASED RECOMMENDATION
product_popularity = (
    recommendation_df.groupby("StockCode")["Quantity"]
    .sum()
    .sort_values(ascending=False)
)

def recommend_popular(top_n=5):

    return product_popularity.head(top_n).index.tolist()


# 3. ITEM-BASED COLLABORATIVE FILTERING
# Calculate similarity between products

product_similarity = cosine_similarity(
    customer_product_matrix.T
)

product_similarity_df = pd.DataFrame(
    product_similarity,
    index=customer_product_matrix.columns,
    columns=customer_product_matrix.columns
)


def recommend_item_based(customer_id, top_n=5):

    if customer_id not in customer_product_matrix.index:
        return []

    # Products already purchased by the customer
    purchased_products = customer_product_matrix.loc[
        customer_id
    ]

    purchased_products = purchased_products[
        purchased_products > 0
    ].index.tolist()

    if not purchased_products:
        return []

    # Calculate recommendation scores
    scores = product_similarity_df[
        purchased_products
    ].sum(axis=1)

    # Remove already purchased products
    scores = scores.drop(
        labels=purchased_products,
        errors="ignore"
    )

    return scores.sort_values(
        ascending=False
    ).head(top_n).index.tolist()


# 4. GENERATE CUSTOMER RECOMMENDATIONS
sample_customer = customer_product_matrix.index[0]

popular_recommendations = recommend_popular(5)

personalized_recommendations = recommend_item_based(
    sample_customer,
    top_n=5
)

print("\nSample Customer ID:", sample_customer)

print("\nPOPULARITY-BASED RECOMMENDATIONS:")
print(popular_recommendations)

print("\nPERSONALIZED ITEM-BASED RECOMMENDATIONS:")
print(personalized_recommendations)


# 5. SAVE RECOMMENDATIONS
recommendation_results = []

for customer_id in customer_product_matrix.index:

    recommended_products = recommend_item_based(
        customer_id,
        top_n=5
    )

    for rank, product in enumerate(
        recommended_products,
        start=1
    ):

        recommendation_results.append({
            "CustomerID": customer_id,
            "RecommendedProduct": product,
            "Rank": rank
        })


recommendation_results_df = pd.DataFrame(
    recommendation_results
)

recommendation_results_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "customer_product_recommendations.csv"
    ),
    index=False
)


# 6. EVALUATE RECOMMENDATION RESULTS
# Leave the customer's last purchased product as a test item.
# Use earlier purchases to generate recommendations.

hits = 0
total_test_customers = 0
precision_values = []
recall_values = []

for customer_id in customer_product_matrix.index:

    customer_history = customer_product_matrix.loc[
        customer_id
    ]

    purchased = customer_history[
        customer_history > 0
    ].index.tolist()

    if len(purchased) < 2:
        continue

    test_product = purchased[-1]
    training_products = purchased[:-1]

    scores = product_similarity_df[
        training_products
    ].sum(axis=1)

    scores = scores.drop(
        labels=training_products,
        errors="ignore"
    )

    recommended = scores.sort_values(
        ascending=False
    ).head(5).index.tolist()

    hit = int(test_product in recommended)

    hits += hit
    total_test_customers += 1

    precision_values.append(hit / 5)
    recall_values.append(hit)

if total_test_customers > 0:

    precision_at_5 = np.mean(precision_values)
    recall_at_5 = np.mean(recall_values)

    print("\nRECOMMENDATION EVALUATION:")
    print("Test customers:", total_test_customers)
    print("Precision@5:", round(precision_at_5, 4))
    print("Recall@5:", round(recall_at_5, 4))

    evaluation_df = pd.DataFrame([{
        "Precision@5": precision_at_5,
        "Recall@5": recall_at_5,
        "TestCustomers": total_test_customers
    }])

    evaluation_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "recommendation_evaluation.csv"
        ),
        index=False
    )

else:
    print("\nNot enough customer purchase history for evaluation.")


print("\nProduct recommendation system completed!")
print("Recommendation files saved in:", OUTPUT_DIR)


# ==========================================
# 11. FINAL PROJECT SUMMARY
# ==========================================

print("\n====================================")
print(" FINAL PROJECT SUMMARY")
print("====================================")

# 1. COLLECT PROJECT STATISTICS
total_records = len(df)
total_customers = df["CustomerID"].nunique()
total_products = df["StockCode"].nunique()
total_transactions = df["InvoiceNo"].nunique()
total_sales = df["TotalSales"].sum()

best_model = metrics_df.sort_values("MAE").iloc[0]

summary = {
    "Metric": [
        "Total Transactions",
        "Total Customers",
        "Total Products",
        "Total Cleaned Records",
        "Total Sales",
        "Best Forecasting Model",
        "Best Model MAE",
        "Best Model RMSE",
        "Best Model R2",
        "Frequent Itemsets Discovered",
        "Customers with Recommendations"
    ],
    "Value": [
        total_transactions,
        total_customers,
        total_products,
        total_records,
        round(total_sales, 2),
        best_model["Model"],
        round(best_model["MAE"], 4),
        round(best_model["RMSE"], 4),
        round(best_model["R2"], 4),
        len(frequent_itemsets),
        recommendation_results_df["CustomerID"].nunique()
    ]
}

summary_df = pd.DataFrame(summary)

# 2. SAVE PROJECT SUMMARY
summary_df.to_csv(
    os.path.join(OUTPUT_DIR, "final_project_summary.csv"),
    index=False
)

# 3. SAVE A TEXT REPORT
report_path = os.path.join(
    OUTPUT_DIR,
    "final_project_report.txt"
)

with open(report_path, "w", encoding="utf-8") as report:

    report.write("INTELLIGENT RETAIL ANALYTICS PROJECT\n")
    report.write("=" * 45 + "\n\n")

    report.write("PROJECT OVERVIEW\n")
    report.write("This project analyzes retail transactions, forecasts product demand, ")
    report.write("discovers frequently purchased product combinations, and recommends products.\n\n")

    report.write("DATASET SUMMARY\n")
    report.write(f"Total transactions: {total_transactions}\n")
    report.write(f"Total customers: {total_customers}\n")
    report.write(f"Total products: {total_products}\n")
    report.write(f"Cleaned records: {total_records}\n")
    report.write(f"Total sales: {total_sales:.2f}\n\n")

    report.write("DEMAND FORECASTING\n")
    report.write(f"Best model: {best_model['Model']}\n")
    report.write(f"MAE: {best_model['MAE']:.4f}\n")
    report.write(f"RMSE: {best_model['RMSE']:.4f}\n")
    report.write(f"R2: {best_model['R2']:.4f}\n\n")

    report.write("FREQUENT PATTERN MINING\n")
    report.write(f"Frequent itemsets discovered: {len(frequent_itemsets)}\n\n")

    report.write("PRODUCT RECOMMENDATIONS\n")
    report.write(
        f"Customers receiving recommendations: "
        f"{recommendation_results_df['CustomerID'].nunique()}\n"
    )

    report.write("\nFORECAST PERIOD\n")
    report.write("Next four weeks of product demand\n")

# 4. VISUALIZE FUTURE DEMAND
forecast_summary = (
    future_df.groupby("StockCode")["PredictedDemand"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

plt.figure(figsize=(10, 6))

forecast_summary.sort_values().plot(kind="barh")

plt.title("Top 10 Products by Forecasted Demand")
plt.xlabel("Predicted Quantity")
plt.ylabel("Product Code")
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "top_forecasted_products.png")
)

plt.close()

# 5. VISUALIZE MOST RECOMMENDED PRODUCTS
top_recommended = (
    recommendation_results_df["RecommendedProduct"]
    .value_counts()
    .head(10)
)

plt.figure(figsize=(10, 6))

top_recommended.sort_values().plot(kind="barh")

plt.title("Top 10 Most Recommended Products")
plt.xlabel("Number of Recommendations")
plt.ylabel("Product Code")
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "top_recommended_products.png")
)

plt.close()

# 6. DISPLAY FINAL SUMMARY
print("\nFINAL PROJECT RESULTS:")
print(summary_df.to_string(index=False))

print("\nFinal summary and report saved successfully!")
print("Final charts generated successfully!")
print("All project outputs are saved in:", OUTPUT_DIR)

print("\n====================================")
print(" PROJECT EXECUTION COMPLETED")
print("====================================")
