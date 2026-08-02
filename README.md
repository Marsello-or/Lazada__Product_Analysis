# Lazada Product Analysis: Optimizing Revenue and Operational Costs

**Live Dashboard Deployment:** [Lazada Analytics Streamlit App](https://lazada-model-deployment.streamlit.app/)

## Libraries needed for streamlit deployment

* streamlit
* pandas
* numpy
* scikit-learn
* xgboost
* joblib

## Table of Contents

1. [Project Overview](https://www.google.com/search?q=%231-project-overview)
2. [Business Statement & Understanding](https://www.google.com/search?q=%232-business-statement--understanding)
3. [Business Questions & Problems](https://www.google.com/search?q=%233-business-questions--problems)
4. [Data Source](https://www.google.com/search?q=%234-data-source)
5. [Data Understanding & Quality Check](https://www.google.com/search?q=%235-data-understanding--quality-check)
6. [Data Preparation & Feature Engineering](https://www.google.com/search?q=%236-data-preparation--feature-engineering)
7. [Exploratory Data Analysis (EDA) & Business Insights](https://www.google.com/search?q=%237-exploratory-data-analysis-eda--business-insights)
* [Q1: Category Pareto Analysis](https://www.google.com/search?q=%23q1-category-pareto-analysis)
* [Q2: Discount Effectiveness & Price Tiers](https://www.google.com/search?q=%23q2-discount-effectiveness--price-tiers)
* [Q3: LazMall & Super Seller Program Impact](https://www.google.com/search?q=%23q3-lazmall--super-seller-program-impact)
* [Q4: Factors Influencing Product Sales](https://www.google.com/search?q=%23q4-factors-influencing-product-sales)


8. [Sales Prediction Modeling](https://www.google.com/search?q=%238-sales-prediction-modeling)
9. [Product Segmentation Modeling](https://www.google.com/search?q=%239-product-segmentation-modeling)
10. [Final Strategic Recommendations](https://www.google.com/search?q=%2310-final-strategic-recommendations)

## 1. Project Overview

This project aims to help Lazada, a leading e-commerce platform in Southeast Asia, optimize its revenue and reduce operational costs. By analyzing publicly available product data, the goal is to uncover actionable business insights, build predictive models for sales, and classify products into meaningful segments to drive targeted marketing and inventory strategies.

## 2. Business Statement & Understanding

Lazada operates across six markets in Southeast Asia with approximately 160 million monthly active users and 1 million active sellers. As a data analyst, the objective is to leverage data to identify key areas for revenue maximization and operational efficiency.

## 3. Business Questions & Problems

1. Which categories and products generate the highest GMV, and how should Lazada allocate its inventory and marketing resources?
2. How effective are discounts in driving sales, and what is the optimal discount range for each product category?
3. How do LazMall and Super Seller programs impact seller performance, and is the platform's investment justified?
4. What factors (rating, reviews, price, seller reputation, promotions) most significantly influence product sales?
5. Can we build a machine learning model to predict product sales volume for inventory planning?
6. How can we classify products into meaningful segments for targeted marketing and pricing strategies?

## 4. Data Source

The dataset used for this analysis was sourced from:
[https://github.com/luminati-io/eCommerce-dataset-samples/blob/main/lazada-products.csv](https://github.com/luminati-io/eCommerce-dataset-samples/blob/main/lazada-products.csv)

## 5. Data Understanding & Quality Check

* **Data Dimension:** 1,000 rows and 29 columns.
* **Data Types:** Generally appropriate, with some object columns requiring further processing.
* **Missing Values:** Detected in `color` (50%), `colors` (50%), `seller_chat_response` (12.2%), `seller_ship_on_time` (6.7%), and `seller_ratings` (2.7%). These were handled during data preparation.
* **Duplicate Rows:** No duplicate rows found.
* **Unique Values:** Good distribution across columns, indicating varied data.
* **Descriptive Statistics:** Provided insights into numerical and categorical data distributions.

## 6. Data Preparation & Feature Engineering

* **Cleaning String Percentages:** Converted `seller_ship_on_time` and `seller_chat_response` from percentage strings to numeric values.
* **Handling Missing Values:** Imputed missing numeric seller features with their median, and filled missing `color`/`colors` with descriptive strings.
* **Breadcrumb Extraction:** Created `main_category` and `sub_category` from the `breadcrumb` column.
* **Discount Calculation:** Engineered `discount_amount` and `discount_pct`.
* **Color Variation Count:** Created `color_variant_count` and `has_multiple_colors`.
* **Price & Sales Tiers:** Categorized `final_price` into `price_tier` and `number_sold` into `sales_tier`.
* **Boolean Flags:** Cleaned `is_super_seller` and `lazmall` boolean columns.
* **Feature Engineering for ML:** Created `target_log`, `reviews_per_price`, `seller_tier`, and `operational_score`.

## 7. Exploratory Data Analysis (EDA) & Business Insights

### Q1: Category Pareto Analysis (80/20 Rule for Revenue)

* **Summary:** 'Mobiles & Tablets' accounts for over 86% of total GMV. The top 3 categories (Mobiles & Tablets, Computers & Laptops, Beauty) generate over 97% of revenue.
* **Recommendations:** Prioritize resources for core revenue drivers (Mobiles & Tablets, Computers & Laptops). Diversify revenue by growing Class B categories (e.g., Beauty). Leverage long-tail categories for cross-selling.

### Q2: Discount Effectiveness & Price Tiers

* **Summary:** Discount effectiveness varies by price tier. Low-tier products (<100k IDR) show high price elasticity, while higher tiers are less responsive.
* **Recommendations:** Aggressive discounts (40%–60%) for Low Tier. Moderate discounts (10%–20%) for Mid Tier. Limited discounts (max 5%–15%) for High/Premium Tier, focusing on value-added promotions.

### Q3: LazMall & Super Seller Program Impact

* **Summary:** Both programs significantly boost seller performance.
* **LazMall:** Drives ~45x higher average GMV.
* **Super Seller:** Increases average sales volume by ~3x.


* **Recommendations:** Promote LazMall for high-value/branded products and Super Seller for high-volume products. Continue leveraging platform features to amplify their growth.

### Q4: Factors Influencing Product Sales

* **Summary:** Sales volume is overwhelmingly influenced by social proof (`reviews` correlation: 0.94). `discount_pct` (0.25) and `rating` (0.16) have weaker impacts. Operational metrics show negligible direct correlation.
* **Recommendations:** Prioritize incentivizing post-purchase reviews. Highlight high-quality customer reviews. Treat operational KPIs as hygiene factors rather than primary sales drivers.

## 8. Sales Prediction Modeling

* **Objective:** Predict product sales volume (`number_sold`) for inventory planning.
* **Approach:** XGBoost Regressor model was built after log-transforming the target variable (`number_sold`) due to its skewed distribution. Features included `reviews`, `discount_pct`, `rating`, `final_price`, `reviews_per_price`, `operational_score`, `seller_tier`, and `color_variant_count`.
* **Performance:** Achieved an R² of 0.91 and a Mean Absolute Error (MAE) of approximately 144 units on the original scale.
* **Impact:** Enables optimized safety stock levels, prevents stockouts, and reduces overstock risks.

## 9. Product Segmentation Modeling

* **Objective:** Classify products into meaningful segments for targeted marketing and pricing strategies.
* **Approach:** K-Means clustering was applied to key features (`number_sold`, `gmv`, `final_price`, `reviews`) after log transformation and scaling. The Elbow Method and Silhouette Analysis indicated an optimal `k=5` clusters.
* **Segments:** Products were classified into 5 tiers:
* **Tier 1 (Star Products):** High GMV, volume, engagement, premium pricing (Cash Cows).
* **Tier 2 (High Potential):** High GMV due to pricing, but lower volume/reviews (Margin Makers).
* **Tier 3 (Mid-Range):** Highest sales volume and reviews, but lowest price points (Traffic Drivers).
* **Tier 4 (Slow Movers):** Cheap items with zero traction (Noise).
* **Tier 5 (Low Potential):** Most expensive, absolute worst sales volume and GMV (Dead Weight).


* **Impact:** Provides clear operational profiles for each segment, allowing for highly precise marketing and inventory strategies.

## 10. Final Strategic Recommendations

### Revenue Optimization Tactics

* **Tier 1 (Star Products):** Secure supply chain, FBL onboarding, prime visibility.
* **Tier 2 (High Potential):** Conversion nudging (flash sales, installment plans).
* **Tier 3 (Mid-Range):** Boost Average Order Value (AOV) with bundling and cross-selling.

### Cost-Cutting & Efficiency Tactics

* **Tier 4 (Slow Movers):** Algorithm demotion, advise sellers to clear inventory or delist.
* **Tier 5 (Low Potential):** Stop subsidized marketing, liquidate inventory, free up warehouse capacity.
