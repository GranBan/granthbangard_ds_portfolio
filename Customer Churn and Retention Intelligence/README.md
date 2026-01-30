# Telco Customer Churn Analysis

This project analyzes customer behavior in a subscription-based telecom company using the Telco Customer Churn Dataset. The goal is to identify customers likely to churn, understand the key drivers behind churn, and provide actionable retention strategies that align with business objectives.


## Project Objectives

- Explore behavioral, contractual, and billing differences between churned and retained customers

- Identify factors that most strongly influence churn

- Build a predictive model to assess churn risk

- Translate insights into actionable retention strategies for the business


## Key Insights

- Churn Distribution: ~27% of customers churn, indicating a substantial retention challenge.

- Tenure: Shorter-tenured customers are more likely to churn; early engagement is critical.

- Contract Type: Month-to-month contracts show higher churn risk than longer-term plans.

- Monthly Charges: Higher charges correlate with higher churn probability, especially among newer customers.

- Service Engagement: Customers lacking value-added services (Tech Support, Streaming, Security) are more likely to leave, highlighting engagement as a retention lever.


## Methods & Techniques

1. Data Cleaning & Preparation

Converted TotalCharges to numeric and removed incomplete records

Dropped non-informative columns (customerID)

Encoded the target variable Churn as 0 (No) / 1 (Yes)

2. Exploratory Data Analysis (EDA)

Visualized churn distribution and feature relationships using seaborn

Identified patterns in tenure, contract type, monthly charges, and service usage

3. Feature Engineering

Tenure Groups: Categorized customers into lifecycle blocks (0–12, 12–24, 24–48, 48+)

Contract Commitment: Encoded month-to-month, one-year, and two-year contracts as ordered features

Engagement Score: Aggregated value-added services (Tech Support, Streaming, Security)

Price Sensitivity: Normalized monthly charges by tenure (charges_per_tenure)

4. Machine Learning Models

Logistic Regression: Baseline model with class balancing and threshold tuning

Random Forest: Captured non-linear relationships and improved predictive performance

Threshold Analysis: Evaluated precision-recall trade-offs; selected 0.4 to maximize churn recall

5. Risk Segmentation

Classified customers into High, Medium, Low churn risk based on predicted probabilities

Designed actionable retention strategies for each segment

Model Performance
Model	ROC-AUC	Recall (Churn)	Precision (Churn)	F1-Score (Churn)
Logistic Regression	0.830	0.877	0.449	0.61
Random Forest	0.801	0.583	0.563	0.55

Final Model: Logistic Regression was selected for its interpretability and ability to guide actionable retention decisions.

Retention Strategy Recommendations

Early-Tenure Intervention: Target customers within the first 6–12 months for proactive engagement

Contract Stabilization: Encourage high-risk customers to switch from month-to-month to longer-term plans

Targeted Offers: Focus retention incentives on customers flagged by the model

Engagement Nudges: Promote value-added services to increase stickiness

Skills Demonstrated

Python (pandas, NumPy, Matplotlib, seaborn)

Data cleaning and preprocessing

Exploratory Data Analysis (EDA)

Feature engineering for churn prediction

Logistic Regression & Random Forest (scikit-learn)

Data storytelling & actionable business insights

Project Structure
telco-customer-churn/
├─ telco-churn-analysis.ipynb
├─ README.md
├─ Plots/

View the Notebook

👉 Open the notebook: Telco Customer Churn Analysis

Future Improvements

Cost-sensitive modeling incorporating customer lifetime value

Two-stage models combining high-recall and precision-focused models

Dynamic thresholds based on customer segments

A/B testing retention strategies to validate real-world impact
