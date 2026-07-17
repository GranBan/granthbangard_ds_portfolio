# Customer Churn and Retention Intelligence

Predicting customer churn is valuable only when predictions translate into business decisions. This project analyzes customer behavior in a subscription-based telecommunications company to identify churn drivers, develop machine learning models, and recommend targeted retention strategies.

Rather than optimizing purely for predictive performance, the project prioritizes identifying customers at high risk of churning while maintaining model interpretability for business stakeholders.

---

## Project Objectives

- Understand behavioral, contractual, and billing differences between churned and retained customers.
- Identify the strongest drivers of customer churn through exploratory analysis.
- Build and compare machine learning models for churn prediction.
- Translate model outputs into actionable customer retention strategies.

---

## Key Insights

- Approximately **27%** of customers churn, making retention a significant business challenge.
- Customers with **short tenure** are substantially more likely to churn, highlighting the importance of early customer engagement.
- **Month-to-month contracts** exhibit considerably higher churn than longer-term contracts.
- Customers paying **higher monthly charges** show greater churn risk, particularly during the early stages of their subscription.
- Low engagement with value-added services such as **Tech Support** and **Online Security** is strongly associated with customer attrition.

---

## Methods & Techniques

### 1. Data Preparation

- Converted `TotalCharges` to numeric and removed incomplete observations.
- Removed non-predictive customer identifiers.
- Encoded the churn target as a binary classification problem.

### 2. Exploratory Data Analysis

- Analyzed churn distribution and class imbalance.
- Investigated churn across tenure, contract type, monthly charges, and service adoption.
- Identified behavioral patterns that informed feature engineering and model development.

### 3. Feature Engineering

Several business-oriented features were engineered to better capture customer behavior:

- **Tenure Groups** representing customer lifecycle stages.
- **Contract Commitment** encoding switching friction across contract types.
- **Engagement Score** aggregating subscribed value-added services.
- **Price Intensity** measuring monthly charges relative to customer tenure.

### 4. Machine Learning

Developed and evaluated multiple classification models:

- Logistic Regression with class balancing
- Random Forest
- Threshold optimization for business-driven decision making

The final Logistic Regression model achieved:

| Model | ROC-AUC | Recall (Churn) | Precision (Churn) | F1-score |
|------|------:|------:|------:|------:|
| **Logistic Regression** | **0.830** | **0.877** | 0.449 | **0.61** |
| Random Forest | 0.801 | 0.583 | **0.563** | 0.55 |

Although Random Forest produced higher precision, Logistic Regression achieved substantially higher recall, making it better suited for proactive customer retention where identifying at-risk customers is the primary objective.

### 5. Decision Layer

Rather than relying on the default probability threshold, multiple thresholds were evaluated to balance recall and precision.

A **0.40 classification threshold** was selected to maximize detection of churn-prone customers while maintaining reasonable precision.

Predicted probabilities were then converted into business-friendly customer segments:

- High Risk
- Medium Risk
- Low Risk

These segments enable targeted retention campaigns instead of blanket incentives.

---

## Business Recommendations

Model insights translate directly into actionable retention strategies:

- **Early-tenure intervention** through onboarding and proactive customer outreach.
- **Contract conversion incentives** encouraging high-risk customers to move to longer-term plans.
- **Targeted retention offers** focused on customers exceeding the churn-risk threshold.
- **Service engagement campaigns** promoting value-added services that improve customer stickiness.

---

## Repository Structure

```
Customer Churn and Retention Intelligence/
│
├── Customer Churn & Retention.ipynb
├── README.md
└── Plots/
```

---

## Skills Demonstrated

- Python (pandas, NumPy, Matplotlib, seaborn)
- Data Cleaning & Preprocessing
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Logistic Regression
- Random Forest
- Threshold Optimization
- Customer Risk Segmentation
- Business Analytics
- Data Storytelling

---

## View the Notebook

👉 **Open the notebook:**  
https://github.com/GranBan/granthbangard_ds_portfolio/blob/main/Customer%20Churn%20and%20Retention%20Intelligence/Customer%20Churn%20%26%20Retention.ipynb

---

## Future Improvements

- Incorporate customer lifetime value into model optimization.
- Evaluate gradient boosting methods such as XGBoost and LightGBM.
- Develop cost-sensitive models balancing retention cost against customer value.
- Validate retention strategies through A/B testing.
- Deploy an interactive Streamlit dashboard for real-time churn monitoring.
