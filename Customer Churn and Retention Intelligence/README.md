# **Customer Churn and Retention Intelligence Analysis**

This project analyzes customer behavior in a subscription-based telecom company to **predict churn risk, identify key churn drivers,** and **translate insights into actionable retention strategies** aligned with business objectives.

The analysis emphasizes decision-making under trade-offs, prioritizing high-risk customer identification over pure predictive accuracy.


## Project Objectives

- Understand behavioral, contractual, and billing differences between churned and retained customers.
- Identify the strongest drivers of customer churn.
- Build and evaluate churn prediction models using business-relevant metrics.
- Translate insights and model outputs into targeted retention strategies.


## Key Insights

- **Churn Rate:** ~27% of customers churn, indicating a substantial retention challenge.
- Tenure: Short-tenured customers are significantly more likely to churn, highlighting early lifecycle risk.
- Contract Type: Month-to-month contracts show higher churn risk than longer-term plans.
- Monthly Charges: Higher charges correlate with higher churn probability, particularly among newer customers.
- Service Engagement: Customers lacking value-added services (Tech Support, Streaming, Security) are more likely to leave, highlighting engagement as a retention lever.


## Methods & Techniques

### 1. Data Cleaning & Preparation
- Converted TotalCharges to numeric and removed incomplete records
- Dropped non-informative identifiers (customerID)
- Encoded churn as a binary target variable {0 (No) / 1 (Yes)}

### 2. Exploratory Data Analysis (EDA)
- Identified patterns in tenure, contract type, monthly charges, and service usage
- Visualized churn distribution and feature relationships using seaborn
- Identified early behavioral signals of churn risk

### 3. Feature Engineering
- Tenure Groups: Categorized customers into lifecycle blocks (0–12, 12–24, 24–48, 48+)
- Contract Commitment: Encoded month-to-month, one-year, and two-year contracts as ordered features
- Engagement Score: Aggregated value-added services (Tech Support, Streaming, Security)
- Price Sensitivity: Normalized monthly charges by tenure (charges_per_tenure)

### 4. Machine Learning Models
- Logistic Regression: Baseline model with class balancing and threshold tuning
- Random Forest: Captured non-linear relationships and improved predictive performance
- Threshold Analysis: Evaluated precision-recall trade-offs; selected 0.4 to maximize churn recall

### 5. Risk Segmentation
- Classified customers into High, Medium, Low churn risk based on predicted probabilities
- Designed actionable retention strategies for each segment


## Model Performance

### Model            	  ROC-AUC	    Recall (Churn)	  Precision (Churn)    F1-Score (Churn)
Logistic Regression	    0.830	      0.877	            0.449	               0.61
Random Forest	          0.801	      0.583	            0.563	               0.55


****Final Model:**** Logistic Regression was selected for its interpretability and ability to guide actionable retention decisions.


## Retention Strategy Recommendations

- **Early-Tenure Intervention:** Target customers within the first 6–12 months for proactive engagement
- **Contract Stabilization:** Incentivize high-risk customers to switch from month-to-month to longer-term plans
- **Targeted Retention Offers:** Apply offers only to customers flagged above the churn risk threshold
- **Engagement Nudges:** Promote value-added services to increase customer stickiness


## Skills Demonstrated

- Python (pandas, NumPy, Matplotlib, seaborn)
- Data cleaning and preprocessing
- Exploratory Data Analysis (EDA)
- Feature engineering for behavioral modeling and churn prediction
- Logistic Regression & Random Forest (scikit-learn)
- Business-oriented data storytelling & actionable insights

Project Structure
telco-customer-churn/
├─ telco-churn-analysis.ipynb
├─ README.md
├─ Plots/


## View the Notebook

Open the notebook: https://github.com/GranBan/granthbangard_ds_portfolio/blob/main/Customer%20Churn%20and%20Retention%20Intelligence/Customer%20Churn%20%26%20Retention.ipynb


## Future Improvements

- Cost-sensitive modeling incorporating customer lifetime value
- Two-stage models combining high-recall and precision-focused models
- Dynamic thresholds based on customer segments
- A/B testing retention strategies to validate real-world impact
