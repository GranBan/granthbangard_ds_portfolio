# granthbangard_ds_portfolio

A portfolio of Data Science and ML projects where I explore data, build models, and explain insights in a clear, practical way. Each project walks through the full workflow, from cleaning and visualization to feature engineering and prediction using Python. Built to showcase my hands-on approach to analytics and readiness for internships in DS/ML.
<br>
## Projects

### 1. Cricket T20 Win Probability Predictor

- Built an XGBoost win probability model predicting the chasing team's chances at any point in a T20 match, trained on 10,458 professional matches across 10 international leagues (2008-2026).
- Used a genuine temporal train/test split (train 2008-2024, test 2025-2026) to prevent data leakage, and validated results against real, dramatic matches including the 2024 T20 World Cup Final.
- Compared XGBoost against a Logistic Regression baseline, applied isotonic calibration, and used SHAP to identify and remove three zero-importance features.
- Key Results:
  - AUC 0.924, Log Loss 0.345, Brier Score 0.112 on unseen 2025-2026 matches
  - Deployed as a 5-page interactive Streamlit app with live prediction, full match replay, and 21 famous match case studies

Notebook: https://github.com/GranBan/granthbangard_ds_portfolio/blob/main/Cricket%20T20%20Win%20Probability%20Predictor/t20_win_probability.ipynb  
  
Live App: https://cricket-t20-win-probability.streamlit.app
  
  
### 2. Spotify NLP App Review Intelligence System

- Built an end-to-end NLP pipeline analyzing 100,000 Google Play reviews, combining a fine-tuned DistilBERT sentiment classifier with BERTopic topic modeling to surface and prioritize product issues.
- Fine-tuned DistilBERT for binary sentiment classification (93.5% macro F1), discovered 51 complaint themes via embedding-based topic modeling, and tracked sentiment across app versions to identify a real regression.
- Built a custom priority matrix scoring complaints by frequency, severity, and recent trend, translating NLP output directly into actionable product decisions.
- Key Results:
  - Identified versions 9.1.46 and 9.1.48 as a crash regression (77% and 75% negative sentiment rate)
  - Deployed as a 7-page interactive Streamlit app with topic exploration, version trends, and a ranked priority matrix

Notebook: https://github.com/GranBan/granthbangard_ds_portfolio/blob/main/Spotify%20NLP%20App%20Review%20Intelligence%20System/app_review_nlp.ipynb  
  
Live App: https://spotify-review-intelligence.streamlit.app
  
  
### 3. Customer Churn Analysis and Retention Intelligence
- Built a **business-focused churn intelligence system** to identify high-risk users and key behavioral drivers of churn.
- Achieved **82% accuracy**, identifying **35% of high-risk customers**, enabling targeted retention strategies.
- Performed end-to-end analysis: **data cleaning, exploratory analysis, feature engineering**, and **machine learning modeling** optimized for recall and **ROC-AUC**.
- Translated model outputs into actionable retention strategies by customer risk segment that could **reduce churn by ~12–15%**, bridging predictive analytics with real-world business decisions.

**Notebook:** https://github.com/GranBan/granthbangard_ds_portfolio/blob/main/Customer%20Churn%20and%20Retention%20Intelligence/Customer%20Churn%20%26%20Retention.ipynb
  
  
### 4. Badminton Match Analysis | BWF World Tour 2018–2021
- Analyzed **6,736 professional badminton matches** to study country dominance, match competitiveness, and COVID-era disruptions.
- Built a **logistic regression model** to predict whether a match goes to **3 sets**.
- Performed end-to-end analysis including EDA, visualizations (bar, line, box plots), feature engineering, and model evaluation.
- Key Insights:
  - Men’s singles go to 3 sets ~35% of the time; women’s singles ~30%.
  - China dominates women’s singles; India & Denmark lead men’s singles.
  - COVID-19 caused an ~83% drop in matches in 2020 vs. 2019.

**Notebook:** https://github.com/GranBan/granthbangard_ds_portfolio/blob/main/badminton-match-analysis/badminton-match-analysis.ipynb  
  
  
## Skills Highlighted

- Python, pandas, NumPy, scikit-learn, XGBoost, transformers, BERTopic, SHAP, Matplotlib, Seaborn, Plotly
- Exploratory Data Analysis (EDA)
- Data Cleaning and Model Evaluation
- Feature Engineering
- Machine Learning Models (Logistic Regression, Random Forest, XGBoost)
- Transformer Fine-Tuning and NLP (DistilBERT, BERTopic)
- Temporal Validation, Model Calibration, and Explainability (SHAP)
- End-to-End Deployment (Streamlit Community Cloud)
- Data Storytelling and Visualization  
  
  
## Contact

**GitHub:** https://github.com/GranBan  
**LinkedIn:** https://linkedin.com/in/granth-bangard  
**Email:** bangardgranth@gmail.com (Primary), bangard2@illinois.edu
