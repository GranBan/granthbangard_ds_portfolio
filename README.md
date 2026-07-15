# Granth Bangard | Data Science Portfolio

End-to-end Machine Learning and Data Science projects spanning predictive modeling, NLP, explainable AI, and interactive deployment.

![Python](https://img.shields.io/badge/Python-3.13-blue) ![XGBoost](https://img.shields.io/badge/XGBoost-ML-success) ![Transformers](https://img.shields.io/badge/Transformers-NLP-orange) ![SHAP](https://img.shields.io/badge/SHAP-Explainability-red) ![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-brightgreen)

**2 Production-style ML Apps** · **100K+ Reviews** · **2.39M Sports Events** · **11+ Million Data Points Processed** · **10,458 Cricket Matches** · **Interactive Streamlit Deployments**

Every project follows the same engineering workflow: build an interpretable model, evaluate it rigorously, document key design decisions, and deploy it as an interactive application.

---

> **Featured Projects:** Click any project below to explore the notebook, live application, and complete technical documentation.

## Contents

- [Skills](#skills)
- [Projects](#projects)
- [Other Projects](#other-projects)
- [Contact](#contact)

---

## Skills

| Category | Tools |
|---|---|
| **Languages** | Python, SQL |
| **ML** | scikit-learn, XGBoost, Logistic Regression, Random Forest |
| **NLP** | Transformers, DistilBERT, BERTopic |
| **Explainability** | SHAP |
| **Validation** | Temporal train/test splits, calibration (isotonic regression) |
| **Visualization** | Plotly, Matplotlib, Seaborn |
| **Deployment** | Streamlit Community Cloud |

---

## Projects

### [Cricket T20 Win Probability Predictor](Cricket%20T20%20Win%20Probability%20Predictor)

![Homepage](Cricket%20T20%20Win%20Probability%20Predictor/assets/screenshots/homepage_hero_t20.png)

- **Dataset:** 10,458 matches · 2.39M deliveries · 10 T20 leagues (2008-2026)
- **Model:** Calibrated XGBoost, benchmarked against a Logistic Regression baseline
- **Results:** AUC 0.924 · Log Loss 0.345 · Brier Score 0.112, on a chronologically held-out 2025-2026 test set
- Temporal train/test split to prevent leakage, isotonic calibration, and SHAP-driven feature selection (three zero-importance features removed with no performance cost)
- Deployed as a 5-page Streamlit app with live prediction, full match replay, and 21 famous match case studies

🔗 [Notebook](https://github.com/GranBan/granthbangard_ds_portfolio/blob/main/Cricket%20T20%20Win%20Probability%20Predictor/t20_win_probability.ipynb) &nbsp;·&nbsp; 🔗 [Live App](https://cricket-t20-win-probability.streamlit.app) &nbsp;·&nbsp; 🔗 [App Source](https://github.com/GranBan/cricket-t20-win-probability)

---

### [Spotify NLP App Review Intelligence System](Spotify%20NLP%20App%20Review%20Intelligence%20System)

![Homepage](Spotify%20NLP%20App%20Review%20Intelligence%20System/assets/screenshots/homepage_hero_spotify.png)

- **Dataset:** 100,000 Google Play reviews, scraped directly, 51 discovered complaint topics
- **Model:** Fine-tuned DistilBERT (binary sentiment), BERTopic clustering
- **Results:** Macro F1 0.935 · 96.2% average prediction confidence
- Identified a real regression (versions 9.1.46/9.1.48, 77% and 75% negative sentiment) via temporal trend analysis
- Custom priority matrix scores complaints by frequency, severity, and recent trend, converting NLP output into a ranked, actionable fix list
- Deployed as a 7-page Streamlit app with topic exploration, version trends, and misclassification analysis

🔗 [Notebook](https://github.com/GranBan/granthbangard_ds_portfolio/blob/main/Spotify%20NLP%20App%20Review%20Intelligence%20System/app_review_nlp.ipynb) &nbsp;·&nbsp; 🔗 [Live App](https://spotify-review-intelligence.streamlit.app) &nbsp;·&nbsp; 🔗 [App Source](https://github.com/GranBan/spotify-review-intelligence)

---

## Other Projects

### [Customer Churn Analysis and Retention Intelligence](granthbangard_ds_portfolio/Customer%Churn%and%Retention%Intelligence)

- Built a business-focused churn intelligence system to identify high-risk users and key behavioral drivers of churn
- Achieved 82% accuracy, identifying 35% of high-risk customers, enabling targeted retention strategies
- End-to-end analysis: data cleaning, EDA, feature engineering, and machine learning modeling optimized for recall and ROC-AUC
- Retention strategies by risk segment estimated to reduce churn by ~12-15%

🔗 [Notebook](https://github.com/GranBan/granthbangard_ds_portfolio/blob/main/Customer%20Churn%20and%20Retention%20Intelligence/Customer%20Churn%20%26%20Retention.ipynb)

---

### [Badminton Match Analysis | BWF World Tour 2018-2021](granthbangard_ds_portfolio/badminton-match-analysis)

- Analyzed 6,736 professional matches to study country dominance, match competitiveness, and COVID-era disruptions
- Built a logistic regression model to predict whether a match goes to 3 sets
- Men's singles go to 3 sets ~35% of the time; women's singles ~30%
- China dominates women's singles; India & Denmark lead men's singles
- COVID-19 caused an ~83% drop in matches in 2020 vs. 2019

🔗 [Notebook](https://github.com/GranBan/granthbangard_ds_portfolio/blob/main/badminton-match-analysis/badminton-match-analysis.ipynb)

---

## Contact

GitHub: https://github.com/GranBan  
LinkedIn: https://linkedin.com/in/granth-bangard  
Email: bangardgranth@gmail.com (Primary), bangard2@illinois.edu
