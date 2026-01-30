# Badminton Match Analysis (BWF World Tour 2018–2021)

This project analyzes 6,736 professional badminton matches from the BWF World Tour between 2018 and 2021. It explores trends in match competitiveness, country performance, player behavior, and includes a predictive model to identify highly competitive matches (three-set outcomes).


## Project Objectives
- Understand competitive patterns across men's and women's singles
- Analyze country dominance and tournament-level differences
- Explore how match characteristics influence competitiveness
- Build a predictive model to classify 2-set vs 3-set matches
- Communicate insights through clear visuals and explanations


## Key Insights
- Three-set matches were more common in evenly ranked matchups, indicating competitiveness is strongly tied to parity
- Post-COVID seasons showed shorter average match durations, suggesting changes in tournament structure and player conditioning
- Countries such as India and China consistently produced longer, more competitive matches across tournaments


## Methods & Techniques

### **1. Exploratory Data Analysis (EDA)**
- Score distributions and match duration patterns  
- Country-wise win trends  
- Comparison of match types (MS vs WS)  
- Pre-/post-COVID participation and competitiveness changes  

### **2. Feature Engineering**
- Creating competitiveness indicators  
- Encoding match category  
- Extracting score-based features  
- Handling missing or inconsistent entries  

### **3. Machine Learning Model**
- Logistic Regression classifier  
- Train/test split, scaling, and evaluation  
- Metrics: accuracy, precision, recall, F1-score  
- Achieved balanced precision and recall, indicating the model’s ability to identify competitive matches beyond simple baselines.


## Project Structure

badminton-match-analysis/  
-badminton-match-analysis.ipynb  
-README.md  
  |- Plots  


## Skills Demonstrated
- Python (pandas, NumPy, Matplotlib, seaborn)
- Data cleaning and preprocessing  
- Exploratory Data Analysis (EDA)  
- Feature engineering  
- Logistic Regression (scikit-learn)  
- Data storytelling and visualization  


## View the Notebook
👉 **Open the notebook:** https://github.com/GranBan/granthbangard_data-science_portfolio/blob/main/badminton-match-analysis/badminton-match-analysis.ipynb


## Future Improvements
- Add Random Forest/XGBoost models for comparison  
- Player ranking analysis over time  
- Tournament-specific prediction models  
- Interactive dashboards (Plotly/Streamlit)  

