# Cricket T20 Win Probability Predictor

Predicts the probability that the chasing team wins a T20 cricket match from any completed over using ball-by-ball historical data. Trained on over 2.3 million deliveries from 10,458 professional matches spanning 10 international leagues.

## Project Objectives

- Build a win probability model that updates after every over of a T20 chase
- Validate the model using a genuine temporal train/test split, not random shuffling
- Compare a baseline model against a more complex model and justify the choice with evidence
- Calibrate predicted probabilities so they reflect real-world win rates
- Explain individual predictions using SHAP, not just report aggregate accuracy
- Deploy a working, interactive product, not just a notebook

## Key Insights

- **Pressure differential (`crr_rrr_difference`) is the single strongest predictor**, more important than raw runs required or wickets in hand individually.
- **Rolling momentum features matter**: teams scoring heavily in the last 3 overs are in a meaningfully different position than teams with the same cumulative total but a slower recent scoring rate.
- **Three engineered features (`remaining_balls`, `remaining_wickets`, `match_phase`) contributed zero marginal SHAP value** once their signal was captured by other features, and were dropped without any loss in model performance.
- **Calibration matters as much as accuracy for a probability model**: XGBoost and Logistic Regression had nearly identical calibration in the mid-probability range, but XGBoost was clearly better calibrated at the high-confidence end (80-100%), the range that matters most in a live, nearly-decided chase.

## Methods & Techniques

### 1. Data Collection & Cleaning
- Parsed 10,458 match JSONs from [Cricsheet](https://cricsheet.org) into a single ball-by-ball dataframe (2.3M+ rows)
- Filtered to 2nd innings (chasing team) only, since that's the natural framing for a live win probability tracker
- Dropped ~2% of matches with no recorded winner (abandoned, no-result, or unrecorded Super Over decisions)

### 2. Feature Engineering
- Required runs, required run rate, current run rate, and their difference (the strongest predictor)
- Interaction term combining wickets in hand and balls remaining
- Rolling 3-over runs and wickets, capturing momentum rather than just cumulative totals
- Label encoding for venue, batting team, and bowling team

### 3. Temporal Train/Test Split
- Seasons 2005-2007 dropped (too little T20 data from cricket's early years)
- Train: 2008-2024, Test: 2025-2026, split by filtering on season directly rather than random shuffling, to prevent the model from being evaluated on the past after training on the future

### 4. Model Comparison
- Logistic Regression baseline (AUC 0.917, Log Loss 0.360, Brier Score 0.118)
- XGBoost (AUC 0.923, Log Loss 0.348, Brier Score 0.113), improving on every metric
- Isotonic calibration applied on top of XGBoost (Brier Score 0.112)

### 5. Explainability
- SHAP used to identify and drop three zero-importance features
- SHAP waterfall plots explain individual predictions in the deployed app

### 6. Case Study Validation
- Win probability curves generated for iconic matches (India vs South Africa T20 WC Final 2024, Mumbai Indians vs Chennai Super Kings IPL Final 2019) to sanity-check the model against real, dramatic match narratives
- Extended to 21 famous T20 matches in the deployed app

## Model Performance

| Model | AUC | Log Loss | Brier Score |
|---|---|---|---|
| Logistic Regression | 0.917 | 0.360 | 0.118 |
| XGBoost | 0.923 | 0.348 | 0.113 |
| XGBoost (Calibrated) | 0.924 | 0.345 | 0.112 |

**Final Model:** Calibrated XGBoost, selected for its superior log loss and high-confidence calibration, the metrics that matter most for a live probability tracker.

## Known Limitations

- Calibration was evaluated using the calibrator fit on the test set itself rather than a separate held-out validation set, so the calibration improvement is somewhat optimistic
- Venue defaults to a fixed value in the deployed app's Live Predictor, since asking users for exact venue names isn't practical
- The model performs poorly on extreme last-ball scenarios, which are rare edge cases where historical data is sparse

## Live App

The model is deployed as a 5-page Streamlit app:
- **Live Match Predictor**: enter current match state, get instant win probability
- **Full Match Replay**: upload any Cricsheet T20 JSON, watch win probability evolve over every over
- **Famous Matches**: relive 21 iconic T20 finals and thrillers through the model's eyes
- **Model Insights**: SHAP feature importance, calibration curves, live prediction explanations
- **Feature Engineering**: full pipeline documentation with design rationale

👉 [Live app](https://cricket-t20-win-probability.streamlit.app)

## Skills Demonstrated

- Python (pandas, NumPy, scikit-learn, XGBoost, SHAP, Matplotlib, Plotly)
- Temporal validation and leakage prevention
- Model comparison and calibration (isotonic regression, calibration curves)
- Feature engineering and SHAP-driven feature selection
- End-to-end deployment (Streamlit Community Cloud)
- Data storytelling through real match case studies

## View the Notebook

👉 [Open the notebook](https://github.com/GranBan/granthbangard_ds_portfolio/blob/main/Cricket%20T20%20Win%20Probability%20Predictor/t20_win_probability.ipynb)

## Future Improvements

- Three-way train/validation/test split for a methodologically cleaner calibration evaluation
- First innings score prediction, chaining into the current 2nd innings model
- Ablation study quantifying the exact contribution of calibration and rolling features
- Error analysis identifying systematic patterns in the model's biggest misses
