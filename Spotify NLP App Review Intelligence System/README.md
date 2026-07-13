# Spotify NLP App Review Intelligence System

This project builds an end-to-end NLP pipeline that transforms 100,000 raw Google Play reviews into a prioritized product decision system for Spotify. Rather than stopping at sentiment classification, the pipeline fine-tunes a transformer for text-based sentiment, discovers complaint themes through topic modeling, tracks sentiment across app versions to catch regressions, and scores every complaint by frequency, severity, and recent trend into a ranked, actionable fix list. Deployed as a 7-page interactive Streamlit app.

## Project Objectives

- Classify review sentiment from raw text, not just star ratings, to catch nuance ratings miss
- Discover recurring complaint themes automatically using embedding-based topic modeling
- Track sentiment across app versions to identify releases that caused regressions
- Translate NLP output into a business-actionable priority matrix, not just a list of topics
- Deploy a working, interactive product that a product team could plausibly use

## Key Insights

- **Versions 9.1.46 and 9.1.48 show the highest negative sentiment rates in the dataset (77% and 75%)**, both dominated by App Crashes and Bugs complaints, suggesting a regression introduced in 9.1.46 that persisted into the next release.
- **Star ratings alone miss real sentiment**: many 5-star reviews contain clearly negative text (e.g. "great app but too many ads"), which is exactly why a text-based model adds value over rating-based filtering. 3,554 reviews showed this exact pattern.
- **Binary classification outperformed 3-class by a wide margin**: initial Negative/Neutral/Positive labeling only reached 71% macro F1, limited by the small, inherently ambiguous neutral class. Switching to binary (Negative/Positive) and using the full available data raised macro F1 to 93.5%.
- **Ads and Premium Upsell is the single largest complaint category** (3,753 reviews), consistently present across nearly every app version tracked.

## Methods & Techniques

### 1. Data Collection & Cleaning
- Scraped 100,000 Spotify reviews directly from the Google Play Store using `google-play-scraper`, preserving app version metadata for later temporal analysis
- Dropped columns with no analytical value and the small number of reviews with null text

### 2. Sentiment Classification
- Initial star-rating-derived labels used as a starting point, then replaced by a fine-tuned DistilBERT model for text-based sentiment
- Fine-tuning done on Google Colab (T4 GPU), binary classification (Negative/Positive), 56,510 balanced training reviews
- Achieved 93.5% macro F1 on held-out validation data
- Full inference run on all 100,000 reviews, not just the training sample

### 3. Topic Modeling
- BERTopic with `all-MiniLM-L6-v2` sentence embeddings, chosen over bag-of-words methods so semantically similar complaints cluster together even without shared vocabulary
- Applied only to predicted-negative reviews with 10+ words (22,630 after English filtering), since short reviews don't carry enough signal for meaningful clustering
- 51 distinct topics discovered, each manually reviewed and labeled into business-readable themes

### 4. Temporal Trend Analysis
- Negative sentiment rate tracked across all app versions with 100+ reviews
- Version-level drill-down cross-references sentiment spikes with dominant complaint topics for that release

### 5. Priority Matrix
- Each complaint topic scored by frequency (50%), severity (30%, based on average star rating), and recent trend (20%, share of the 5 most recent versions)
- Produces a ranked list with a recommended action for each top issue, translating NLP output directly into product decisions

## Model Performance

| Metric | Value |
|---|---|
| Macro F1 | 0.935 |
| Precision (Negative) | 0.93 |
| Precision (Positive) | 0.94 |
| Recall (Negative) | 0.94 |
| Recall (Positive) | 0.93 |
| Accuracy | 0.94 |
| Average Prediction Confidence | 96.2% |

**Final Model:** Binary DistilBERT classifier, chosen over the initial 3-class setup since neutral predictions were never used downstream and binary classification allowed using the full available training data.

## Known Limitations

- 35% of negative reviews fall into a noise cluster and aren't represented in the priority matrix, these are reviews too short or ambiguous to cluster meaningfully even after filtering
- 16.4% of reviews lack app version metadata and are excluded from temporal analysis
- The in-app review search feature operates on an 820-review sample, not the full corpus, for deployment memory efficiency

## Live App

The pipeline is deployed as a 7-page Streamlit app:
- **Overview**: sentiment distribution, review volume trends, dataset summary
- **Complaint Intelligence**: BERTopic clusters with representative reviews and keyword search
- **Version Trends**: negative sentiment rate tracked across app versions with spike detection
- **Priority Matrix**: ranked, actionable fix list with transparent scoring methodology
- **Misclassification Analysis**: where model predictions disagree with star ratings, and why
- **Pipeline Overview**: full pipeline documentation with design rationale
- **Embedding Visualization**: 2D UMAP projection showing how BERTopic separates complaint themes

👉 [Live app](https://spotify-review-intelligence.streamlit.app)

## Skills Demonstrated

- Python (pandas, NumPy, transformers, BERTopic, sentence-transformers, scikit-learn)
- Transformer fine-tuning (DistilBERT, binary classification, Google Colab GPU training)
- Embedding-based topic modeling (BERTopic, UMAP, sentence embeddings)
- Temporal trend analysis and regression detection
- Business-oriented scoring systems (custom priority matrix formula)
- End-to-end deployment (Streamlit Community Cloud, memory-optimized data pipelines)

## View the Notebook

👉 [Open the notebook](https://github.com/GranBan/granthbangard_ds_portfolio/blob/main/Spotify%20NLP%20App%20Review%20Intelligence%20System/nlp_app_review.ipynb)

## Future Improvements

- Topic coherence and diversity metrics to formally validate BERTopic cluster quality
- Automated weekly re-scraping and inference pipeline to keep sentiment tracking current
- Aspect-based sentiment analysis to separate multiple complaints within a single review
- Cold-start handling for detecting entirely new complaint categories as they emerge
