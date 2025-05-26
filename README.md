# Amazon Reviews Sentiment Analysis and Economic Forecasting

## Project Goal

This project investigates whether consumer sentiment in Amazon product reviews can serve as an early indicator of broader economic trends. By analyzing millions of reviews using natural language processing (NLP), we aim to uncover patterns in tone, emotional intensity, and volume—especially in price-sensitive product categories—and evaluate their correlation with economic indicators such as inflation and consumer confidence.

## Data Exploration Summary

We began our analysis by evaluating the structure and properties of the review dataset:

- Number of Observations: Millions of product reviews spanning multiple years and categories.
- Columns: Key columns include:
  - asin: Product ID
  - reviewText: Main review text
  - overall: Star rating (1–5 scale)
  - helpful: Votes indicating usefulness of a review
  - reviewTime: Date of review
- Distributions & Scales:
  - overall ratings are discrete and range from 1 to 5.
  - reviewText length varies widely and will be normalized during preprocessing.
  - Helpful votes are highly skewed with a long tail.
- Missing Data: Some reviews may be missing helpful vote info or may contain empty reviewText. These will be filtered out during preprocessing.
- Preliminary Visualizations:
  - Histograms of ratings and helpful votes
  - Token frequency counts
  - Word clouds for different star ratings

## Preprocessing Plan

Our preprocessing pipeline includes:

1. Filtering: Remove reviews with missing or empty text, and non-English entries if detected.
2. Text Cleaning: Lowercasing, punctuation removal, stopword filtering, lemmatization.
3. Tokenization & Vectorization: Using Word2Vec on Spark to create semantic representations of review text.
4. Time Grouping: Aggregating sentiment scores by month/year for temporal analysis.
5. Category Selection: Focusing on key product categories that reflect essential or price-sensitive spending (e.g., groceries, electronics).
6. Sentiment Scoring: Using NLP sentiment models to quantify emotional tone for each review.

## Notebook

Data exploration and data download can be found in the linked Jupyter Notebooks:

**[Amazon Reviews Exploration Notebook](./Amazon_rewiews_exploration.ipynb)**
**[GDP Data Exploration](./GDP_Explore.ipynb)**
**[Data Download](./Data_download.ipynb)**


---

## Updated Preprocessing Summary

We finalized our major preprocessing pipeline with the following steps:

- **Text Cleaning**: Lowercasing, punctuation removal, stopword filtering, lemmatization using NLTK.
- **Sentiment Scoring**: Applied the VADER sentiment model to compute a compound sentiment score for each cleaned review.
- **Timestamp Normalization**: Converted UNIX timestamps to year-based format for alignment with GDP data.
- **Aggregation**: Aggregated sentiment scores and review volume per year.
- **Feature Engineering**:
  - Created two features per year: average sentiment and total review volume.
  - No categorical features to encode at this stage.
- **Missing Data Handling**: Removed rows with missing sentiment or timestamp.
- **Scaling/Transformation**: Initial model used raw numeric features; later stages may consider normalization or log transforms for skewed counts.

---

## First Model: 

**Train/Test Split**: 
**Evaluation Metrics**:


---

## Model Interpretation

- **Fitting Status**: Underfitting  

---

## Conclusion

Our initial model showed that:
- Our initial results indicate that the current model is promising, but there remains headroom for improvement. 
- We will attempt an ensemble framework by experimenting with logistic regression variants and gradient
boosted libraries such as XGBoost and LightGBM. Each will be evaluated within a validation pipeline to
ensure fair comparisons of accuracy and F1 score.
- If the basic weak learner ensemble shows potential, we’ll layer a weighted voting system to further
improve performance.
- At the same time, we’ll explore additional feature engineering ideas. On the text side, we’ll benchmark
and optimize different tokenization strategies, such as word level, character n grams, and contextual
embeddings, while also streamlining the processing pipeline for speed and processing efficiency.
- Throughout all iterations, we will monitor not only improvements in predictive metrics but also
overfitting/underfitting and model stability.
- By iterating on both algorithmic and feature engineering fronts, we are confident we can fully
realize and exceed our model’s performance goals.


**Next Steps**:
- Moving forward, our plan is for each team member to independently develop a distinct 
predictive model using the same training data. 
- By leveraging the natural diversity in modeling approaches, whether through different 
feature selections or hyperparameter settings, we aim to assemble these varied learners 
into a weak learner ensemble. 
- Once the baseline ensemble is in place, we will rigorously evaluate its performance 
using cross‐validation and hold out test sets, tracking metrics such as accuracy, 
precision, recall, and AUC. 
- If the unweighted ensemble demonstrates a meaningful uplift in generalization compared 
to any single model, our next step will be to introduce a weighted‐voting scheme. 
- In this setup, models that consistently contribute stronger performance on validation 
folds will receive proportionally higher voting power, further refining our aggregate 
prediction and potentially smoothing out remaining errors.
- Should this staged ensemble strategy fail to deliver the accuracy gains we’re 
targeting, or if we uncover scalability or maintenance challenges, we will instead 
utilize XGBoost or LightGBM.  
- We’ll benchmark them under identical evaluation protocols, comparing not only 
predictive metrics but also training time, resource usage, and interpretability 
considerations, to determine the most effective path forward.

