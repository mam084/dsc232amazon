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
