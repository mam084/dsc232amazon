# Amazon Reviews Sentiment Analysis and Economic Forecasting

## Project Goal

This project investigates whether consumer sentiment in Amazon product reviews can serve as an early indicator of broader economic trends. By analyzing millions of reviews using natural language processing (NLP), we aim to uncover patterns in tone, emotional intensity, and volume—especially in price-sensitive product categories—and evaluate their correlation with economic indicators such as inflation and consumer confidence.

## Data Exploration Summary

We began our analysis by evaluating the structure and properties of the review dataset:

- Number of Observations: Millions of product reviews spanning multiple years and categories.
- Columns: Key columns include:
  - asin: Stores the unique product ID (Amazon Standard Identification Number) for each item. Used to group and identify reviews for the same product.
  - reviewText: Contains the full text of each customer’s review. This column will be the primary input for running sentiment analysis and other NLP tasks such as keyword extraction or topic modeling.
  - overall: Reflects the star rating (on a 1–5 scale) that a user gives a product. This serves as the ground truth label for supervised learning models and allows us to compare predicted sentiment with actual user ratings.
  - helpful: Indicates how many users found a review helpful, typically formatted as a list. This can be used to filter low-quality reviews or to weight reviews by perceived usefulness.
  - reviewTime: The date when the review was submitted. Useful for analyzing trends over time, identifying seasonal patterns, or detecting anomalies in review activity.
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

## Plot Explanations

### Plot 1: Helpful Rating

### Plot 2: Comparison of Rating Distributions by Product Category

- This graph gives

## Notebook

Data exploration and data download can be found in the linked Jupyter Notebooks:

**[Amazon Reviews Exploration Notebook](./Amazon_rewiews_exploration.ipynb)**
**[GDP Data Exploration](./GDP_Explore.ipynb)**
**[Data Download](./Data_download.ipynb)**
