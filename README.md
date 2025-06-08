# Amazon Review Sentiment as a Predictor of Economic Trends

## Project Overview

This project investigates whether consumer sentiment expressed in Amazon product reviews can serve as an early indicator of broader economic trends, specifically annual changes in U.S. GDP per capita. By applying natural language processing (NLP), sentiment analysis, and machine learning to millions of reviews, we explore whether aggregated consumer sentiment in product categories like groceries, baby products, toys, and media can reveal patterns aligned with economic expansions or contractions.

The final model uses review-level features—such as polarity scores and TF-IDF components—aggregated annually and processed through a logistic regression classifier. All modeling is done using a time-respecting chronological split to reflect realistic forecasting conditions.

## Repository Contents

This repository includes:

- Final Report in PDF format
- Jupyter notebooks for all preprocessing, modeling, and analysis steps
- Prior milestone submissions
- This README summarizing the full project workflow and files

## Final Report

The complete written report is available here:  
[**Amazon Review Sentiment as a Predictor of Economic Trends (PDF)**](./Amazon_view_Sentiment_as_a_Predictor_of_Economic_Trends.pdf)

The report includes:
- Introduction and hypothesis
- Data pipeline and modeling methodology
- Exploratory analysis and results
- Discussion of limitations, overfitting, and evaluation techniques
- Future directions for improvement and scale

## Environment Setup

We used SDSU Expanse to run this project. Sbatch was not used to submit jobs, instead we set up Jupyter notebook session with these parameters for data exploration:
- Partition: shared
- Time limit: 90 mins
- Number of cores: 9
- Memory required per node (GB): 184
- Singularity image file location: ~/esolares/spark_py_latest_jupyter_dsc232r.sif
- Environment module: singularitypro
- Working directory: Home

Used packages and libraries:
- os, pickle, glob, pyspark, pyspark.sql, numpy, matplotlib, seaborn, json

## Dataset and Source

We use two primary datasets:

1. **Amazon Product Reviews**: Filtered by 4 product categories: Groceries, Baby Products, Toys & Games, Movies & TV.  
   Source: [https://s3.amazonaws.com/amazon-reviews-pds/tsv/index.txt](https://s3.amazonaws.com/amazon-reviews-pds/tsv/index.txt)

2. **U.S. GDP per Capita**: Yearly data from 1996 to 2023.  
   Source: World Bank and supplemental public datasets

## Method Summary

- **Preprocessing**: Filtered reviews (excluding neutral 3-star), TextBlob sentiment scoring, TF-IDF vectorization, and PCA compression
- **Features**: Year-over-year sentiment deltas, sentiment volatility, and PCA-reduced TF-IDF components
- **Label**: Binary indicator (1 = GDP per capita increase, 0 = decrease or no change)
- **Model**: Logistic regression with L1 regularization (elasticNetParam = 1.0, regParam = 0.01)
- **Train/Test Split**: Chronological (Train: 1997–2018, Test: 2019–2023)

## Results Summary

- **Training Accuracy**: 1.000
- **Training AUC**: 1.000
- **Test Accuracy**: 0.800
- **Test AUC**: 0.833

Randomized cross-validation performed poorly (AUC ~0.58), highlighting the importance of temporal structure. The model correctly predicted key years like the 2020 downturn based on sentiment-only features.

## Limitations and Future Work

- Only 27 years of data—sufficient in volume, but sparse in samples
- Analysis limited to 4 product categories
- Future work may include:
  - Expanding to quarterly sentiment
  - Incorporating verified purchase data or metadata
  - Ensemble models combining traditional economic indicators

## Prior Submissions

- `milestone_2_submission.ipynb` – Initial data exploration and preprocessing pipeline
- `milestone_3_submission.ipynb` – Model development and cross-validation testing
- `Amazon_rewiews_GDP_analysis.ipynb` – Final model using year-over-year sentiment features and PCA components
