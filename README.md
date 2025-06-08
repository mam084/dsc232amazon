# Amazon Review Sentiment as a Predictor of Economic Trends

## Repository Contents

This repository includes:
- Jupyter notebooks for all preprocessing, modeling, and analysis steps
- Final Report in PDF format
- Prior milestone submissions
- This README summarizing the full project workflow and files

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

## Notebook

Final model and data download can be found in the linked Jupyter Notebooks:

**[Amazon Reviews GDP Analysis](./Amazon_reviews_GDP_analysis.ipynb)** || 
**[Data Download](./Data_download.ipynb)**

## Final Report

The pdf version of the written report is available here:  
[**Amazon Review Sentiment as a Predictor of Economic Trends (PDF)**](./Amazon_Review_Sentiment_as_a_Predictor_of_Economic_Trends.pdf)

## Introduction

This project investigates whether consumer sentiment expressed in Amazon product reviews can serve as an early indicator of broader economic trends, specifically annual changes in U.S. GDP per capita. By applying natural language processing (NLP), sentiment analysis, and machine learning to millions of reviews, we explore whether aggregated consumer sentiment in product categories like groceries, baby products, toys, and media can reveal patterns aligned with economic expansions or contractions.

Our hypothesis is that online reviews, written voluntarily and frequently, can provide a more immediate reflection of consumer confidence than traditional economic statistics. If successful, this method could offer a scalable, crowd-sourced tool for real-time economic forecasting.

## Methods

This study investigates whether large-scale sentiment data from Amazon product reviews can serve as an early signal of macroeconomic changes, specifically U.S. GDP per capita growth. Our modeling process followed a structured pipeline of data preprocessing, feature extraction, aggregation, transformation, and supervised learning using logistic regression. All modeling was conducted with attention to temporal structure, using chronological splits to prevent leakage.

### Data Sources and Preprocessing

We utilized two main datasets. The first consisted of millions of Amazon reviews, filtered to focus on four product categories—two “essential” (Groceries and Baby Products) and two “luxury” (Toys & Games and Movies & TV). The second was a historical GDP per capita dataset for the United States, spanning 1996 to 2023.

Within the Amazon dataset, reviews rated 3 stars were excluded to eliminate sentiment ambiguity. The remaining reviews (1, 2, 4, or 5 stars) were retained along with their associated product category, timestamp, and full review text. This allowed for sentiment extraction and text-based feature engineering. The GDP dataset was cleaned and processed to generate binary outcome labels: 1 if the GDP per capita increased compared to the previous year, and 0 otherwise. These labels served as the classification target.

### Text Processing and Feature Extraction

We implemented a PySpark NLP pipeline to transform review text into numeric representations. Reviews were tokenized, stop words were removed, and the resulting tokens were mapped into high-dimensional hashed term frequency vectors using the HashingTF function with 10,000 buckets. These frequency vectors were then scaled using inverse document frequency (IDF) weighting to emphasize rare but potentially informative terms.

In parallel, we performed sentiment scoring using TextBlob. Each review received a polarity score between –1 and +1, indicating its sentiment intensity. The polarity score was later combined with the TF-IDF representation to form a unified feature vector for each review.

### Aggregation and Feature Engineering

To align text and sentiment features with economic indicators, we aggregated data at the year and product category level. For each combination, we computed:

- Mean sentiment polarity
- Sentiment volatility (standard deviation)
- Total review count
- Mean TF-IDF vector

We also calculated the sentiment difference between essential and luxury categories, allowing the model to learn contrasts in consumer confidence across economic necessity. To reduce dimensionality, we applied Principal Component Analysis (PCA) separately to each product group’s TF-IDF vectors, reducing them to 50 components each.

Finally, we engineered year-over-year delta features that captured directional sentiment changes:

- essential_sentiment_yoy_change
- luxury_sentiment_yoy_change
- sentiment_gap_yoy_change

After merging all features by year, rows missing valid YOY differences were removed, resulting in 27 usable years (1997–2023).

### Model Setup and Evaluation

We used a logistic regression model with L1 regularization, trained on 23 years of data (1997–2018) and tested on 5 future years (2019–2023). Cross-validation was avoided due to temporal leakage. The final input vector combined the three YOY sentiment features with the 100 PCA components. All features were standardized.

Performance was evaluated using AUC and accuracy.

- **Train Accuracy**: 1.000
- **Train AUC**: 1.000
- **Test Accuracy**: 0.800
- **Test AUC**: 0.833

The model successfully predicted major economic shifts, including the 2020 downturn, using only review-derived features.

## Results

Exploratory analysis revealed several trends. Amazon reviews showed a polarized rating distribution, with 3-star reviews being rare. Volume of reviews grew significantly after 2012. Luxury goods exhibited higher sentiment volatility than essentials, especially during disruptive periods such as 2020.

The model’s strong test performance using a chronological split affirmed the value of preserving temporal integrity. In contrast, cross-validation with random splits performed poorly (AUC ~0.58), highlighting the risk of time leakage in economic forecasting contexts.

Despite perfect training performance, the model maintained solid generalization on unseen data. This suggests that review-based sentiment, especially in year-over-year form, can capture meaningful economic signals.

## Discussion

This study demonstrates that consumer sentiment expressed in Amazon product reviews, when properly processed and temporally aligned, can reveal meaningful macroeconomic signals. Our final logistic regression model, trained on year-over-year sentiment changes and PCA-reduced textual features, was able to predict shifts in U.S. GDP per capita with strong accuracy and AUC on future test years. The model’s successful anticipation of the 2020 economic downturn provides compelling support for the broader hypothesis that spontaneous consumer expression has predictive economic value.

However, the model’s perfect training performance (accuracy and AUC of 1.000) raises concerns about overfitting. Although the model remains relatively simple and generalizes well to unseen years—with a test AUC of 0.833—the 0.167 drop from training to testing performance suggests that some degree of overfitting is present. This reinforces the importance of using strictly chronological splits in time series contexts, as initial cross-validation experiments yielded misleadingly poor results due to temporal leakage.

While the current approach shows promise, it is constrained by several limitations. First, the dataset spans only 27 annual observations, which, while rich in review volume, remains sparse as a sample for supervised learning. Second, only four product categories were included in the analysis, limiting the diversity of sentiment signals captured. Third, the use of annual aggregation restricts the model’s ability to detect short-term economic fluctuations.

## Conclusion

Despite these limitations, our findings suggest that large-scale sentiment data from consumer reviews can serve as a low-cost, high-frequency complement to traditional economic indicators. With further refinement—such as expanding to quarterly aggregation, integrating additional metadata (e.g., verified purchase tags or helpful vote counts), or combining sentiment-based predictors with standard economic variables in an ensemble framework—sentiment-driven forecasting could evolve into a valuable real-time tool for economic analysis and policy.

This project demonstrates the feasibility of using consumer language as a macroeconomic input and opens the door to future research on scalable, crowd-sourced economic indicators.

## Contributions

-**Ann Nguyen**: 
-**Artien Voskanian**:  
-**Joanna Tam**:  
-**Matthew Mitchell**: GitHub management (hosting the repo, uploading files, and merging branches), README writing (original Module 2 and 3 submissions as well as the final submission), and Final Report Writing (namely Introduction, Method, and Results sections)


## Prior Submissions

- **[Amazon_reviews_exploration.ipynb](./Amazon_reviews_exploration.ipynb)** – Initial data exploration and preprocessing pipeline
- **[Milestone_3.ipynb](./Milestone_3.ipynb)** – First model development
