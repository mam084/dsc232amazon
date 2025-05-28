# Amazon Reviews Sentiment Analysis and Economic Forecasting

## Project Goal

This project investigates whether consumer sentiment in Amazon product reviews can serve as an early indicator of broader economic trends. By analyzing millions of reviews using natural language processing (NLP), we aim to uncover patterns in tone, emotional intensity, and volume—especially in price-sensitive product categories—and evaluate their correlation with economic indicators such as inflation and consumer confidence.

## Environment Setup

We used SDSU Expanse to run this project. Sbatch was not used to submit jobs, instead we set up Jupyter notebook session with these parameters for data exploration:
- Partition: shared
- Time limit: 90 mins
- Number of cores: 9
- Memory required per node (GB): 184
- Singularity image file location: ~/esolares/spark_py_latest_jupyter_dsc232r.sif
- Environment module: singularitypro
- Working directory: Home

## Data download link: 

https://amazon-reviews-2023.github.io/

### Data sets used:
- Luxury items
- Baby Products
- Essential Products


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

### Plot 1: Rating vs Helpful Votes

- This scatter plot maps the number of helpful votes against star ratings. Most reviews with high helpfulness cluster around the 1, 4, or 5 star rating. Reviews with a massive amount of helpful ratings occur at all levels, but are extremely rare in all cases. The vast majority of reviews have fewer than 100 helpful votes. While high star ratings receive vast amounts of helpful ratings, negative reviews seem to also gather a large amount of these votes.

### Plot 2: Comparison of Rating Distributions by Product Category

- Stacked bar charts that have the year as the x-axis and amount of votes as the y-axis (seperated by rating level). A chart for essential products, luxury products, and all products helps give insight into any distribution trends. Most ratings are 4/5 stars across all product types. Rating volume grows significantly over time, peaking around 2014-2016. Luxury PRoducts have a sharper rise and fall in reviews compared to essentials, a potentially crucial insight in the context of this project. Overall, product reviews have become more frequent over time and consumers tend to give high ratings regardless of product type. The volatility in the luxury products category, however, is a notable trend.

### Plot 3: Yearly Average Ratings by Product Category

- The plot on the left represents the average rating of essential products over time (years). The middle plot accomplishes the same goal with luxury items. Finally, the plot on the right showcases the average rating across all product categories. It is immediately apparent that the plots of luxury items and all products share a significant resembelence to one another (with luxury items perhaps scoring slightly higher during certain years). The average ratings for essential products are significantly lower from 2000 - 2005, but then resembles the other charts starting around 2015. In all 3 cases, a noticeable dip and down trend can be observered at 2020, which may be attributed to the COVID-19 pandemic.

### Plot 4: Comparison of Average Ratings & Review Growth Over Time

- In the left chart, the previously described line charts of average ratings are stacked together for direct comparison. A convergance seems to occur after 2010, with all categories averaging between 4.0 - 4.3. For the right chart, we can clearly seen explosive growth in the number of reviews from 2000 - 2015. The trends still point to more growth, but has relatively plateaued compared to the previous decade. We can see that average consumer sentiment is consistent across product types in recent years, and the boom in online shopping and review activity is clearly visible post 2010.

## Notebook

Data exploration and data download can be found in the linked Jupyter Notebooks:

**[Amazon Reviews Exploration Notebook](./Amazon_rewiews_exploration.ipynb)**
**[GDP Data Exploration](./GDP_Explore.ipynb)**
**[Data Download](./Data_download.ipynb)**
