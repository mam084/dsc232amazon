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

Used packages and libraries:
- os, pickle, glob, pyspark, pyspark.sql, numpy, matplotlib, seaborn, json

## Data download link: 

https://amazon-reviews-2023.github.io/

### Data sets used:
- Luxury items: Movies and TV, Toys and Games
- Essential Products: Baby Products, Grocery and Gourmet Food

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

**[First Model Notebook](./Milestone_3.ipynb)** || 
**[Amazon Reviews Exploration Notebook](./Amazon_rewiews_exploration.ipynb)** || 
**[GDP Data Exploration](./GDP_Explore.ipynb)** || 
**[Data Download](./Data_download.ipynb)**


---

## Updated Preprocessing Summary:

Our refined preprocessing pipeline consists of several steps aimed at improving text quality,
ensuring consistency, and preparing features for modeling:

**File Filtering**: From a large directory of Amazon .jsonl review files, we selected reviews from 
specific product categories-Grocery and Baby (labeled "essential") and Movies and Video Games 
(labeled "luxury")—to facilitate socioeconomic segmentation.

**Initial Cleaning**: Removed reviews that were blank, image-only, or contained no meaningful text 
(e.g., strings like "image only", "just a picture", or numbers-only placeholders for images).
Applied filters to exclude reviews with null or placeholder content in the text field, 
particularly those with accompanying images but no review content.

**Feature Augmentation**: Computed a text_len feature for each review by measuring the character 
length of the review text. This feature helps distinguish short or low-effort reviews from detailed 
ones and may correlate with sentiment strength or helpfulness.

**Missing Data Handling**: Systematically identified and dropped records with null values in key 
fields such as review text, sentiment score, or rating.

**Sampling for Analysi**s: For downstream modeling and visualization tasks, a 2% sample of the 
cleaned merged dataset was used to efficiently explore relationships between review helpfulness, 
rating, and text-based features.

---

## First Model: 

We developed our initial model using a comprehensive text preprocessing pipeline implemented through Spark ML. 
The preprocessing workflow included tokenization, removal of common stop words, and feature extraction using HashingTF 
combined with Inverse Document Frequency (IDF) weighting to create meaningful numerical representations of the text data.

For sentiment labeling, we leveraged TextBlob's sentiment polarity scores, which range from -1 (negative) to +1 (positive).
To create a robust binary classification problem, we filtered out reviews with neutral sentiment (polarity scores near zero), 
focusing our model on clearly positive and negative reviews that would provide stronger training signals.

We implemented logistic regression as our baseline classifier and conducted systematic hyperparameter tuning on the validation
set to identify optimal model configurations. After selecting the best-performing parameters based on validation metrics, 
we evaluated the final model's performance on our held-out test set to assess its generalization capability.

**Train/Test Split**: We split the dataset into training and test sets using an 60/20/20 ratio. Reviews were randomly sampled after 
filtering to ensure balanced positive and negative examples.
**Evaluation Metrics**:
- Training Accuracy (C = 0.1): 0.8912
- Validation AUC: 0.9400
- Validation Accuracy: 0.8909
- Test Accuracy: 0.8910
- Test AUC: 0.9424

---

## Model Interpretation

**Fitting Status**: These results suggest that the model performed well at distinguishing between positive and negative
sentiment. However, the small difference between training and test performance may indicate mild overfitting, 
and further regularization or more diverse training data could be explored to improve generalization.

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

