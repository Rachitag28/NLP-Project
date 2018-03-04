# NLP-Project
## NLP-Giants - Stance analysis project at University of Cincinnati

### SemEval 2016 Task-6 was performed as a term project for NAtural Language Processing class

## Task Details

#### Aim of the task was to analyse the dataset and determine the correct stance for each tweet with respect to a particular topic
#### The task was divided into two sub task:
#### 1. Supervised Learning: In this sub task, 5 topics and tweets related to those topics were provided. The training data consisted of the labels associated with each tweet. Stance was to be analyzed for the testing data without the labels using supervised learning.
#### 2. Unsupervised/ Semi-Supervised learning: In this subtask only one specific topic was given and a corpus of tweets associcated with that topic was provided. Tweets in the data had no inital labels. Using unsupervised or semi-supervised learning stance for each tweet was to be determined.

## Pre-Processing
#### In the pre-processing phase the data was filtered and cleaned to remove all the noisy data and some elements from the tweets which were unncessary 
#### Following steps were performed in this phase:-
#### 1. Lowercasing for all the tweets were doen
#### 2. Hashtags were removed and stored separately
#### 3. Unnecessary symbols, emoticons and special characters were removed.
#### 4. Hyper links were removed from the data
#### 5. Slangs and abbreviations were transformed to the original english translation

## Feature Selection
#### Features were extracted from the data set to feed them into machine learning algortihms for predicting the stance of each tweeet
#### Following is the feature list:
#### 1. Bi-grams (using TF-IDF vectorization)
#### 2. Noun Phrases (using POS Tagging)
#### 3. Hashtags extracted from each tweet during pre-processing
#### 4. Sentiment Scores (calculated using AFINN dictionary)

## Classification
#### Sub-Task A: For supervised learning, Naive Bayes Classification was implemented. Using the feature set defined above prediction for the correct stance were provided as result.
#### Sub-Task B: For unsupervised learning, K-means Clustering was used to cluster the tweets into 3 categories Against, For and Neutral. Based on these clusters final prediction for stance were provided.
