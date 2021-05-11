# Fake News Sentiment Analysis

This project has a number of dependencies, defined in `requirements.txt`. These can be fetched by entering `pip install -r requirements.txt` into a terminal in the root directory. 

## Introduction

This project acts as a pipeline for fetching and analysing Tweets discussing fake and real articles. Many of the scripts will take a long time to run. Because of this, this project makes extensive use of the `pickle` and `json` modules in order to save progress after each step. The steps, including a suggested order, are outlined below in this README.

## 1. extract_tweets.py

This script is responsible for extracting the tweets that have been fetched by **FakeNewsNet**. It will apply some light pre-processing, namely removing any exact duplicate tweets before outputting to two pickle files: `fake.pkl` and `real.pkl`. Note that the **FakeNewsNet** folder has been omitted from this repository. As it stored each tweet as a separate `.json` file, Git was not capable of handling this volume of files. The `fake.pkl` and `real.pkl` files have been included to allow us to circumvent this limitation.

## 2. extract_training_data.py

This script is responsible for extracting the training tweets from the **Sentiment140** dataset. Once again, it applies preprocessing before outputting the tweets to two files: `training_positive.json` and `training_negative.json`. 

## 3. filter.py

This script applies more pre-processing to the **FakeNewsNet** dataset. Namely, it is responsible for removing tweets that do not match certain criteria. Tweets that are not wholly ASCII text will be removed. This is to help avoid non-english tweets, something that is out of the scope of this project.

It will also remove article titles from tweets. Tweets that are of very low length after having had the title removed will be excluded from further analysis. 

This script outputs to `fake_cleaned.pkl` and `real_cleaned.pkl`. 

## 4. train.py

This script will train the three classifiers that this project uses: Naive Bayes, Support Vector Machine and Logistic Regression. They are trained on the **Sentiment140** dataset we extracted in `extract_training_data.py`. 

The models are exported as pickle files to the `classifiers` directory.

## 5. rule_based_classifier.py

This file provides a bespoke Rule Based Classifier. It uses the **SentiWords** dataset to assign scores to words according to their usage in a sentence. It then applies a multiplier depending on punctuation, capitalisation and the usage of emoticons. As this model is rule based, it does not require training. 

## 6. classify.py

This module uses the above statistical classifiers together with the rule-based classifier to produce sentiment labels for each tweet in our **FakeNewsNet** dataset. It exports these classifications to `classifications_fake.json` and `classifications_real.json`.

## 7. app.py

This python script produces a **Dash** dashboard for data exploration. It allows us to see the distribution of tweets and their positivity polarity, depending on which model is selected. It also allows us to classify new text and view outputs from each classifier. This can be used to provide a useful intuition to the strengths and weaknesses of each classifier.

## 8. utils.py

This file provides utilities that are re-used throughout the project including some pre-processing methods.

## label.py

This unused file allows us to manually classify tweets from the **FakeNewsNet** dataset. It was created as a potential way to address early limitations with the training data that was available to the models. However, manually tagging tweets proved impractical, especially without context on the news article being discussed. 