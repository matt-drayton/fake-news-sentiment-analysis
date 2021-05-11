import pickle
from tqdm import tqdm
import json
from nltk.tokenize.casual import casual_tokenize
import re, string, random
from utils import lemmatize_and_strip, log
from rule_based_classifier import RuleBasedClassifier


def load_models():
    """Loads models from their pickle files

    Returns:
        Tuple: Tuple contianing the naive bayes, svc, and logistic regression models that have been
               previously trained
    """
    log("Loading models")
    with open("./classifiers/naive.pkl", "rb") as file:
        naivebayes = pickle.load(file)
    with open("./classifiers/svc.pkl", "rb") as file:
        svc = pickle.load(file)
    with open("./classifiers/logistic.pkl", "rb") as file:
        logistic = pickle.load(file)

    return (naivebayes, svc, logistic)


def get_tweets_for_model(tokens):
    """Generator to load tweets in format for model. Mark each feature as 'true' for classifier

    Args:
        cleaned_tokens_list: List of cleaned tweets to be fed to model

    """
    for tweet_tokens in tokens:
        yield dict([token, True] for token in tweet_tokens)


def _classify_tweets(tweets, raw_tweets, article_sources, classifier):
    """Private helper method to classify tweets using a determined classifier

    Args:
        tweets: List of cleaned tweets
        raw_tweets: Corresponding list of un-prepared tweets
        article_sources: Corresponding list of article URLs
        classifier: Object of model to be used. Must have a 'classify' and 'score' method

    Returns:
        A list of classified tweets
    """
    total_output = []
    for tweet, raw_tweet, article_source in tqdm(
        zip(tweets, raw_tweets, article_sources)
    ):
        joined_tweet = " ".join(tweet.keys())
        if type(classifier) is RuleBasedClassifier:
            positive_score = classifier.score(raw_tweet)
            classification = classifier.classify(raw_tweet)
        else:
            probs = classifier.prob_classify(tweet)
            positive_score = probs.prob("Positive")
            classification = classifier.classify(tweet)
        output = {
            "tweet": joined_tweet,
            "raw_tweet": raw_tweet,
            "article_title": article_source,
            "classification": classification,
            "positive_score": positive_score,
        }
        total_output.append(output)
    return total_output


def run_classifiers(fake_tweets, real_tweets, classifiers):
    """Runs sentiment analysis using a list of classifiers

    Args:
        fake_tweets: A list of fake tweet JSONs
        real_tweets: A list of true tweet JSONs
        classifiers: A list of classifiers to use with the tweet data
    """
    log("Extracting tweets from pickle")
    fake_tweets_unmodified = [tweet["tweet"] for tweet in fake_tweets]
    fake_article_sources = [tweet["article_title"] for tweet in fake_tweets]
    fake_tweets = [casual_tokenize(tweet["tweet"]) for tweet in tqdm(fake_tweets)]

    real_tweets_unmodified = [tweet["tweet"] for tweet in real_tweets]
    real_article_sources = [tweet["article_title"] for tweet in real_tweets]
    real_tweets = [casual_tokenize(tweet["tweet"]) for tweet in tqdm(real_tweets)]

    log("Removing noise")
    fake_tweets = [lemmatize_and_strip(tweet) for tweet in tqdm(fake_tweets)]

    real_tweets = [lemmatize_and_strip(tweet) for tweet in tqdm(real_tweets)]

    log("Beginning classifications")
    for name, classifier in classifiers.items():
        fake_tweets_generator = get_tweets_for_model(fake_tweets)
        real_tweets_generator = get_tweets_for_model(real_tweets)

        log("Using classifier " + name)
        log("Classifying fake tweets")
        fake_results = _classify_tweets(
            fake_tweets_generator,
            fake_tweets_unmodified,
            fake_article_sources,
            classifier,
        )

        log("Classifying real tweets")
        real_results = _classify_tweets(
            real_tweets_generator,
            real_tweets_unmodified,
            real_article_sources,
            classifier,
        )

        output_path = f"./results/{name}/"
        log("Outputting results to " + output_path)

        with open(output_path + "classifications_real.json", "w") as file:
            dump = json.dumps(real_results, indent=4)
            file.write(dump)
        with open(output_path + "classifications_fake.json", "w") as file:
            dump = json.dumps(fake_results, indent=4)
            file.write(dump)
    log("Execution complete")


if __name__ == "__main__":

    log("Loading real tweets")
    with open("real_cleaned.pkl", "rb") as file:
        real_tweets = pickle.load(file)

    log("Loading fake tweets")
    with open("fake_cleaned.pkl", "rb") as file:
        fake_tweets = pickle.load(file)

    naivebayes, svc, logistic = load_models()
    classifiers = {
        "naive": naivebayes,
        "svc": svc,
        "logistic": logistic,
        "rulebased": RuleBasedClassifier(),
    }
    run_classifiers(fake_tweets, real_tweets, classifiers)
