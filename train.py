import pickle
from nltk import NaiveBayesClassifier
from nltk import classify
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import re, string, random
import json
from tqdm import tqdm
from utils import log
from rule_based_classifier import RuleBasedClassifier
from sklearn.metrics import confusion_matrix

def load_training_data():
    with open("training_positive.json", "r") as file:
        positives = json.load(file)
    with open("training_negative.json", "r") as file:
        negatives = json.load(file)
    return positives, negatives

def get_tweets_for_model(cleaned_tokens_list):
    for tweet_tokens in cleaned_tokens_list:
        yield dict([token, True] for token in tweet_tokens)

def create_confusion_matrix(model, data):
    tweets = [point[0] for point in data]
    true_labels = [point[1] for point in data]
    pred_labels = model.classify_many(tweets)

    return confusion_matrix(y_true=true_labels, y_pred=pred_labels, labels=['Positive', 'Negative'])

if __name__ == "__main__":
    # Load tweet examples to train model
    log("Loading training data")
    positive_training_tweets, negative_training_tweets = load_training_data()

    positive_training_tweets = get_tweets_for_model(positive_training_tweets)
    negative_training_tweets = get_tweets_for_model(negative_training_tweets)


    # Model Training
    positive_dataset = [(tweet, "Positive") for tweet in positive_training_tweets]
    negative_dataset = [(tweet, "Negative") for tweet in negative_training_tweets]

    total_dataset = positive_dataset + negative_dataset
    random.shuffle(total_dataset)

    # All classifiers can take different amounts of training data in reasonable time
    # Naive Bayes
    naive_train_data = total_dataset[: int(len(total_dataset) * 0.7)]
    naive_test_data = total_dataset[int(len(total_dataset) * 0.7) :]

    # Logistic
    logistic_train_data = total_dataset[: int(len(total_dataset) * 0.7)]
    logistic_test_data = total_dataset[int(len(total_dataset) * 0.7) :]

    # SVC
    svc_train_data = total_dataset[:5000]
    svc_test_data = total_dataset[5000:]

    classifiers = {
        "naive": NaiveBayesClassifier.train(naive_train_data),
        "logistic": classify.SklearnClassifier(LogisticRegression()),
        "svc": classify.SklearnClassifier(SVC(kernel="linear", probability=True)),
    }

    data = {
        "naive": (naive_train_data, naive_test_data),
        "logistic": (logistic_train_data, logistic_test_data),
        "svc": (svc_train_data, svc_test_data),
    }

    for name, classifier in classifiers.items():
        log("Training model " + name)
        train, test = data[name]
        if name != "naive":
            classifier.train(train)
        accuracy = classify.accuracy(classifier, test)

        log("Creating confusion matrix")
        conf_matrix = create_confusion_matrix(classifier, test)

        log(f"Exporting classifier {name} with accuracy {accuracy}.")
        meta_info = {
            "accuracy": accuracy,
            "conf_matrix": conf_matrix,
        }
        with open(f"./classifiers/meta_{name}.pkl", "wb") as file:
            pickle.dump(meta_info, file)

        with open(f"./classifiers/{name}.pkl", "wb") as file:
            pickle.dump(classifier, file)

    
    log("Finished Executing.")
