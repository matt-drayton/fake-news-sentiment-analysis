import os
import random
import json
import pickle


positive_tweets = []
objective_tweets = []
negative_tweets = []


def tweet_generator(fake_tweets, real_tweets):
    while True:
        tweets = random.choice([fake_tweets, real_tweets])
        yield random.choice(tweets)["tweet"]


def classify_tweet(tweet):
    print(
        "\nNew Tweet fetched. Please classify (p: Positive; _: Objective; n: Negative; q: Exit)\n"
    )
    print(tweet)
    while True:
        response = input()
        if response == "p":
            positive_tweets.append(tweet)
            print("Classified as positive")
            return True
        if response == "n":
            negative_tweets.append(tweet)
            print("Classified as negative")
            return True

        if response == "":
            objective_tweets.append(tweet)
            print("Classified as objective")
            return True

        if response == "q":
            print("Exiting classifier")
            return False

        print(
            "Invalid selection. Please re-classify (p: Positive; _: Objective; n: Negative; q: Exit)"
        )


if __name__ == "__main__":
    with open("real_cleaned.pkl", "rb") as file:
        real_tweets = pickle.load(file)

    with open("fake_cleaned.pkl", "rb") as file:
        fake_tweets = pickle.load(file)

    print(
        "Welcome to the data labeller. Please label the sentiment of the tweet(s) you are shown as follows: \n(p: Positive; _: Objective; n: Negative; q: Exit)"
    )

    tweet_gen = tweet_generator(fake_tweets, real_tweets)
    for tweet in tweet_gen:
        if not classify_tweet(tweet):
            break

    with open("positive_labelled_tweets.json", "r+") as pos_tweets:
        current_labels = json.load(pos_tweets)
        current_labels["tweets"] += positive_tweets
        pos_tweets.seek(0)
        json.dump(current_labels, pos_tweets)
        pos_tweets.truncate()

    with open("negative_labelled_tweets.json", "r+") as neg_tweets:
        current_labels = json.load(neg_tweets)
        current_labels["tweets"] += negative_tweets
        neg_tweets.seek(0)
        json.dump(current_labels, neg_tweets)
        neg_tweets.truncate()

    with open("objective_labelled_tweets.json", "r+") as obj_tweets:
        current_labels = json.load(obj_tweets)
        current_labels["tweets"] += objective_tweets
        obj_tweets.seek(0)
        json.dump(current_labels, obj_tweets)
        obj_tweets.truncate()
