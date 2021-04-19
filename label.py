import os
import random
import json


BASEURL = os.getcwd()+"/fakenewsanalysis/fakenewsnet_dataset/politifact/"

positive_tweets = []
objective_tweets = []
negative_tweets = []


def tweet_generator():
    while True:
        real_or_fake = random.choice(['real/', 'fake/'])
        directory = BASEURL + real_or_fake

        rand_tweet_json = None

        while rand_tweet_json is None:
            try:
                article = random.choice(os.listdir(directory))
                article_path = directory + article + "/tweets/"
                rand_tweet_json = article_path + random.choice(os.listdir(article_path))
            except:
                pass

        with open(rand_tweet_json) as tweet_file:
            tweet_data = json.load(tweet_file)
            yield tweet_data['text']


def classify_tweet(tweet):
    print("\nNew Tweet fetched. Please classify (p: Positive; _: Objective; n: Negative; q: Exit)\n")
    print(tweet)
    while True:
        response = input()
        if response == 'p':
            positive_tweets.append(tweet)
            print("Classified as positive")
            return True
        if response == 'n':
            negative_tweets.append(tweet)
            print("Classified as negative")
            return True

        if response == '':
            objective_tweets.append(tweet)
            print("Classified as objective")
            return True

        if response == 'q':
            print("Exiting classifier")
            return False

        print("Invalid selection. Please re-classify (p: Positive; _: Objective; n: Negative; q: Exit)")
        


if __name__ == '__main__':
    
    print("Welcome to the data labeller. Please label the sentiment of the tweet(s) you are shown as follows: \n(p: Positive; _: Objective; n: Negative; q: Exit)")

    tweet_gen = tweet_generator()
    for tweet in tweet_gen:
        if not classify_tweet(tweet):
            break    

    with open("positive_labelled_tweets.json", "r+") as pos_tweets:
        current_labels = json.load(pos_tweets)
        current_labels['tweets'] += positive_tweets
        pos_tweets.seek(0)
        json.dump(current_labels, pos_tweets)
        pos_tweets.truncate()

    with open("negative_labelled_tweets.json", "r+") as neg_tweets:
        current_labels = json.load(neg_tweets)
        current_labels['tweets'] += negative_tweets
        neg_tweets.seek(0)
        json.dump(current_labels, neg_tweets)
        neg_tweets.truncate()

    with open("objective_labelled_tweets.json", "r+") as obj_tweets:
        current_labels = json.load(obj_tweets)
        current_labels['tweets'] += objective_tweets
        obj_tweets.seek(0)
        json.dump(current_labels, obj_tweets)
        obj_tweets.truncate()