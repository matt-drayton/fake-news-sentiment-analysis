import json
from tqdm import tqdm
import pandas as pd

from utils import remove_noise, preprocess_bulk



def load_training_data():
    csv = pd.read_csv("training.1600000.processed.noemoticon.csv")
    csv.columns = ["target", "id", "date", "query", "user", "text"]
    positives = csv.loc[csv["target"] == 4]["text"].tolist()
    negatives = csv.loc[csv["target"] == 0]["text"].tolist()
    neutrals = csv.loc[csv["target"] == 2]["text"].tolist()
    return preprocess_bulk(positives), preprocess_bulk(negatives), preprocess_bulk(neutrals)


(
    positive_training_tweets,
    negative_training_tweets,
    neutral_training_tweets,
) = load_training_data()


# Clean up training / raw data
positive_training_tweets = [remove_noise(tweet) for tweet in positive_training_tweets]
negative_training_tweets = [remove_noise(tweet) for tweet in negative_training_tweets]
neutral_training_tweets = [remove_noise(tweet) for tweet in neutral_training_tweets]

with open("training_positive.json", "w") as file:
    dump = json.dumps(positive_training_tweets, indent=4)
    file.write(dump)

with open("training_negative.json", "w") as file:
    dump = json.dumps(negative_training_tweets, indent=4)
    file.write(dump)

