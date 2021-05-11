import json
from tqdm import tqdm
import pandas as pd
from utils import lemmatize_and_strip, preprocess_bulk, log


def load_training_data():
    """Parse the training data from its CSV file

    Returns:
        Tuple containing the pre-processed positive and negative tweets
    """
    csv = pd.read_csv("training.1600000.processed.noemoticon.csv")
    csv.columns = ["target", "id", "date", "query", "user", "text"]
    positives = csv.loc[csv["target"] == 4]["text"].tolist()
    negatives = csv.loc[csv["target"] == 0]["text"].tolist()

    return (
        preprocess_bulk(positives),
        preprocess_bulk(negatives),
    )


if __name__ == "__main__":
    log("Loading training data")
    (
        positive_training_tweets,
        negative_training_tweets,
    ) = load_training_data()

    # Clean up training / raw data
    log("Preprocessing positive and negative training tweets")
    positive_training_tweets = [
        lemmatize_and_strip(tweet) for tweet in tqdm(positive_training_tweets)
    ]
    negative_training_tweets = [
        lemmatize_and_strip(tweet) for tweet in tqdm(negative_training_tweets)
    ]

    log("Saving to files")
    with open("training_positive.json", "w") as file:
        dump = json.dumps(positive_training_tweets, indent=4)
        file.write(dump)

    with open("training_negative.json", "w") as file:
        dump = json.dumps(negative_training_tweets, indent=4)
        file.write(dump)
    log("Execution complete")
