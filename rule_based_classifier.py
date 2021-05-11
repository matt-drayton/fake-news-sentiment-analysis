from utils import preprocess
from nltk.tokenize.casual import casual_tokenize
from nltk import pos_tag
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm


class RuleBasedClassifier:
    def __init__(self):
        self.dictionary = {}
        self.load_dictionary()
        self.dictionary_additions()

    def score(self, raw_text):
        total_score = 0
        for token, tag in pos_tag(casual_tokenize(raw_text)):
            if tag.startswith("NN"):
                pos = "n"
            elif tag.startswith("VB"):
                pos = "v"
            else:
                pos = "a"

            if (token.lower(), pos) in self.dictionary:
                interim_score = self.dictionary[(token.lower(), pos)]
                if token == token.upper():
                    interim_score *= 1.5
                total_score += interim_score

        return total_score

    def classify(self, raw_text):
        score = self.score(raw_text)
        return "Positive" if score >= 0 else "Negative"

    def load_dictionary(self):
        with open("SentiWords_1.1.txt", "r") as dictionary:
            lines = dictionary.readlines()

        for line in lines:
            # Remove commented lines at top
            if line.startswith("#"):
                continue

            split_on_hash = line.split("#")
            word = split_on_hash[0]
            word.replace("_", " ")
            pos, score = split_on_hash[1].split("\t")
            self.dictionary[(word, pos)] = float(score)

    def dictionary_additions(self):
        for pos in ("v", "n", "a"):
            # Note all have been lowercased as tokens are lower before searching through dictionary
            self.dictionary[(":)", pos)] = 1
            self.dictionary[(":d", pos)] = 2
            self.dictionary[("xd", pos)] = 2
            self.dictionary[(":/", pos)] = -0.5
            self.dictionary[(":(", pos)] = -1
            self.dictionary[(":'(", pos)] = -2
            self.dictionary[(">:(", pos)] = -4

    def accuracy(self, test):
        correctly_predicted = 0
        incorrectly_predicted = 0
        for data, label in test:
            prediction = self.classify(data)
            if prediction == label:
                correctly_predicted += 1
            else:
                incorrectly_predicted += 1

        return correctly_predicted / (correctly_predicted + incorrectly_predicted)


if __name__ == "__main__":
    classifier = RuleBasedClassifier()
    csv = pd.read_csv("training.1600000.processed.noemoticon.csv")
    csv.columns = ["target", "id", "date", "query", "user", "text"]
    positives = csv.loc[csv["target"] == 4]["text"].tolist()
    negatives = csv.loc[csv["target"] == 0]["text"].tolist()

    #              Predicted Values
    #           Positive    Negative
    # Negative
    # Positive
    conf_matrix = np.zeros((2, 2))

    for positive in tqdm(positives):
        pred = classifier.classify(positive)
        if pred == "Positive":
            conf_matrix[0][1] += 1
        else:
            conf_matrix[1][1] += 1

    for negative in tqdm(negatives):
        pred = classifier.classify(negative)
        if pred == "Negative":
            conf_matrix[1][0] += 1
        else:
            conf_matrix[0][0] += 1

    accuracy = (conf_matrix[0][1] + conf_matrix[1][0]) / np.sum(conf_matrix)

    meta = {
        "accuracy": accuracy,
        "conf_matrix": conf_matrix,
    }
    with open("./classifiers/meta_rulebased.pkl", "wb") as file:
        pickle.dump(meta, file)
