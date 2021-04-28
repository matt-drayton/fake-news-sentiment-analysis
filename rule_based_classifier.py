from utils import remove_noise, preprocess
from nltk.tokenize import word_tokenize
from nltk import pos_tag

class RuleBasedClassifier():
    def __init__(self):
        self.dictionary = {}
        self.load_dictionary()
        self.dictionary_additions()

    def score(self, raw_text):
        total_score = 0 
        # processed_text = word_tokenize(preprocess(raw_text))
        for token, tag in pos_tag(raw_text.split(" ")):
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
        return "Positive" if score > 0 else "Negative"
        
                
    def load_dictionary(self):
        with open("SentiWords_1.1.txt", 'r') as dictionary:
            lines = dictionary.readlines()

        for line in lines:
            # Remove commented lines at top
            if line.startswith("#"):
                continue
            
            split_on_hash = line.split("#")
            word = split_on_hash[0]
            word.replace("_", " ")
            pos, score = split_on_hash[1].split('\t')
            self.dictionary[(word, pos)] = float(score)

    def dictionary_additions(self):
        for pos in ('v', 'n', 'a'):
            self.dictionary[(':)', pos)] = 1
            self.dictionary[(':D', pos)] = 2
            self.dictionary[('XD', pos)] = 2
            self.dictionary[(':/', pos)] = -0.5
            self.dictionary[(':(', pos)] = -1
            self.dictionary[(":'(", pos)] = -2
            self.dictionary[(">:(", pos)] = -4


            


