import pickle
from datetime import datetime
from tqdm import tqdm
import json
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re, string, random
from nltk import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer

LOG = True

def remove_noise(tweet_tokens):
    stop_words = stopwords.words('english')

    cleaned_tokens = []

    for token, tag in pos_tag(tweet_tokens):
        token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'\
                       '(?:%[0-9a-fA-F][0-9a-fA-F]))+','', token)
        token = re.sub("(@[A-Za-z0-9_]+)","", token)

        if tag.startswith("NN"):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'

        lemmatizer = WordNetLemmatizer()
        token = lemmatizer.lemmatize(token, pos)

        if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:
            cleaned_tokens.append(token.lower())
    return cleaned_tokens

def log(text):
    if LOG:
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print(f"[{current_time}] {text}")

def load_models():
    log("Loading models")
    with open("./classifications/NAIVE/naivebayes.pkl", 'rb') as file:
        naivebayes = pickle.load(file)
    with open("./classifications/NAIVE/naivebayes.pkl", 'rb') as file:
        svc = pickle.load(file)
    with open("./classifications/NAIVE/naivebayes.pkl", 'rb') as file:
        logistic = pickle.load(file)
    
    
    return (naivebayes, svc, logistic)

naivebayes, svc, logistic = load_models()

classifiers = {
    "naive": naivebayes,
    "svc": svc,
    "logistic": logistic
}

log("Loading real tweets")
with open("real1.pkl", 'rb') as file:
    real_tweets = pickle.load(file)

log("Loading fake tweets")
with open("fake1.pkl", 'rb') as file:
    fake_tweets = pickle.load(file)

def get_tweets_for_model(cleaned_tokens_list):
    for tweet_tokens in cleaned_tokens_list:
        yield dict([token, True] for token in tweet_tokens)

def _classify_tweets(tweets, raw_tweets, article_sources, classifier):
    total_output = []
    for tweet, raw_tweet, article_source in tqdm(zip(tweets, raw_tweets, article_sources)):
        
        joined_tweet = ' '.join(tweet.keys())
        probs = classifier.prob_classify(tweet)
        output = {
            "tweet": joined_tweet,
            "raw_tweet": raw_tweet,
            "article_title": article_source,
            "classification": classifier.classify(tweet),
            "positive_score": probs.prob("Positive"),
            "negative_score": probs.prob("Negative"),
        }
        total_output.append(output)
    return total_output

def run_classifiers(fake_tweets, real_tweets, classifiers):
    log("Extracting tweets from pickle")
    fake_tweets_unmodified = [tweet['tweet'] for tweet in tqdm(fake_tweets)]
    fake_article_sources = [tweet['article_title'] for tweet in tqdm(fake_tweets)]
    fake_tweets = [word_tokenize(tweet['tweet']) for tweet in tqdm(fake_tweets)]

    real_tweets_unmodified = [tweet['tweet'] for tweet in tqdm(real_tweets)]
    real_article_sources = [tweet['article_title'] for tweet in tqdm(real_tweets)]
    real_tweets = [word_tokenize(tweet['tweet']) for tweet in tqdm(real_tweets)]

    log("Removing noise")
    fake_tweets = [remove_noise(tweet) for tweet in tqdm(fake_tweets)]

    real_tweets = [remove_noise(tweet) for tweet in tqdm(real_tweets)]

    log("Beginning classifications")
    for name, classifier in classifiers.items():
        fake_tweets_generator = get_tweets_for_model(fake_tweets)
        real_tweets_generator = get_tweets_for_model(real_tweets)

        log("Using classifier "+name)
        log("Classifying fake tweets")
        fake_results = _classify_tweets(fake_tweets_generator, fake_tweets_unmodified, fake_article_sources, classifier)

        log("Classifying real tweets")
        real_results = _classify_tweets(real_tweets_generator, real_tweets_unmodified, real_article_sources, classifier)

        output_path = f"./results/{name}/"
        log("Outputting results to "+output_path)

        with open(output_path+"classifications_real.json", 'w') as file:
            dump = json.dumps(real_results, indent=4)
            file.write(dump)
        with open(output_path+"classifications_fake.json", 'w') as file:
            dump = json.dumps(fake_results, indent=4)
            file.write(dump)
    log("Execution complete")
run_classifiers(fake_tweets, real_tweets, classifiers)