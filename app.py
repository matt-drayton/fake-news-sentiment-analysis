import json
import pickle
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
import re, string, random

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

def preprocess(tweet):
    new_tweet = tweet
    new_tweet = re.sub(r"http\S+", "", new_tweet) # Remove URLs
    new_tweet = re.sub(r"@[A-Za-z0-9_-]*", "", new_tweet) # Remove Handles
    new_tweet = re.sub(r"RT+", "", new_tweet) # Remove RT
    new_tweet = re.sub(r"\\.+?(\b)", "", new_tweet) # Remove newlines, unicodes, etc
    new_tweet = re.sub(r"\bvia\b", "", new_tweet) # Remove 'via' from tweet
    new_tweet = re.sub(r"#", "", new_tweet) # Remove # signs
    new_tweet = word_tokenize(new_tweet)
    return new_tweet


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

def load_models():
    with open("./classifications/NAIVE/naivebayes.pkl", 'rb') as file:
        naivebayes = pickle.load(file)
    with open("./classifications/NAIVE/naivebayes.pkl", 'rb') as file:
        svc = pickle.load(file)
    with open("./classifications/NAIVE/naivebayes.pkl", 'rb') as file:
        logistic = pickle.load(file)
    
    
    return (naivebayes, svc, logistic)

def load_results(model):
    with open(f"./classifications/{model}/classifications_fake.json", 'r') as file:
        classifications_fake = json.load(file) 
    with open(f"./classifications/{model}/classifications_real.json", 'r') as file:
        classifications_real = json.load(file) 
    
    return pd.DataFrame.from_dict(classifications_fake), pd.DataFrame.from_dict(classifications_real) 


def categorise_new(text, model):
    preprocessed = preprocess(text)
    removed_noise = remove_noise(preprocessed)

    return model.classify(dict([token, True] for token in removed_noise))



# naive_fake, naive_real = load_results("NAIVE")
# svc_fake, svc_real = load_results("SVC")
# log_fake, log_real = load_results("LOGISTIC")
data = {
    "naive": load_results("NAIVE"),
    "svc": load_results("SVC"),
    "lr": load_results("LOGISTIC")
}

naivebayes, svc, logistic = load_models()

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)



app.layout = html.Div([
    dcc.Markdown("""
    # Fake News Sentiment Analysis
    Social Media has allowed for widespread poltiical discourse. At the same time, the amount of misinformation, or 'Fake News'
    has also increased. This project explores the usage of Sentiment Analysis to draw comparisons between reactions to false and accurate
    news articles. It will also compare different techniques of Sentiment Analysis.
    ## Methods
    """),
    dcc.Dropdown(id="model-select",
                options=[
                    {'label': 'Naive Bayes', 'value': 'naive'},
                    {'label': 'Support Vector Classification', 'value': 'svc'},
                    {'label': 'Logistic Regression', 'value': 'lr'},
                ],
                value="naive"
                ),
    dcc.Graph(id="graph")
])


@app.callback(
    Output("graph", "figure"),
    [Input("model-select", "value")]
)
def display_data(model):
    fake, real = data[model]
    fig = px.histogram(fake['positive_score'])
    return fig


if __name__ == '__main__':
    app.run_server(debug=True)
