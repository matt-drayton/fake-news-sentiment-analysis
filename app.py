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
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from rule_based_classifier import RuleBasedClassifier

external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]


def preprocess(tweet):
    new_tweet = tweet
    new_tweet = re.sub(r"http\S+", "", new_tweet)  # Remove URLs
    new_tweet = re.sub(r"@[A-Za-z0-9_-]*", "", new_tweet)  # Remove Handles
    new_tweet = re.sub(r"RT+", "", new_tweet)  # Remove RT
    new_tweet = re.sub(r"\\.+?(\b)", "", new_tweet)  # Remove newlines, unicodes, etc
    new_tweet = re.sub(r"\bvia\b", "", new_tweet)  # Remove 'via' from tweet
    new_tweet = re.sub(r"#", "", new_tweet)  # Remove # signs
    new_tweet = word_tokenize(new_tweet)
    return new_tweet


def remove_noise(tweet_tokens):
    stop_words = stopwords.words("english")

    cleaned_tokens = []

    for token, tag in pos_tag(tweet_tokens):
        token = re.sub(
            "http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|"
            "(?:%[0-9a-fA-F][0-9a-fA-F]))+",
            "",
            token,
        )
        token = re.sub("(@[A-Za-z0-9_]+)", "", token)

        if tag.startswith("NN"):
            pos = "n"
        elif tag.startswith("VB"):
            pos = "v"
        else:
            pos = "a"

        lemmatizer = WordNetLemmatizer()
        token = lemmatizer.lemmatize(token, pos)

        if (
            len(token) > 0
            and token not in string.punctuation
            and token.lower() not in stop_words
        ):
            cleaned_tokens.append(token.lower())
    return cleaned_tokens


def load_models():
    with open("./classifiers/naive.pkl", "rb") as file:
        naivebayes = pickle.load(file)
    with open("./classifiers/svc.pkl", "rb") as file:
        svc = pickle.load(file)
    with open("./classifiers/logistic.pkl", "rb") as file:
        logistic = pickle.load(file)

    return (naivebayes, svc, logistic)


def load_results(model):
    with open(f"./results/{model}/classifications_fake.json", "r") as file:
        classifications_fake = pd.DataFrame.from_dict(json.load(file))
    with open(f"./results/{model}/classifications_real.json", "r") as file:
        classifications_real = pd.DataFrame.from_dict(json.load(file))

    # Add columns to df specifying if the tweet is about a fake or true article
    classifications_fake["Truthful Article"] = False
    classifications_real["Truthful Article"] = True
    return classifications_fake, classifications_real


def classify_new(text, models):
    preprocessed = preprocess(text)
    removed_noise = remove_noise(preprocessed)
    tokens = dict([token, True] for token in removed_noise)
    results = []
    for model in models:
        if type(model) is RuleBasedClassifier:
            results.append(f"{model.classify(text)} : {model.score(text)}")
        else:
            results.append(f"{model.classify(tokens)} : {model.prob_classify(tokens).prob('Positive')}")
    return results


data = {
    "naive": load_results("naive"),
    "svc": load_results("svc"),
    "lr": load_results("logistic"),
    "rb": load_results("rulebased"),
}

naivebayes, svc, logistic = load_models()
classifiers = {"naive": naivebayes, "svc": svc, "lr": logistic, "rb": RuleBasedClassifier()}

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)


app.layout = html.Div(
    [
        dcc.Markdown(
            """
    # Fake News Sentiment Analysis
    Social Media has allowed for widespread poltiical discourse. At the same time, the amount of misinformation, or 'Fake News'
    has also increased. This project explores the usage of Sentiment Analysis to draw comparisons between reactions to false and accurate
    news articles. It will also compare different techniques of Sentiment Analysis.
    ## Methods
    """
        ),
        dcc.Dropdown(
            id="model-select",
            options=[
                {"label": "Naive Bayes", "value": "naive"},
                {"label": "Support Vector Classification", "value": "svc"},
                {"label": "Logistic Regression", "value": "lr"},
                {"label": "Rule Based", "value": "rb"},
            ],
            value="naive",
        ),
        dcc.Graph(id="histpos"),
        dcc.Markdown(
            """
            ## Classification
            Type text into the input box below. Then click the "Enter" button. This will return the classification yielded
            by the chosen model in the dropdown.
            """
        ),
        html.Div(
            [
                dcc.Input(id="classify-text", value="I'm Happy!"),
                html.P(
                    children=["Naive Bayes: ", html.Span(id="classify-naive-result")]
                ),
                html.P(
                    children=[
                        "Suppor Vector Classification: ",
                        html.Span(id="classify-svc-result"),
                    ]
                ),
                html.P(
                    children=[
                        "Logistic Regression: ",
                        html.Span(id="classify-lr-result"),
                    ]
                ),
                html.P(
                    children=[
                        "Rule-based: ",
                        html.Span(id="classify-rb-result"),
                    ]
                ),
            ]
        ),
    ]
)


@app.callback(Output("histpos", "figure"), [Input("model-select", "value")])
def display_data(model):
    fake, real = data[model]
    combined = pd.concat(data[model])

    fig = px.histogram(combined, x="positive_score", color="Truthful Article", histnorm="percent", marginal="box", barmode="overlay", nbins=20)
    # fig = go.Figure()
    # fig.add_trace(go.Histogram(x=real['positive_score'], histnorm="probability"))
    # fig.add_trace(go.Histogram(x=fake['positive_score'], histnorm="probability"))
    # fig.update_layout(barmode='overlay')
    # fig.update_traces(opacity=0.75)
    return fig
# TODO: WRITE OWN "RULE BASED" SENTIMENT ANALYISISISIS

@app.callback(
    Output("classify-naive-result", "children"),
    Output("classify-svc-result", "children"),
    Output("classify-lr-result", "children"),
    Output("classify-rb-result", "children"),
    [Input("classify-text", "value")],
)
def classify_text(text):
    if len(text) > 0:
        return classify_new(text, classifiers.values())
    else:
        return ["Null" for classifier in classifiers]

if __name__ == "__main__":
    app.run_server(debug=True)
