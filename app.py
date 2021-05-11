import json
import pickle
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
from nltk import pos_tag
from nltk.tokenize.casual import casual_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
import re, string, random
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from rule_based_classifier import RuleBasedClassifier
from utils import preprocess, lemmatize_and_strip
import plotly.figure_factory as ff
import plotly.graph_objects as go

external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]


def load_models():
    with open("./classifiers/naive.pkl", "rb") as file:
        naivebayes = pickle.load(file)
    with open("./classifiers/svc.pkl", "rb") as file:
        svc = pickle.load(file)
    with open("./classifiers/logistic.pkl", "rb") as file:
        logistic = pickle.load(file)

    return (naivebayes, svc, logistic)


def load_models_meta():
    with open("./classifiers/meta_naive.pkl", "rb") as file:
        naive = pickle.load(file)
    with open("./classifiers/meta_svc.pkl", "rb") as file:
        svc = pickle.load(file)
    with open("./classifiers/meta_logistic.pkl", "rb") as file:
        logistic = pickle.load(file)
    with open("./classifiers/meta_rulebased.pkl", "rb") as file:
        rulebased = pickle.load(file)
    return {"naive": naive, "svc": svc, "lr": logistic, "rb": rulebased}


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
    removed_noise = lemmatize_and_strip(casual_tokenize(preprocessed))
    tokens = dict([token, True] for token in removed_noise)
    results = []
    for model in models:
        if type(model) is RuleBasedClassifier:
            results.append(f"{model.classify(text)} : {model.score(text)}")
        else:
            results.append(
                f"{model.classify(tokens)} : {model.prob_classify(tokens).prob('Positive')}"
            )
    return results


data = {
    "naive": load_results("naive"),
    "svc": load_results("svc"),
    "lr": load_results("logistic"),
    "rb": load_results("rulebased"),
}

meta = load_models_meta()

# naive_score, svc_score, logistic_score
naivebayes, svc, logistic = load_models()
classifiers = {
    "naive": naivebayes,
    "svc": svc,
    "lr": logistic,
    "rb": RuleBasedClassifier(),
}

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)


app.layout = html.Div(
    [
        dcc.Markdown(
            """
    # Fake News Sentiment Analysis
    Social Media has allowed for widespread poltiical discourse. At the same time, the amount of misinformation, or 'Fake News'
    has also increased. This project explores the usage of Sentiment Analysis to draw comparisons between reactions to false and accurate
    news articles. It will also compare different techniques of Sentiment Analysis.
    ## About the data
    The models seen have been trained on the Sentiment 140 dataset. They then have been given a series of tweets disucssing both fake and real
    news articles. These tweets are from the FakeNewsNet dataset. The visual demonstrations below show how the distribution of positive and negative
    tweets differs between tweets discussing fake and real news articles. 
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
        dcc.Graph(id="confusion-matrix"),
        html.P(id="modelstats"),
        html.P(id="combinedstatsinfo"),
        html.P(id="fakestatsinfo"),
        html.P(id="realstatsinfo"),
        dcc.Markdown(
            """
            ## Classification
            Type text into the input box below. The classification score from each method will be shown below.
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


def analyse_data(frame):
    article_grouping = frame.groupby("article_title")
    average_tweets_per_article = article_grouping.count().mean()["tweet"]
    mean_scores = frame.mean(numeric_only=True)
    std = frame.std()
    return f"Mean: {mean_scores['positive_score']:.3f}, Standard Deviation: {std['positive_score']:.3f}"


@app.callback(
    Output("histpos", "figure"),
    Output("confusion-matrix", "figure"),
    Output("modelstats", "children"),
    Output("combinedstatsinfo", "children"),
    Output("fakestatsinfo", "children"),
    Output("realstatsinfo", "children"),
    [Input("model-select", "value")],
)
def display_data(model):
    fake, real = data[model]
    model_meta = meta[model]

    combined = pd.concat(data[model])
    accuracy_stat = f"Model Accuracy on Test Data: {model_meta['accuracy']}"
    combined_stats = "All Tweets: " + analyse_data(combined)
    fake_stats = "Fake Tweets: " + analyse_data(fake)
    real_stats = "Real Tweets: " + analyse_data(real)
    fig = px.histogram(
        combined,
        x="positive_score",
        color="Truthful Article",
        histnorm="percent",
        marginal="box",
        barmode="overlay",
        nbins=20,
    )

    matrix_figure = ff.create_annotated_heatmap(
        model_meta["conf_matrix"],
        x=["Positive", "Negative"],
        y=["Positive", "Negative"],
    )
    matrix_figure.add_annotation(
        dict(
            font=dict(color="black", size=14),
            x=0.5,
            y=-0.15,
            showarrow=False,
            text="Predicted value",
            xref="paper",
            yref="paper",
        )
    )
    matrix_figure.add_annotation(
        dict(
            font=dict(color="white", size=14),
            x=0,
            y=0.5,
            showarrow=False,
            text="Real value",
            textangle=-90,
            xref="paper",
            yref="paper",
        )
    )
    matrix_figure.update_layout(margin=dict(t=50, l=200))
    matrix_figure["data"][0]["showscale"] = True

    return fig, matrix_figure, accuracy_stat, combined_stats, fake_stats, real_stats


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
    app.run_server(debug=False, dev_tools_ui=False)
