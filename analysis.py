import json
import pandas as pd
import matplotlib.pyplot as plt

with open("classifications_fake.json", "r") as file:
    classifications_fake = json.load(file)
with open("classifications_real.json", "r") as file:
    classifications_real = json.load(file)


fake = pd.DataFrame.from_dict(classifications_fake)
real = pd.DataFrame.from_dict(classifications_real)


def analyse_data(frame):
    article_grouping = frame.groupby("article_title")
    average_tweets_per_article = article_grouping.count().mean()["tweet"]
    mean_scores = frame.mean(numeric_only=True)
    std = frame.std()
    print("Average scores:")
    print(mean_scores)
    print("Standard Deviation:")
    print(std)
    print("Standard Deviation Mean Ratio:")
    print(std / mean_scores)
    print("\nAverage tweets/article:", average_tweets_per_article)
    print("\n\n")


print("========Fake Tweets Analysis========")
analyse_data(fake)
plt.hist(fake["negative_score"])
plt.show()

print("========Real Tweets Analysis========")
analyse_data(real)
plt.hist(real["negative_score"])
plt.show()
