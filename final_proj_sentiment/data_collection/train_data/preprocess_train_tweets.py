"""
used the following sentiment dataset: https://www.kaggle.com/datasets/kazanova/sentiment140?resource=download
The file size is huge. For ease of analysis, I did not include the full size dataset, but this is the
script used to obtain the abridged verison used in training (data/training_tweets.csv)
"""
import pandas as pd 

def main(): 
    sentiment_data = pd.read_csv(
            "/Users/analisebottinger/Data/training.1600000.processed.noemoticon.csv",
            encoding="ISO-8859-1"  # or encoding="latin1"
        )
    cleaned_tweets = preprocess_tweets(sentiment_data)

    cleaned_tweets.to_csv("data/training_tweets_100k.csv", index=False)
    print("Saved training tweets to training_tweets.csv'")


def preprocess_tweets(tweet_data):
    # Sample negative tweets (label 0)
    negative = tweet_data[tweet_data.iloc[:, 0] == 0].iloc[:, [0, 5]]
    sampled_negative = negative.sample(n=100000, random_state=42)

    # Sample positive tweets (label 4)
    positive = tweet_data[tweet_data.iloc[:, 0] == 4].iloc[:, [0, 5]]
    sampled_positive = positive.sample(n=100000, random_state=42)

    # Concatenate
    cleaned_tweet_df = pd.concat([sampled_negative, sampled_positive], axis=0).reset_index(drop=True)

    # Rename columns
    cleaned_tweet_df.columns = ["sentiment_label", "tweet"]

    # Convert label 4 → 1 (positive)
    cleaned_tweet_df["sentiment_label"] = cleaned_tweet_df["sentiment_label"].replace(4, 1)

    # Shuffle rows
    cleaned_tweet_df = cleaned_tweet_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Clean tweet text: remove mentions, lowercase, strip whitespace
    cleaned_tweet_df["tweet"] = (
        cleaned_tweet_df["tweet"]
        .str.replace(r"@\w+", "", regex=True)
        .str.lower()
        .str.strip()
    )

    # Remove any NaNs, convert to str
    cleaned_tweet_df = cleaned_tweet_df.dropna(subset=["tweet", "sentiment_label"])
    cleaned_tweet_df["tweet"] = cleaned_tweet_df["tweet"].astype(str)

    assert cleaned_tweet_df["tweet"].isnull().sum() == 0, "tweets still contain NaNs!"

    print(cleaned_tweet_df.shape)
    print(cleaned_tweet_df["sentiment_label"].value_counts())

    return cleaned_tweet_df


if __name__ == "__main__":
    main()