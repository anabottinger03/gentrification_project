""""
search with backoff method used for twitter API scraping. 
"""

import asyncio
from twikit import Client
import twikit
import os
import random
import pandas as pd

# Use environment variables for sensitive information
USERNAME = os.getenv('TWITTER_USERNAME')
EMAIL = os.getenv('TWITTER_EMAIL')
PASSWORD = os.getenv('TWITTER_PASSWORD')

# Initialize client globally
client = Client('en-US')

async def login():
    await client.login(
        auth_info_1=USERNAME,
        auth_info_2=EMAIL,
        password=PASSWORD,
        cookies_file=f'cookies_stealth.json'
    )

async def search_with_backoff(client, query, mode="Latest", retries=3):
    for i in range(retries):
        try:
            return await client.search_tweet(query, mode)
        except twikit.errors.TooManyRequests:
            wait_time = random.randint(900, 1200)  # 15-20 min cooldown
            print(f"[Rate Limit] Waiting {wait_time // 60} minutes before retrying...")
            await asyncio.sleep(wait_time)
    raise Exception("Exceeded retry attempts due to repeated rate limits.")

async def collect_tweets(neighborhood_name):
    tweet_list = []

    print(f"Searching batch 1 for '{neighborhood_name}'...")
    tweets = await search_with_backoff(client, neighborhood_name, "Latest")
    tweet_list.extend([tweet.text for tweet in tweets])
    await asyncio.sleep(random.uniform(10, 20))

    #pagination_depth = random.randint(2, 10)
    for i in range(1, 10):
        try:
            print(f"Searching batch {i+1} for '{neighborhood_name}'...")
            tweets = await tweets.next()
            tweet_list.extend([tweet.text for tweet in tweets])
            await asyncio.sleep(random.uniform(10, 20))
        except Exception as e:
            print(f"[Pagination Error] Stopping early due to: {e}")
            break

    return tweet_list

async def main():
    neighborhoods = [
        "borough park brooklyn", "bushwick brooklyn", "flatbush brooklyn",
        "upper east side manhattan", "sunset park brooklyn",
        "astoria queens", "central harlem manhattan",
        "coney island brooklyn", "rockaways queens"
    ]

    await login()
    print("Login successful")

    csv_path = "data/neighborhood_tweets_nyc_final.csv"
    first_write = not os.path.exists(csv_path)

    for neighborhood in neighborhoods:
        print(f"\n--- Collecting tweets for {neighborhood} ---")

        tweet_list = await collect_tweets(neighborhood)

        tweet_rows = [{"neighborhood": neighborhood, "tweet": tweet} for tweet in tweet_list]
        df = pd.DataFrame(tweet_rows)

        df.to_csv(csv_path, mode='a', index=False, header=first_write)
        if first_write:
            first_write = False

        print(f"Saved {len(df)} tweets for {neighborhood} to CSV.")

        # Random human-like pause before next neighborhood
        pause = random.uniform(60, 180)
        print(f"Waiting {pause:.2f} seconds before next neighborhood...")
        await asyncio.sleep(pause)

    print("\n--- All collections complete ---")

if __name__ == "__main__":
    asyncio.run(main())