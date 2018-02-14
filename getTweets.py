import tweepy as tweepy
import json
import csv
from tqdm import tqdm

# Twitter API credentials
ckey = "eHmnwjQHDPDrfOKql9Qiut9wj"
csecret = "aVA9SICgTE0kobeBhyge2GYpF30BKXu92iJDDV5AoTJ5qUxxXn"
atoken = "897826166438014976-Otxe5wK936yEGx4zSGzWxNFph7rvFzs"
asecret = "1naa8yg03TIuWrjGiTKnLk2RiYWoL5Q8mq3GZIRlhvZS1"

json_file = 'imports.json'
csv_file = 'new_dataset.csv'

#loads tweets to a file "imports.json". Enter the hashtag and the numbers of tweets
def load(hashtag,items):
    print("Importing tweets...")
    OAUTH_KEYS = {'consumer_key':ckey, 'consumer_secret':csecret,'access_token_key':atoken, 'access_token_secret':asecret}
    auth = tweepy.OAuthHandler(OAUTH_KEYS['consumer_key'], OAUTH_KEYS['consumer_secret'])
    api = tweepy.API(auth)

    tweet_list = {}
    for tweet in tweepy.Cursor(api.search, q= hashtag).items(items):
        dict1={ 'twt_id':tweet.id_str,
                'created_at':tweet.created_at.strftime("%Y-%m-%d %H:%M:%S"),
                'user':tweet.user.name,
                'fav_cnt':tweet.favorite_count,
                'rt_count':tweet.retweet_count,
                'content':tweet.text}
        tweet_list[tweet.id_str]=dict1
    print("Saving data...")
    with open(json_file,'w') as importfile:
        json.dump(tweet_list, importfile, sort_keys=True, indent=4, separators=(',', ': '))

def move(info=False):
    print("Converting to csv...")
    with open(json_file, 'r') as jfile, open(csv_file, 'w',newline='', encoding='utf-8') as cfile:
        writer = csv.writer(cfile, delimiter=',')
        tweet_dict = json.load(jfile)
        for tweet in tweet_dict.values():
            dataset = []
            if info:
                dataset.append(tweet['twt_id'])
                dataset.append(tweet['created_at'])
                dataset.append(tweet['user'])
                dataset.append(tweet['fav_cnt'])
                dataset.append(tweet['content'])
            else:
                dataset.append(tweet['content'])
            writer.writerow(dataset)

load('maga',3)
move()
