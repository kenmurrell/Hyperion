import tweepy as tweepy
import formatter as fmt
import json

# Twitter API credentials
ckey = "eHmnwjQHDPDrfOKql9Qiut9wj"
csecret = "aVA9SICgTE0kobeBhyge2GYpF30BKXu92iJDDV5AoTJ5qUxxXn"
atoken = "897826166438014976-Otxe5wK936yEGx4zSGzWxNFph7rvFzs"
asecret = "1naa8yg03TIuWrjGiTKnLk2RiYWoL5Q8mq3GZIRlhvZS1"

#loads tweets to a file "imports.json". Enter the hashtag and the numbers of tweets
def load(hashtag,items):
    OAUTH_KEYS = {'consumer_key':ckey, 'consumer_secret':csecret,'access_token_key':atoken, 'access_token_secret':asecret}
    auth = tweepy.OAuthHandler(OAUTH_KEYS['consumer_key'], OAUTH_KEYS['consumer_secret'])
    api = tweepy.API(auth)

    tweet_list = {}
    x=0
    for tweet in tweepy.Cursor(api.search, q= hashtag).items(items):
        dict1={ 'twt_id':tweet.id_str,
                'created_at':tweet.created_at.strftime("%Y-%m-%d %H:%M:%S"),
                'user':tweet.user.name,
                'fav_cnt':tweet.favorite_count,
                'rt_count':tweet.retweet_count,
                'content':fmt.all(tweet.text)}
        tweet_list[x]=dict1
        x+=1

    with open('imports.json','w') as importfile:
        json.dump(tweet_list, importfile, sort_keys=True, indent=4, separators=(',', ': '))

def get():
    tweet_list = json.load(open('imports.json', 'r' ) )
    return tweet_list

# for i in tweet_list[:]:
#     print(i)
