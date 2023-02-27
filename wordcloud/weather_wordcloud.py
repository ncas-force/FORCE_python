import tweepy
import pandas as pd
import time
import re
import datetime
import numpy as np
from PIL import Image
from bs4 import BeautifulSoup
from nltk.tokenize import WordPunctTokenizer
import os
token = WordPunctTokenizer()

def cleaning_tweets(t):
    del_amp = BeautifulSoup(t, 'lxml')
    del_amp_text = del_amp.get_text()
    del_link_mentions = re.sub(combined_re, '', del_amp_text)
    del_emoticons = re.sub(regex_pattern, '', del_link_mentions)
    lower_case = del_emoticons.lower()
    words = token.tokenize(lower_case)
    result_words = [x for x in words if len(x) > 2]
    return (" ".join(result_words)).strip()

FORCE_WRF_plotting=os.getenv('FORCE_WRF_plotting')
keys_dir=FORCE_WRF_plotting+"/wordcloud/twitter/.keys"

API_key = open(keys_dir+"/API_key", 'r').read().strip()
API_key_secret = open(keys_dir+"/API_key_secret", 'r').read().strip()
access_token = open(keys_dir+"/access_token", 'r').read().strip()
access_token_secret =open(keys_dir+"/access_token_secret", 'r').read().strip()
bearer_token_txt = open(keys_dir+"/bearer_token", 'r').read().strip()

client = tweepy.Client(bearer_token=bearer_token_txt, consumer_key=API_key, consumer_secret=API_key_secret, access_token=access_token, access_token_secret=access_token_secret)

users_ID = ["metoffice", "bbcweather", "PeterGWeather", "MetMattTaylor", "DerektheWeather", "SimonOKing", "Liamdutton", "ChrisPage90", "HollyJGreen", "TorroUK", "carolkirkwood", "fish4weather", "Netweather", "NickJF75"]

lastdayDateTime = datetime.datetime.now() - datetime.timedelta(hours = 24)

keyword = "("
for ID in users_ID:
    if ID == users_ID[-1]:
       keyword = keyword+"from:"+ID+") -is:retweet" 
    else:
       keyword = keyword+"from:"+ID+" OR "

csv_name = "forecast_wordcloud"

tweets = client.search_recent_tweets(query=keyword,start_time=lastdayDateTime.strftime('%Y-%m-%dT%H:00:00Z'))[0]
tweets_list=[]
for tweet in tweets:
   tweets_list.append(tweet)

df = pd.DataFrame(tweets_list)
df.to_csv('{}.csv'.format(csv_name), sep=',', index = False)

df = pd.read_csv("./"+csv_name+".csv")
pd.options.display.max_colwidth = 200
df.head()
df.shape

re_list = ['(https?://)?(www\.)?(\w+\.)?(\w+)(\.\w+)(/.+)?', '@[A-Za-z0-9_]+','#']
combined_re = re.compile( '|'.join( re_list) )

regex_pattern = re.compile(pattern = "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags = re.UNICODE)

print("Cleaning the tweets...\n")
cleaned_tweets = []
for t in df.text:
   if isinstance(t, str):
      cleaned_tweets.append(cleaning_tweets(t))

string = pd.Series(cleaned_tweets).str.cat(sep=' ')

from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

stopwords = set(STOPWORDS)
stopwords.update(["ukweather", "temp", "across", "parts", "today", "latest", "many", "rather", "week", "issue", "issued", "eng", "see", "past", "will", "around", "areas", "monday", "mon", "tuesday", "tues", "wednesday", "wed", "thursday", "thurs", "friday", "fri", "saturday", "sat", "sunday", "details", "met" "office", "metofficecop27", "matt", "simon", "x", "xx", "xxx", "netweather", "dot", "bbcweatherwatcher","mrbluesky", "god", "kellspics", "carol", "kirkwood", "photo", "photos", "photograph", "photographs","haha", "ha", "hehe", "didn", "mondaymorning", "typed", "come", "oo", "ooo", "oooo", "ooooo"])

# Rectangular word cloud

#wordcloud = WordCloud(width=1600, stopwords=stopwords,height=800,max_font_size=200,max_words=50,collocations=False, background_color='grey').generate(string)
#plt.figure(figsize=(10,7.5))
#plt.imshow(wordcloud, interpolation="bilinear")
#plt.axis("off")
#plt.savefig("test.png")

mask = np.array(Image.open('./Cloud.png'))

wordcloud = WordCloud(width=1500, mask = mask,stopwords=stopwords,height=909,max_font_size=200,max_words=50,collocation_threshold=15,background_color='black').generate(string)
f = plt.figure(figsize=(15,9))
plt.margins(x=0)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.savefig("Weather_wordcloud.png")

os.system("convert Weather_wordcloud.png -trim Weather_wordcloud.png")
os.system("rm ./"+csv_name+".csv")
