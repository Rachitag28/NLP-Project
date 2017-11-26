import nltk
import collections
from collections import Counter
import random
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB,BernoulliNB,GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn import metrics
from sklearn.metrics import f1_score
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from nltk.tokenize import TweetTokenizer
import numpy as np
from nltk import sent_tokenize
from nltk.corpus import stopwords

document=open('C:\\Users\\user\\Desktop\\NLP\\NLP Project\\Special\\training_hashtag.txt',"r").read()
tuples=document.split('\n')
tweets=[]
stances=[]
sentiments=[]
hashtags=[]
for t in tuples:
        tweets.append(t.split(',')[0])
        stances.append(t.split(',')[1])
        sentiments.append(t.split(',')[2])
        hashtags.append(t.split(',')[3])
tweets=[t.lower() for t in tweets]
tweets = [t for t in tweets if t not in stopwords.words('english')]

atheism_tweets=tweets[0:512]
atheism_sentiments=sentiments[0:512]
atheism_stances=stances[0:512]
atheism_hashtags=hashtags[0:512]
for at in atheism_tweets:
        at=at.split(' ',1)[0]
atheism_bigrams=[]

atheism_bigrams=[b for t in atheism_tweets for b in zip(t.split(" ")[:-1], t.split(" ")[1:])]
b_features=Counter(atheism_bigrams).most_common(3000)
frequent_atheism_bigrams=[]
for (bigram,freq) in b_features:
        frequent_atheism_bigrams.append(bigram)

def extract_np(psent):
  for subtree in psent.subtrees():
    if subtree.label() == 'NP':
      yield '_'.join(word for word, tag in subtree.leaves())

def find_all_noun_phrases(tweet):
	tokenized_sent = sent_tokenize(tweet)
	words=[nltk.word_tokenize(i) for i in tokenized_sent]
	tagged=[nltk.pos_tag(word) for word in words]
	grammar = r"""NP: {<DT|PP\$>?<JJ>*<NN.*>+}"""
	cp=nltk.RegexpParser(grammar)
	phrases=[]
	for i in words:
		tagged=nltk.pos_tag(i)
		parsed_sent = cp.parse(tagged)
	for npstr in extract_np(parsed_sent):
		phrases.append(npstr)
	return phrases

atheism_noun_phrases=[]
for ath_t in atheism_tweets:
        nounps=find_all_noun_phrases(ath_t)
        for nounp in nounps:
                atheism_noun_phrases.append(nounp)

np_features=Counter(atheism_noun_phrases).most_common(700)
frequent_noun_phrases=[]
for (nounp,freq) in np_features:
        frequent_noun_phrases.append(nounp)

all_hashtags=[]      
for ht in atheism_hashtags:
        tags=nltk.word_tokenize(ht)
        for tag in tags:
                all_hashtags.append(tag)
signs=['#',',','!','.']
all_hashtags=[t for t in all_hashtags if t not in signs]

frequent_hashtag_features=[]
ht_features=Counter(all_hashtags).most_common(457)
for (tags,freq) in ht_features:
        frequent_hashtag_features.append(tags)
        
document_test=open('C:\\Users\\user\\Desktop\\NLP\\NLP Project\\Special\\test_hashtag.txt',"r").read()
tuples_test=document_test.split('\n')
tweets_test=[]
stances_test=[]
sentiments_test=[]
hashtags_test=[]
for t in tuples_test:
        tweets_test.append(t.split(',')[0])
        stances_test.append(t.split(',')[1])
        sentiments_test.append(t.split(',')[2])
        hashtags_test.append(t.split(',')[3])
tweets_test=[t.lower() for t in tweets_test]

atheism_tweets_test=tweets_test[0:219]
atheism_sentiments_test=sentiments_test[0:219]
atheism_stances_test=stances_test[0:219]
atheism_hashtags_test=hashtags_test[0:219]
for at in atheism_tweets_test:
        at=at.split(' ',1)[0]

for att in atheism_tweets_test:
        atheism_tweets.append(att)
for ast in atheism_sentiments_test:
        atheism_sentiments.append(ast)
for astt in atheism_stances_test:
        atheism_stances.append(astt)
for aht in atheism_hashtags_test:
        atheism_hashtags.append(aht)
atheism_sets=zip(atheism_tweets,atheism_sentiments,atheism_stances,atheism_hashtags)

def find_features(single_tweet):
        big=[b for b in zip(single_tweet.split(" ")[:-1], single_tweet.split(" ")[1:])]
        st=[]
        for bi in big:
                if bi in frequent_atheism_bigrams:
                        joined_bigram="_".join(bi)
                        st.append(joined_bigram)                       
        string=" ".join(st)
####        features["sentiment"]=senti
        return string

def find_noun_phrases(tweet):
        tokenized_sent = sent_tokenize(tweet)
        words=[nltk.word_tokenize(i) for i in tokenized_sent]
        tagged=[nltk.pos_tag(word) for word in words]
        grammar = r"""NP: {<DT|PP\$>?<JJ>*<NN.*>+}"""
        cp=nltk.RegexpParser(grammar)
        phrases=[]
        for i in words:
                tagged=nltk.pos_tag(i)
                parsed_sent = cp.parse(tagged)
        for npstr in extract_np(parsed_sent):
                if npstr in frequent_noun_phrases:
                        phrases.append(npstr)
        string_of_phrases=" ".join(phrases)
        return string_of_phrases

def find_hashtag_features(hts):
        tags=nltk.word_tokenize(hts)
        ht=[]
        for tag in tags:
                if tag in frequent_hashtag_features:
                        ht.append(tag)
        return ' '.join(ht)
                

atheism_tweets_sentiments_labels=zip(atheism_tweets, atheism_sentiments, atheism_stances, atheism_hashtags)
atheism_tweets_sentiments_labels2=zip(atheism_tweets, atheism_sentiments, atheism_stances, atheism_hashtags)
##featuresets=[(find_features(tweets, sentiments), stances) for (tweets, sentiments, stances) in tweets_sentiments_labels]

featuresets=[find_features(tweet) for tweet in atheism_tweets]
tfidf=TfidfVectorizer()
bigram_features=tfidf.fit_transform(featuresets).toarray()

noun_phrases=[find_noun_phrases(tweet) for tweet in atheism_tweets]
cv1=CountVectorizer()
np_features=cv1.fit_transform(noun_phrases).toarray()

sentiments=[float(i) for i in atheism_sentiments]
sentiments=[int(i) for i in atheism_sentiments]
sentiments=[(i-min(sentiments))/(max(sentiments)-min(sentiments)) for i in sentiments]
sentiment_features=np.vstack((sentiments,sentiments)).T
atheism_sentiment_features=sentiment_features[:]

hashtag_features=[find_hashtag_features(atheism_hashtag) for atheism_hashtag in atheism_hashtags] 
cv2=CountVectorizer()
ht_features=cv2.fit_transform(hashtag_features).toarray()

all_features=np.hstack((bigram_features,np_features,atheism_sentiment_features,ht_features))
training_features=all_features[:512]
training_stances=atheism_stances[:512]
testing_features=all_features[513:]
testing_stances=atheism_stances[513:]

mnb=GaussianNB()
mnb.fit(training_features,training_stances)
pred=mnb.predict(testing_features)
print(f1_score(testing_stances, pred, average='macro'))
