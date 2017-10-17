import nltk
import collections
from collections import Counter
import random
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC, LinearSVC
from sklearn import metrics

document=open('C:\\Users\\user\\Desktop\\NLP\\NLP Project\\Special\\semeval2016-task6-trainingdata.txt',"r").read()
tuples=document.split('\n')
ids=[]
targets=[]
tweets=[]
stances=[]
for t in tuples:
	ids.append(t.split('\t')[0])
	targets.append(t.split('\t')[1])
	tweets.append(t.split('\t')[2])
	stances.append(t.split('\t')[3])
ids=ids[1:]
targets=targets[1:]
tweets=tweets[1:]
stances=stances[1:]

print(len(stances))

clean_tweets=[]
document2=open('C:\\Users\\user\\Desktop\\NLP\\NLP Project\\Special\\Final_training_tweet.txt',"r").read()
tuples2=document2.split('\n')
for t2 in tuples2:
	clean_tweets.append(t2)
print(len(clean_tweets))

tweets_with_labels=list(zip(clean_tweets, stances))
tweets_with_labels_atheism=tweets_with_labels[0:513]

all_bigrams=[]
all_bigrams=[b for t in tweets for b in zip(t.split(" ")[:-1], t.split(" ")[1:])]
b_features=Counter(all_bigrams).most_common(500)
bigram_features=[]
for (bigram,freq) in b_features:
	bigram_features.append(bigram)

def find_features(single_tweet):
	words=set(single_tweet)
	features={}
	for w in bigram_features:
		features[w]=(w in words)
	return features

featuresets=[(find_features(tweets), stances) for (tweets, stances) in tweets_with_labels]
random.shuffle(featuresets)
training_set=featuresets[:500]
testing_set=featuresets[501:]
MNB_classifier=SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print("MNB_classifier accuracy percentage:",(nltk.classify.accuracy(MNB_classifier,testing_set))*100)
SVC_classifier=SklearnClassifier(SVC())
SVC_classifier.train(training_set)
print("SVC_classifier accuracy percentage:",(nltk.classify.accuracy(SVC_classifier,testing_set))*100)

