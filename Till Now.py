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
from sklearn.metrics import f1_score
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import numpy as np
from nltk import sent_tokenize

document=open('C:\\Users\\user\\Desktop\\NLP\\NLP Project\\Special\\tweet_sentiment_label.txt',"r").read()
tuples=document.split('\n')
tweets=[]
stances=[]
sentiments=[]
for t in tuples:
        tweets.append(t.split(',')[0])
        stances.append(t.split(',')[1])
        sentiments.append(t.split(',')[2])
tweets=[t.lower() for t in tweets]
tweets_sentiments_labels=zip(tweets, sentiments, stances)

atheism_tweets=tweets[0:512]
atheism_sentiments=sentiments[0:512]
atheism_stances=stances[0:512]

atheism_bigrams=[]
atheism_bigrams=[b for t in atheism_tweets for b in zip(t.split(" ")[:-1], t.split(" ")[1:])]
b_features=Counter(atheism_bigrams).most_common(2000)
bigram_features=[]

for (bigram,freq) in b_features:
        bigram_features.append(bigram)
def find_features(single_tweet):
        big=[b for b in zip(single_tweet.split(" ")[:-1], single_tweet.split(" ")[1:])]
        st=[]
        for bi in big:
                if bi in bigram_features:
                        joined_bigram="_".join(bi)
                        st.append(joined_bigram)                       
        string=" ".join(st)
##        features["sentiment"]=senti
        return string

def extract_np(psent):
  for subtree in psent.subtrees():
    if subtree.label() == 'NP':
      yield '_'.join(word for word, tag in subtree.leaves())

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
		phrases.append(npstr)
	return(" ".join(phrases))


atheism_tweets_sentiments_labels=zip(atheism_tweets, atheism_sentiments, atheism_stances)
atheism_tweets_sentiments_labels2=zip(atheism_tweets, atheism_sentiments, atheism_stances)
##featuresets=[(find_features(tweets, sentiments), stances) for (tweets, sentiments, stances) in tweets_sentiments_labels]

featuresets=[(find_features(tweets)) for (tweets, sentiments, stances) in atheism_tweets_sentiments_labels]
tfidf=CountVectorizer()
bigram_features=tfidf.fit_transform(featuresets).toarray()

noun_phrases=[(find_noun_phrases(tweets)) for (tweets, sentiments, stances) in atheism_tweets_sentiments_labels2]
cv1=CountVectorizer()
np_features=cv1.fit_transform(noun_phrases).toarray()

sentiments=[float(i) for i in sentiments]
sentiments=[int(i) for i in sentiments]
sentiments=[(i-min(sentiments))/(max(sentiments)-min(sentiments)) for i in sentiments]
sentiment_features=np.vstack((sentiments,sentiments)).T
atheism_sentiment_features=sentiment_features[0:512]


all_features=np.hstack((bigram_features,np_features,atheism_sentiment_features))
print(np.shape(bigram_features))
print(np.shape(np_features))
print(np.shape(all_features))
print(np.shape(atheism_sentiment_features))
x_train,x_test,y_train,y_test=train_test_split(all_features, atheism_stances, test_size=0.20, random_state=4)
mnb=MultinomialNB()
mnb.fit(x_train,y_train)
pred=mnb.predict(x_test)
print(f1_score(y_test, pred, average='macro'))
##training_set_atheism=featuresets[1:413]
##testing_features_atheism=featuresets[414:513]
##random.shuffle(training_set_atheism)
##random.shuffle(testing_features_atheism)
##training_set_climate=featuresets[514:908]
##random.shuffle(training_set_climate)
##training_set_feminist=featuresets[909:1572]
##random.shuffle(training_set_feminist)
##training_set_hillary=featuresets[1573:2211]
##random.shuffle(training_set_hillary)
##training_set_abortion=featuresets[2212:2814]
##random.shuffle(training_set_abortion)
##
##
##MNB_atheism=SklearnClassifier(MultinomialNB())
##MNB_atheism.train(training_set_atheism)
##MNB_climate=SklearnClassifier(MultinomialNB())
##MNB_climate.train(training_set_climate)
##MNB_feminist=SklearnClassifier(MultinomialNB())
##MNB_feminist.train(training_set_feminist)
##MNB_hillary=SklearnClassifier(MultinomialNB())
##MNB_hillary.train(training_set_hillary)
##MNB_abortion=SklearnClassifier(MultinomialNB())
##MNB_abortion.train(training_set_abortion)
######print("MNB_classifier accuracy percentage:",(nltk.classify.accuracy(MNB_classifier,testing_set))*100)
####SVC_classifier=SklearnClassifier(SVC())
####SVC_classifier.train(training_set)
######print("SVC_classifier accuracy percentage:",(nltk.classify.accuracy(SVC_classifier,testing_set))*100)
##
##
##clean_tweets_testing=[]
##document3=open('C:\\Users\\user\\Desktop\\NLP\\NLP Project\\Special\\Final_testing_tweet.txt',"r").read()
##tuples3=document3.split('\n')
##for t3 in tuples3:
##        clean_tweets_testing.append(t3)
##              
##document=open('C:\\Users\\user\\Desktop\\NLP\\NLP Project\\Special\\SemEval2016-Task6-subtaskA-testdata-gold.txt',"r").read()
##tuples4=document.split('\n')
##real_labels=[]
##for t in tuples4:
##      real_labels.append(t.split('\t')[3])
##
##testing_tweets_with_labels=list(zip(clean_tweets_testing, real_labels))
##featuresets_test=[(find_features(tweets), stances) for (tweets, stances) in testing_tweets_with_labels]
##fea,real_label=zip(*featuresets_test)
##testing_features_atheism=featuresets_test[1:220]
##random.shuffle(testing_features_atheism)
##testing_features_climate=featuresets_test[221:389]
##random.shuffle(testing_features_climate)
##testing_features_feminist=featuresets_test[390:674]
##random.shuffle(testing_features_feminist)
##testing_features_hillary=featuresets_test[675:969]
##random.shuffle(testing_features_hillary)
##testing_features_abortion=featuresets_test[970:1249]
##random.shuffle(testing_features_abortion)
##print("MNB_classifier accuracy percentage for target Atheism:",(nltk.classify.accuracy(MNB_atheism,testing_features_atheism))*100)
##print("MNB_classifier accuracy percentage for target Climate Change is a Real Concern:",(nltk.classify.accuracy(MNB_climate,testing_features_climate))*100)
##print("MNB_classifier accuracy percentage for target Feminist Movement:",(nltk.classify.accuracy(MNB_feminist,testing_features_feminist))*100)
##print("MNB_classifier accuracy percentage for target Hillary Clinton:",(nltk.classify.accuracy(MNB_hillary,testing_features_hillary))*100)
##print("MNB_classifier accuracy percentage for Legalization of Abortion:",(nltk.classify.accuracy(MNB_abortion,testing_features_abortion))*100)
##features, labels=zip(*testing_features_atheism)
##pred_labels=MNB_atheism.classify_many(features)
##print(f1_score(labels, pred_labels, average='macro'))
