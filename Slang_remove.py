import csv
import string
from afinn import Afinn
afinn = Afinn()

'''A function to map slangs with their correct forms'''
'''Input: Slangs.csv'''
'''Output: Dictionary with [slang,correct form]'''
def Slang_Remove(filename, Slang, Meaning):
    with open(filename, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            Slang.append(row[0])
            Meaning.append(row[1])

    Slang = [x.strip(' ') for x in Slang]
    Meaning = [y.strip(' ') for y in Meaning]
    Slang_Vocab = dict(zip(Slang,Meaning))
    return(Slang_Vocab)

Slang =[]
Meaning =[]
Slang_vocab = Slang_Remove('Slangs.csv',Slang,Meaning)


"""Tweet filter is used for separating tweet_id, tweet, target and label from the trial data"""

def tweet_filter(filename,tweet_id,tweet):
    with open(filename, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            tweet_id.append(row[0])
            tweet.append(row[1].lower)
    tweet_id = [a.strip(' ') for a in tweet_id]
    tweet = [b.strip(' ') for b in tweet]

tweet_id = []
tweet = []
tweet_filter('TaskB_trial.csv', tweet_id, tweet)

"""tweet filter ends"""

"""Collecting Hashtags and removing them from the tweets"""

Hashtags = open("Hashtags.txt", 'w')
Corpus = open("Corpus.txt", 'w')
Id_Hash = open("Id_Hash.txt", 'w')

l = 0
for j in tweet:
    l = l+1
    for k in j.split():
        if k[0]=='#':
            Id_Hash.write(str(l))
            Id_Hash.write(',')
            Id_Hash.write(str(k))
            Id_Hash.write('\n')
            Hashtags.write(k + '\n')
        else:
            Corpus.write(k + ' ')
    Corpus.write('\n')

Hashtags.close()
Corpus.close()
Id_Hash.close()

""" Collecting Hashtag ends """

""" Reading a list of tweets with no hashtags """
tweet4 =[]
with open("Corpus.txt", 'r') as f:
    reader = csv.reader(f, delimiter = '\n')
    for row in reader:
        tweet4.append(row[0])
""" List created """


""" Removing words with '@' characters """
Clean_tweet = open('Clean_tweet.txt', 'w')
for j in tweet4:
    for k in j.split():
        if k[0]!='@':
            Clean_tweet.write(k + ' ')
        else:
            continue
    Clean_tweet.write('\n')
Clean_tweet.close()
""" Removed """

""" Creating new list of tweets to remove punctuations """
tweet2 = []
with open("Clean_tweet.txt", 'r') as f:
    reader = csv.reader(f, delimiter = '\n')
    for row in reader:
        tweet2.append(row[0])
""" List created """

""" Removing Punctuations from the tweet"""
Correct_tweet = open('Correct_tweet.txt', 'w')
def remove_punctuation(tweet):
    for d in tweet:
        exclude = set(string.punctuation)
        str1 = ''.join(ch for ch in d if ch not in exclude)
        Correct_tweet.write(str1 + '\n')

remove_punctuation(tweet2)
Correct_tweet.close()

""" Punctuations removed """

""" Creating new list of tweets to remove slangs """
tweet3 = []
with open("Correct_tweet.txt", 'r') as f:
    reader = csv.reader(f, delimiter = '\n')
    for row in reader:
        tweet3.append(row[0])

""" List created """


"""Removing Slangs"""
Final_tweet = open('Final_tweet.txt', 'w')
for s in tweet3:
    for w in s.split():
        for key, value in Slang_vocab.items():
            if w == key:
                w = value
                Final_tweet.write(' ')
        Final_tweet.write(w + ' ')
    Final_tweet.write('\n')

"""Slangs Removed"""
Final_tweet.close()
""" Creating new list of tweets to remove slangs """
tweet5 = []
with open("Final_tweet.txt", 'r') as f:
    reader = csv.reader(f, delimiter = '\n')
    for row in reader:
        tweet5.append(row[0])


'''Computing sentiments of each tweet'''
filename = "Final_tweet.txt"
scored_sentences = []
with open(filename, 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        scored_sentences.append(str((afinn.score(row[0]))))
print(scored_sentences)

'''End of sentiment'''

tweet_label = open('final_senti.txt', 'w')
for f in zip(tweet5,scored_sentences):
    tweet_label.write(f[0] + ',' + f[1])
    tweet_label.write('\n')
""" List created """

Corpus.close()
Clean_tweet.close()
Correct_tweet.close()
Final_tweet.close()
tweet_label.close()