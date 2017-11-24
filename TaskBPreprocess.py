import csv
import string
import re



tweet1 =[]
with open("taskB_corpus.txt", 'r', encoding="utf8") as f:
    reader = csv.reader(f, delimiter = '\n')
    for row in reader:
        tweet1.append(row[0])


Clean_tweet= open('CleanedB1.txt', 'w',  encoding="utf8")
for j in tweet1:
    result = ' '.join(re.sub("(http\S+)|(Not Availa\S+)|(,)"," ",j).split())
    Clean_tweet.write(result)
    Clean_tweet.write('\n')
Clean_tweet.close()





