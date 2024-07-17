""" An example of sentiment analysis
modify By Kruti Patel
Sam Scott, Mohawk College, 2021
"""
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from joblib import dump
### Load docs and labels
with open("topic-offtopic.txt") as file:
    docs = []
    labels = []
    for line in file:
        line = line.strip()
        labels.append(int(line[-1]))
        docs.append(line[:-2].strip())
    print(docs)
    print(labels)
for i in range(50):
    ## split into training and testing data

    split = train_test_split(docs, labels)
    train_docs, test_docs, train_labels, test_labels = split

    ## Vectorize

    topic_vectorizer = CountVectorizer()
    topic_vectorizer.fit(docs)
    vectors= topic_vectorizer.transform(docs)

    ## create and train the classifier


    topic_clf = MLPClassifier()
    topic_clf.fit(vectors,labels)



dump(topic_clf, 'classifier_topic.joblib')
dump(topic_vectorizer, 'vectorizer_topic.joblib')
print("created")