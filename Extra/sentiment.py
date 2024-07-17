""" Loads the sentiment analysis documents and labels
train the classifier and after it is train.Pickle it to save loading time.
modify By Kruti Patel
Sam Scott, Mohawk College, 2021
"""
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier

### Load docs and labels
filenames = ["amazon_cells_labelled.txt", "imdb_labelled.txt", "yelp_labelled.txt"]
docs = []
labels = []
for filename in filenames:
    with open("sentiment/"+filename) as file:
        for line in file:
            line = line.strip()
            labels.append(int(line[-1]))
            docs.append(line[:-2].strip())
print(labels)

## vectorize
vectorizer = CountVectorizer()
vectorizer.fit(docs)
vectors= vectorizer.transform(docs)


## train classifier
clf = DecisionTreeClassifier(max_depth=126, criterion="entropy")
#clf.fit(vectors,labels)
##pickling
from joblib import dump
dump(clf, 'classifier.joblib')
dump(vectorizer, 'vectorizer.joblib')
print("created")


