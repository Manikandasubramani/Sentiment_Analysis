# importing the libraries
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

import re
import string
import pickle
from nltk.corpus import stopwords

sw = stopwords.words('english')

# positive dataset
#fillig 1's for positive mood
pos_rev = pd.read_csv('../dataset/pos.txt', sep='\n', header = None, encoding='latin-1')
pos_rev = pd.concat([pos_rev, pd.Series(np.ones(pos_rev.shape[0]))], ignore_index=True, axis=1)
pos_rev.columns = ['review', 'mood']

# negative dataset
# filling 0's for negative mood
neg_rev = pd.read_csv("../dataset/negative.txt", sep = "\n", header = None, encoding = 'latin-1')
neg_rev = pd.concat([neg_rev,pd.Series(np.zeros(pos_rev.shape[0]))], ignore_index=True, axis =1)
neg_rev.columns = ['review', 'mood']

# processing positive comments
# converting the review to lowercase
# subsituting anything with "@" continued with string values to null
# removing punctuations
# removing stopwords
pos_rev.loc[:, 'review'] = pos_rev.loc[:, 'review'].apply(lambda x: x.lower())
pos_rev.loc[:, 'review'] = pos_rev.loc[:, 'review'].apply(lambda x: re.sub(r"@\S+", "", x))
pos_rev.loc[:, 'review'] = pos_rev.loc[:, 'review'].apply(lambda x: x.translate(str.maketrans(dict.fromkeys(string.punctuation))))
pos_rev.loc[:, 'review'] = pos_rev.loc[:, 'review'].apply(lambda x:' '.join([word for word in x.split() if word not in (sw)]))

# processing negative comments
# converting the review to lowercase
# subsituting anything with "@" continued with string values to null
# removing punctuations
# removing stopwords
neg_rev.loc[:, 'review'] = neg_rev.loc[:, 'review'].apply(lambda x: x.lower())
neg_rev.loc[:, 'review'] = neg_rev.loc[:, 'review'].apply(lambda x: re.sub(r"@\S+", "", x))
neg_rev.loc[:, 'review'] = neg_rev.loc[:, 'review'].apply(lambda x: x.translate(str.maketrans(dict.fromkeys(string.punctuation))))
neg_rev.loc[:, 'review'] = neg_rev.loc[:, 'review'].apply(lambda x:' '.join([word for word in x.split() if word not in (sw)]))

# connecting both positive and negative reviews
com_rev = pd.concat([pos_rev, neg_rev], axis =0).reset_index()

# train and test split with comapred review's "review" and "mood" data
X_train, X_test, Y_train, Y_test = train_test_split(com_rev['review'].values,com_rev['mood'].values, test_size = 0.32, random_state = 42)

train_data = pd.DataFrame({'review':X_train, 'mood':Y_train})
test_data = pd.DataFrame({'review':X_test, 'mood':Y_test})

# initializing the TFIDF vectorizer
vectorizer = TfidfVectorizer()

train_vectors = vectorizer.fit_transform(train_data['review'])
test_vectors = vectorizer.transform(test_data['review'])

# NaiveBayes classification
classifier = MultinomialNB()
classifier.fit(train_vectors, train_data['mood'])
prediction = classifier.predict(test_vectors)

print(classifier.score(test_vectors,test_data['mood']))

#classification report
report = classification_report(test_data['mood'], prediction, output_dict=True)
print('positive:', report['1.0']['recall'])
print('negative:', report['0.0']['recall'])

pickle.dump(vectorizer, open('../models/tranform.pkl', 'wb'))
pickle.dump(classifier, open('../models/model.pkl', 'wb'))


