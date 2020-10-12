import pandas as pd
import pickle
import numpy as np
from scipy.sparse import data
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


# pandas read csv
df = pd.read_csv('shuffled-full-set-hashed.csv', names=['label', 'content'])
print(df.label.size)

#df.label.value_counts().plot(kind='bar', figsize=(16,9))

# train test split
X_train, X_test, y_train, y_test = train_test_split(df, df.label, test_size = 0.50, random_state = 0)

# transfer hashed text into numeric matrix
vectorizer = TfidfVectorizer()
tfidf1 = vectorizer.fit_transform(X_train.content.values.astype('U'))
print(tfidf1)

tfidf2 = vectorizer.transform(X_test.content.values.astype('U'))

# setup model, LogisticRegression()
model = LogisticRegression()
model.fit(tfidf1, y_train)

print(model.score(tfidf2, y_test))