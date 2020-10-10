import pandas as pd
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


# pandas read csv
df = pd.read_csv('shuffled-full-set-hashed.csv', names=['label', 'content'])


#print(df.label.size)
#df.label.value_counts().plot(kind='bar', figsize=(16,9))