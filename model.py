import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


# pandas read csv
names = ['label', 'content']
dataframe = pd.read_csv('shuffled-full-set-hashed.csv', names=names)


print(dataframe.label.size)
