import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from collections import Counter

from sklearn import feature_extraction, model_selection
from nltk.corpus import stopwords

sw = stopwords.words("english")

# Exploring the Dataset
data = pd.read_csv('spam.csv', encoding='latin-1')
print(data.head(n=10))

# Distribution spam/non-spam plots
count_Class = pd.value_counts(data["v1"], sort=True)
count_Class.plot(kind='bar', color=["blue", "orange"])
plt.title('Bar chart')
plt.show()

count_Class.plot(kind='pie', autopct='%1.0f%%')
plt.title('Pie chart')
plt.ylabel('')
plt.show()

# We want to find the frequencies of words in the spam and non-spam messages. The words of the messages will be model features.
count1 = Counter(" ".join(data[data['v1'] == 'ham']["v2"]).split()).most_common(20)
df1 = pd.DataFrame.from_dict(count1)
df1 = df1.rename(columns={0: "words in non-spam", 1: "count"})
count2 = Counter(" ".join(data[data['v1'] == 'spam']["v2"]).split()).most_common(20)
df2 = pd.DataFrame.from_dict(count2)
df2 = df2.rename(columns={0: "words in spam", 1: "count_"})

df1.plot.bar(legend=False)
y_pos = np.arange(len(df1["words in non-spam"]))
plt.xticks(y_pos, df1["words in non-spam"])
plt.title('More frequent words in non-spam messages')
plt.xlabel('words')
plt.ylabel('number')
plt.show()

df2.plot.bar(legend=False, color='orange')
y_pos = np.arange(len(df2["words in spam"]))
plt.xticks(y_pos, df2["words in spam"])
plt.title('More frequent words in spam messages')
plt.xlabel('words')
plt.ylabel('number')
plt.show()
# We can see that the majority of frequent words in both classes are stop words such as 'to', 'a', 'or' and so on.
# With stop words we refer to the most common words in a language, there is no simgle, universal list of stop words.


# Text preprocessing, tokenizing and filtering of stopwords are included in a high level component that is able to build a dictionary of features and transform documents to feature vectors.
# We remove the stop words in order to improve the analytics
sw = stopwords.words("english")

vectorizer = feature_extraction.text.TfidfVectorizer(stop_words=sw, lowercase=True, )
vectorizer.fit(data["v2"])
X = vectorizer.transform(data["v2"]).toarray()
print(X.shape)
# We have created more than 8500 new features.


# My goal is to predict if a new sms is spam or non-spam.
# I assume that is much worse misclassify non-spam than misclassify an spam.
# (I don't want to have false positives)
#
# The reason is because I normally don't check the spam messages.
#
# The two possible situations are:
#
# New spam sms in my inbox. (False negative).
# OUTCOME: I delete it.
#
# New non-spam sms in my spam folder (False positive).
# OUTCOME: I probably don't read it.
#
# I prefer the first option!!!


# First we transform the variable spam/non-spam into binary variable, then we split our data set in training set and test set.
data["v1"] = data["v1"].map({'spam': 1, 'ham': 0})
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, data['v1'], test_size=0.33, random_state=42)
print([np.shape(X_train), np.shape(X_test)])
print(y_test)

dataset = [X_train, X_test, y_train, y_test]

# save dataset to use later
with open('dataset.pkl', 'wb') as output:
    pickle.dump(dataset, output)

# save vectorizer to use later
with open('vectorizer.pkl', 'wb') as output:
    pickle.dump(vectorizer, output)
