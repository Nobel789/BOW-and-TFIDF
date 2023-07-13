#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 15:03:25 2023

@author: myyntiimac
"""

import nltk

paragraph ="""Data Collection or Imputation Issues: An IQR of 0 and an upper bound of 0 could also indicate potential data collection or imputation issues. It's worth investigating the reason behind the lack of variability and ensuring the data is accurate and correctly represented.
Impact on Analysis: Consider the impact of removing the variable on your analysis or modeling goals. Removing a variable may alter the feature space, affect the performance of certain algorithms, or influence the interpretability of your results. Evaluate the potential consequences before making a decision.
In summary, if a variable has an IQR of 0 and an upper bound of 0, it indicates a lack of variability and redundancy. Removing such a variable can be a reasonable choice if it is deemed irrelevant or redundant to your analysis. However, consider the aforementioned factors to ensure the decision aligns with the specific context and requirements of your analysis."""


# Cleaning the texts
import re # re libray will use for regular expression 
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

ps = PorterStemmer()
wordnet=WordNetLemmatizer()
sentences = nltk.sent_tokenize(paragraph)

corpus = []


# Create the empty list name as corpus becuase after cleaned the data corpus will store this clean data

for i in range(len(sentences)):
    review = re.sub('[^a-zA-Z]', ' ', sentences[i])
    review = review.lower()
    review = review.split()
#   review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = [wordnet.lemmatize(word) for word in review if not word in set(stopwords.words('english'))]   
    review = ' '.join(review)
    corpus.append(review)


# Creating the Bag of Words model 

# Also we called as document matrix 
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(corpus).toarray()

from sklearn.feature_extraction.text import TfidfVectorizer
cv_1 = TfidfVectorizer()
X_tf= cv_1.fit_transform(corpus).toarray()