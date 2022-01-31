#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
Created on Sat Jan  1 16:24:24 2022

Following a tutorial on sentiment analysis on towardsdatascience.com 
https://towardsdatascience.com/a-beginners-guide-to-sentiment-analysis-in
-python-95e354ea84f6

data set is 'Reviews.csv' a csv file containing amazon reviews

@author: marlc
"""


# Step 1: Importing the data set into a data frame and previewing data 

# In[2]:


import pandas as pd

dataFrame = pd.read_csv('Reviews.csv') #reading the csv data set onto data frame called dataFrame
dataFrame.head() #gets the first few rows of the csv 


# Importing packages: Ensure plotly package is installed

# In[3]:


# Imports
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
get_ipython().run_line_magic('matplotlib', 'inline')
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.express as px


# Step 2: Data analysis and inferences
# cell below plots a histogram of customer reviews, indicating overall reviews will be mostly positive since score > 5

# In[4]:


# Product Scores
fig = px.histogram(dataFrame, x="Score")
fig.update_traces(marker_color="turquoise",marker_line_color='rgb(8,48,107)',
                  marker_line_width=1.5)
fig.update_layout(title_text='Product Score')
fig.show()


# Creates a WordCloud of most used words in reviews. Indicates "Taste" "product" "love" etc. are most used words.
# These words are mostly positively connotated suggesting reviews are mostly positive
# Ensure wordcloud package is used. 

# In[5]:


import nltk
import wordcloud
from nltk.corpus import stopwords
from wordcloud import WordCloud, STOPWORDS
# Create stopword list:
stopwords = set(STOPWORDS)
stopwords.update(["br", "href"])
textt = " ".join(review for review in dataFrame.Text)
wordcloud = WordCloud(stopwords=stopwords).generate(textt)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.savefig('wordcloud11.png')
plt.show()


# Step 3: Classifying reviews into positive negative.
# This can be used as training data for sentiment classification model 
# Reviews with score >3 are positive +1 
# Reviews with score <3 are negative -1 
# Reviews with score = 3 are dropped and not be used in model 

# In[6]:


dataFrame = dataFrame[dataFrame['Score'] != 3] #removes data with score 3 from the data frame 
dataFrame['sentiment'] = dataFrame['Score'].apply(lambda rating : +1 if rating > 3 else -1) #creates a new column 'sentiment'
dataFrame.head()


#  Step 4: More data analysis: positive and negative word clouds 

# In[7]:


positive = dataFrame[dataFrame['sentiment'] == 1]
negative = dataFrame[dataFrame['sentiment'] == -1]


# In[8]:


#positive word cloud 
stopwords = set(STOPWORDS)
stopwords.update(["br", "href","good","great"]) 
## good and great removed because they were included in negative sentiment
pos = " ".join(review for review in positive.Summary)
wordcloud2 = WordCloud(stopwords=stopwords).generate(pos)
plt.imshow(wordcloud2, interpolation='bilinear')
plt.axis("off")
plt.show()


# In[9]:


#negative word cloud 
negative['Summary'] = (negative.Summary.astype(str)) # added to fix type error
neg = " ".join(review for review in negative.Summary)
wordcloud3 = WordCloud(stopwords=stopwords).generate(neg)
plt.imshow(wordcloud3, interpolation='bilinear')
plt.axis("off")
plt.savefig('wordcloud33.png')
plt.show()


# MORE DATA ANALYSIS: Distribution of positive reviews compared to negative reviews 

# In[13]:


dataFrame['sentimentt'] = dataFrame['sentiment'].replace({-1 : 'negative'})
dataFrame['sentimentt'] = dataFrame['sentimentt'].replace({1 : 'positive'})
fig = px.histogram(dataFrame, x="sentimentt")
fig.update_traces(marker_color="indianred",marker_line_color='rgb(8,48,107)',
                  marker_line_width=1.5)
fig.update_layout(title_text='Product Sentiment')
fig.show()


# In[14]:


dataFrame.head()


# Step 5: Building the Sentiment Analysis Model 

# In[17]:


#removes punctuation from data frame
def remove_punctuation(text):
    final = "".join(u for u in text if u not in ("?", ".", ";", ":",  "!",'"'))
    return final
dataFrame['Text'] = dataFrame['Text'].apply(remove_punctuation)
dataFrame = dataFrame.dropna(subset=['Summary'])
dataFrame['Summary'] = dataFrame['Summary'].apply(remove_punctuation)

#splits the data frame 
dfNew = dataFrame[['Summary','sentiment']]
dfNew.head()


# In[20]:


# random split train and test data
import numpy as np
index = dfNew.index
dfNew['random_number'] = np.random.randn(len(index))
train = dfNew[dfNew['random_number'] <= 0.8]
test = dfNew[dfNew['random_number'] > 0.8]


# Bag of Words Model NLP:
# Transform the text in our data frame into a bag of words model, which will contain a sparse matrix of integers. 
# The number of occurrences of each word will be counted and printed.
# We will need to convert the text into a bag-of-words model since the logistic regression algorithm cannot understand text.

# In[21]:


#creating a 'bag of words model' count vectorizer
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(token_pattern=r'\b\w+\b') #creates a 
train_matrix = vectorizer.fit_transform(train['Summary'])
test_matrix = vectorizer.transform(test['Summary'])


# In[22]:


# Importing Logistic Regression
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()


# In[23]:


#split target and independent variables 
X_train = train_matrix
X_test = test_matrix
y_train = train['sentiment']
y_test = test['sentiment']


# In[24]:


# Model Fitting on data
lr.fit(X_train,y_train)


# In[25]:


# predictions 
predictions = lr.predict(X_test)


# Step 6: Testing
# Confusion Matrix: a table that is often used to describe the performance of a classification model (or "classifier") on a set of test data for which the true values are known.

# In[27]:


# find accuracy, precision, recall:
from sklearn.metrics import confusion_matrix,classification_report
new = np.asarray(y_test)
confusion_matrix(predictions,y_test) #confusion matrix


# In[28]:


# classification report 
print(classification_report(predictions,y_test))


# In Summary the overall accuracy of the model is 93% without feature extraction or preprocessing.
# We imported a data set (csv of Amazon Reviews). Converted it to a data frame. Labled the each review as positive or negative based on score (score > 3 is positive,  score < 3 is negative, exclude =3). Split the labeled data for training (80%) and testing (20%). Used NLP 'Bag of Words Model' to be used in the logistic regression model (lgm). Next, lgm was fitted onto data and predicted. Lgm was trained and predicted and tested. A confusion matrix was created to describe the performance of classification model. 
# 
# Remarks: It is important to note, that there is a class imbalance (e.g. there are more positive reviews compared to negative reviews). We might be able to improve the accuracy of our model by balancing the positive and negative reviews counts. Future excercise can be to label neutral reviews. 
# 
# Things Learned: 
#     Data Analysis: Creating a word cloud 
#     ML: Sentiment Analysis/ Natural Language Processing
#         Labeling data as positive or negative
#         Splittig data for training and testing
#         Classifying data using simple logistic regression algorithm
#         Using Bag of Words Model NLP: text is represented as the bag of its words, disregarding grammar and even word order but               keeping multiplicity. Contain a sparse matrix of integers. The number of occurrences of each word will be counted               and printed.
#         Learned the term Confusion Matrix: a table that is often used to describe the performance of a classification model (or               "classifier") on a set of test data for which the true values are known.
#         
