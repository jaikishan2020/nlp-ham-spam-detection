import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

df= pd.read_csv("spam_data.csv", encoding="latin-1")

# Features and Labels

df['label'] = df['class'].map({'ham': 0, 'spam': 1})
X = df['message']
y = df['label']
y_dist=y.value_counts()

print ("The distribution of given data set is :\n",y_dist)

# Extract Feature With CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(X) # Fit the Data

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

#Naive Bayes Classifier
clf = MultinomialNB()
clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)
y_values=y_train.unique()

print(confusion_matrix(y_pred=y_pred, y_true=y_test,labels=y_values))
print ('The Accuracy of the NB classifier model formed is :{0}'. format(accuracy_score(y_pred=y_pred, y_true=y_test)))
print(classification_report(y_true=y_test,y_pred= y_pred))

print(" Saving the transform and model as a pickle file")
pickle.dump(cv,open('transform.pkl','wb'))
file_name='nlp_model.pkl'
pickle.dump(clf,open(file_name,'wb'))
print(" Saved the transform and model as a pickle file to use in the predict")


