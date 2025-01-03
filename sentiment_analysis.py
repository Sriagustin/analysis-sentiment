import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report

def preprocess_data(data):
    data['Text Tweet'] = data['Text Tweet'].str.lower()
    data['Text Tweet'] = data['Text Tweet'].str.replace(r'[^a-zA-Z\s]', '', regex=True)
    data['Text Tweet'] = data['Text Tweet'].str.replace('\d', '', regex=True)
    data['Text Tweet'] = data['Text Tweet'].str.replace('[^\w\s]', '', regex=True)
    data['Text Tweet'] = data['Text Tweet'].str.replace('_', '')
    data['Text Tweet'] = data['Text Tweet'].str.replace('usermention', '')
    data['Text Tweet'] = data['Text Tweet'].str.replace('providername', '')
    data['Text Tweet'] = data['Text Tweet'].str.replace('productname', '')
    data['Text Tweet'] = data['Text Tweet'].str.replace('url', '')
    return data

def train_sentiment_model(data):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(data['Text Tweet'])
    y = data['Sentiment']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = MultinomialNB()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    
    return model, vectorizer

def predict_sentiment(model, vectorizer, text):
    text_vector = vectorizer.transform([text])
    return model.predict(text_vector)[0]
