import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sentiment_analysis import preprocess_data, train_sentiment_model, predict_sentiment

# Load dataset
st.title("Sentiment Analysis Dashboard")
uploaded_file = st.file_uploader("upload your dataset (CSV)", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview")
    st.dataframe(data.head())

    # Preprocess data
    data = preprocess_data(data)

    # Train sentiment model
    model, vectorizer = train_sentiment_model(data)

    # WordCloud for each sentiment
    sentiments = data['Sentiment'].unique()
    for sentiment in sentiments:
        st.write(f"### WordCloud for {sentiment} Sentiment")
        sentiment_data = data[data['Sentiment'] == sentiment]
        text = " ".join(sentiment_data['Text Tweet'])
        
        wordcloud = WordCloud(background_color="white",
                              mask=plt.imread("assets/twitter_logo.png")).generate(text)
        
        plt.figure(figsize=(8, 8))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(plt)

    # Data distribution
    st.write("### Sentiment Distribution (Pie Chart)")
    sentiment_counts = data['Sentiment'].value_counts()
    fig, ax = plt.subplots()
    ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    st.pyplot(fig)

    # Input for real-time prediction
    st.write("### Real-Time Sentiment Prediction")
    user_input = st.text_input("Enter text to analyze sentiment:")
    if user_input:
        sentiment = predict_sentiment(model, vectorizer, user_input)
        st.write(f"Predicted Sentiment: **{sentiment}**")