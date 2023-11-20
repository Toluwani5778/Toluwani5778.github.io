import streamlit as st
from keras.preprocessing.sequence import pad_sequences
import time
from keras.preprocessing.text import Tokenizer
from keras.models import load_model
import numpy as np
import pandas as pd
from PIL import Image
import re

#Tweet Sentiment Data
tweets = pd.read_csv('Tweets.csv')
features =tweets.iloc[:, 10].values
processed_features = []

for sentence in range(0, len(features)):
    # Remove all the special characters
    processed_feature = re.sub(r'\W', ' ', str(features[sentence]))

    # remove all single characters
    processed_feature= re.sub(r'\s+[a-zA-Z]\s+', ' ', processed_feature)

    # Remove single characters from the start
    processed_feature = re.sub(r'\^[a-zA-Z]\s+', ' ', processed_feature)

    # Substituting multiple spaces with single space
    processed_feature = re.sub(r'\s+', ' ', processed_feature, flags=re.I)

    # Removing prefixed 'b'
    processed_feature = re.sub(r'^b\s+', '', processed_feature)

    # Converting to Lowercase
    processed_feature = processed_feature.lower()

    processed_features.append(processed_feature)

processed_features=np.array(processed_features)
tokenizer = Tokenizer()
tokenizer.fit_on_texts(processed_features)
sequences = tokenizer.texts_to_sequences(processed_features)

model = load_model("Tweet_detection.h5", compile=False)


# Streamlit Side
st.title("Sentiment Analysis with Deep Learning: Uncover Emotions from Text😊☹")
text = "This Streamlit application is your gateway to the fascinating world of sentiment analysis. Have you ever wondered how to extract emotions and sentiments from text? Well, wonder no more! Our deep learning model is here to help you decipher the sentiments behind the words. Whether it's a heartwarming message, a scathing review, or anything in between, our application will provide you with valuable insights into the emotions hidden within the text. Simply paste or type your text, and let's dive into the world of sentiment analysis together."
st.write(text)

img = Image.open('sent_analys.jpg')
st.image(img, use_column_width=True)

imge = Image.open('nlp cloud.jpg')
st.image(imge, use_column_width=True)

st.subheader("Welcome to the Sentiment Analyzer")
tweet = st.text_input("Please input your text: \n")


def predict_senti():
    my_tweet = [tweet]
    my_tweet_seq = tokenizer.texts_to_sequences(my_tweet)
    my_tweet_pad = pad_sequences(my_tweet_seq, maxlen=33)
    predict = model.predict(my_tweet_pad)
    if predict.argmax()==2:
        st.success("Your text is Positive 😊")
    elif predict.argmax()==1:
        st.warning("Your text is Neutral 😐")
    elif predict.argmax()==0:
        st.error("Your text is Negative ☹")


st.button("Predict", on_click = predict_senti)


