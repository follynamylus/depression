import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import string
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
# nltk.download("punkt")
# nltk.download("stopwords")
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
import streamlit as st

wnet = WordNetLemmatizer()

# Load your trained model
model = joblib.load('random_forest_model.pkl')


def predict(text):
    # Perform any necessary preprocessing on the text_input if needed

    # Make a prediction using the loaded model
    prediction = model.predict(text)

    return prediction[0]

def main():
    st.title("Text Prediction App")

    # Create a text input box
    text_input = st.text_area("Enter text:", "")
    df = pd.DataFrame()
    df['text'] = [text_input]

    if st.button("Predict"):
        if text_input:
            def convert_lowercase(text):
                text = text.lower()
                return text
            df['text'] = df['text'].apply(convert_lowercase)

            def remove_url(text):
                re_url = re.compile('https?://\S+|www\.\S+')
                return re_url.sub('', text)
            df['text'] = df['text'].apply(remove_url)

            exclude = string.punctuation
            def remove_punc(text):
                return text.translate(str.maketrans('', '', exclude))
            df['text'] = df['text'].apply(remove_punc)

            def remove_stopwords(text):
                new_list = []
                words = word_tokenize(text)
                stopwrds = stopwords.words('english')
                for word in words:
                    if word not in stopwrds:
                        new_list.append(word)
                return ' '.join(new_list)

            df['text'] = df['text'].apply(remove_stopwords)

            def perform_stemming(text):
                stemmer = PorterStemmer()
                new_list = []
                words = word_tokenize(text)
                for word in words:
                    new_list.append(stemmer.stem(word))
                return " ".join(new_list)
            df['text'] = df['text'].apply(perform_stemming)
            tout = df['text']

            tfidf = TfidfVectorizer(max_features= 2500, min_df= 1, max_df= 5)
            text = tfidf.fit_transform(tout).toarray()
            #text = text.reshape((text.shape[0], -1))
            zeros = [i for i in range(2500)]
            DF = []
            for i in zeros:
                if i in text:
                    if i != 0:
                        DF.append(i)
                else:
                    DF.append(0)
            new_df = pd.DataFrame([DF])
            new = new_df.values
            X_flattened = new.reshape((new.shape[0], -1))
            st.write(X_flattened.shape)
            st.write(X_flattened.ndim)
            
            prediction = predict(X_flattened)
            if prediction == 0.0 :
                st.write("The Patient is not Depressed")
            else :
                st.write("The Patient is depressed")
        else:
            st.warning("Please enter some text.")

if __name__ == '__main__':
    main()
