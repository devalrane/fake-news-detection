import streamlit as st
import joblib
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from gensim.models import Word2Vec
import pickle


nltk.data.path.append("nltk_data")
# Load your pre-trained Word2Vec model and logistic regression model
# word2vec_model = joblib.load("w2v_model.pkl")
word2vec_model = Word2Vec.load("w2v_model_gensim")
# logistic_model = joblib.load("logistic_model.pkl")
with open("logistic_model_pickle.pkl", "rb") as f:
    logistic_model = pickle.load(f)


def preprocess_text(text):
    if not isinstance(text, str):
        raise ValueError("Input text must be a string")
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [
        token for token in tokens if token.isalpha()
    ]  # Removing non-alphabetic tokens
    tokens = [token for token in tokens if token not in stopwords.words("english")]

    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in tokens]

    # Convert tokens to Word2Vec embeddings
    vectorized_tokens = [
        word2vec_model.wv[token]
        for token in stemmed_tokens
        if token in word2vec_model.wv
    ]

    return vectorized_tokens


def predict_fake_news(text):
    # Preprocess the text to get Word2Vec embeddings
    preprocessed_text = preprocess_text(text)

    # Convert the list of Word2Vec embeddings to a suitable format for the model
    if len(preprocessed_text) > 0:
        average_embedding = np.mean(preprocessed_text, axis=0)
    else:
        # Handling the case where the preprocessed text might be empty
        average_embedding = np.zeros(word2vec_model.vector_size)

    # Reshape the embedding to match the model's expected input format
    average_embedding = average_embedding.reshape(1, -1)

    # Make a prediction
    prediction = logistic_model.predict(average_embedding)

    # return "Fake" if prediction == 0 else "Real"

    probabilities = logistic_model.predict_proba(average_embedding)

    # Extract probabilities for both classes
    fake_probability = probabilities[0][0] * 100  # Convert to percentage
    real_probability = probabilities[0][1] * 100  # Convert to percentage

    return fake_probability, real_probability


st.title("Fake News Detection")
user_input = st.text_area("Enter Text", "Type your news content here...")
# print("Type of user_input:", type(user_input))  # Debugging line


if st.button("Predict"):
    # preprocessed_text = preprocess_text(user_input)
    # prediction = predict_fake_news(user_input)
    # st.write(f"The news is predicted as: {prediction}")
    fake_prob, real_prob = predict_fake_news(user_input)
    st.write(f"The probability of the news being fake is: {fake_prob:.2f}%")
    st.write(f"The probability of the news being real is: {real_prob:.2f}%")
