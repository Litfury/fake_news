import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Load saved model and vectorizer
with open("model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("vectorizer.pkl", "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

def predict_fake_news(input_text):
    """Predict if the given news is real or fake."""
    cleaned_text = input_text.lower()  # Simplified preprocessing
    vectorized_text = vectorizer.transform([cleaned_text])  # Vectorize text using the trained vectorizer
    prediction = model.predict(vectorized_text)  # Predict using the model
    return "Real" if prediction[0] == 1 else "Fake"  # Return prediction

# Streamlit app UI
st.set_page_config(page_title="Fake News Detection", page_icon="📰")
st.title("Fake News Detection App 📰")
st.write("Enter a news headline or article content to check if it's real or fake.")

# User input for news text
user_input = st.text_area("Enter Text:", height=200)

# Button to check the news
if st.button("Check", key="check_button"):
    if user_input:
        result = predict_fake_news(user_input)
        st.markdown(f"**Prediction**: The news is predicted to be: **{result}**")
    else:
        st.warning("Please enter some text to check!")
