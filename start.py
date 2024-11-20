# Import Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import re
import string
import pickle

# Text Preprocessing Function
def preprocess_text(text):
    """Clean and tokenize text."""
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    words = text.split()  # Split text into words
    return " ".join(words)

# Load Dataset
df = pd.read_csv("news.csv")  # Replace with your dataset file path
df.dropna(inplace=True)  # Remove rows with missing values
df['clean_text'] = df['text'].apply(preprocess_text)  # Apply text preprocessing

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    df['clean_text'], 
    df['label'].map({'fake': 0, 'real': 1}),  # Map labels to binary
    test_size=0.2, 
    random_state=42
)

# Feature Extraction
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Define and Train Classifiers
classifiers = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42)
}

for name, model in classifiers.items():
    model.fit(X_train_tfidf, y_train)
    predictions = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, predictions)
    print(f"{name} Accuracy: {accuracy:.2f}")
    print(f"{name} Classification Report:\n{classification_report(y_test, predictions)}\n")

# Save the Best Model and Vectorizer
best_model = LogisticRegression(max_iter=1000)
best_model.fit(X_train_tfidf, y_train)
with open("model.pkl", "wb") as model_file:
    pickle.dump(best_model, model_file)
with open("vectorizer.pkl", "wb") as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

# Load and Test the Saved Model
with open("model.pkl", "rb") as model_file:
    loaded_model = pickle.load(model_file)
with open("vectorizer.pkl", "rb") as vectorizer_file:
    loaded_vectorizer = pickle.load(vectorizer_file)

new_texts = ["Breaking news: Aliens have landed!", "The government announces a new tax policy."]
new_texts_cleaned = [preprocess_text(text) for text in new_texts]
new_texts_tfidf = loaded_vectorizer.transform(new_texts_cleaned)
predictions = loaded_model.predict(new_texts_tfidf)

for i, text in enumerate(new_texts):
    print(f"Text: {text}\nPrediction: {'Real' if predictions[i] == 1 else 'Fake'}\n")
