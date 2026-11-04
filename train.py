import pandas as pd
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("spam.csv", encoding="latin-1")

# Select columns
data = data[["v1", "v2"]]
data.columns = ["label", "message"]

# Convert labels
data["label"] = data["label"].map({
    "ham": 0,
    "spam": 1
})

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    data["message"],
    data["label"],
    test_size=0.2,
    random_state=42
)

# Convert text to numbers
vectorizer = TfidfVectorizer(stop_words="english")

X_train_vector = vectorizer.fit_transform(X_train)
X_test_vector = vectorizer.transform(X_test)

# Create and train model
model = MultinomialNB()
model.fit(X_train_vector, y_train)

# Test model
prediction = model.predict(X_test_vector)

accuracy = accuracy_score(y_test, prediction)

print("===================================")
print("   EMAIL SPAM DETECTION MODEL")
print("===================================")
print("Model trained successfully!")
print("Accuracy:", round(accuracy * 100, 2), "%")

# Save model
with open("spam_model.pkl", "wb") as file:
    pickle.dump(model, file)

# Save vectorizer
with open("vectorizer.pkl", "wb") as file:
    pickle.dump(vectorizer, file)

print("Model saved successfully!")