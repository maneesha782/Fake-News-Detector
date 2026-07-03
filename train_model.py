
# Import required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
import joblib
import os

# Step 1: Load datasets
fake_df = pd.read_csv("dataset/archive/Fake.csv")
true_df = pd.read_csv("dataset/archive/True.csv")

# Step 2: Add labels
fake_df["label"] = 0   # Fake news
true_df["label"] = 1   # Real news

# Print one fake and one real news article
print("\n===== FAKE NEWS =====")
print(fake_df["text"].iloc[5])

print("\n===== REAL NEWS =====")
print(true_df["text"].iloc[5])

# Step 3: Combine datasets
df = pd.concat([fake_df, true_df])
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
print(df.head())
print(df["label"].value_counts())

# Step 4: Select important columns
df["content"] = df["title"] + " " + df["text"]
df = df[["content", "label"]]

X = df["content"]

# Step 5: Remove missing values
df = df.dropna()

# Step 6: Split data into training and testing
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 7: Convert text into numbers
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Step 8: Train model
model = PassiveAggressiveClassifier(max_iter=1000, random_state=42)
model.fit(X_train_vec, y_train)
from sklearn.metrics import accuracy_score

y_pred = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, y_pred))


# Step 9: Save model and vectorizer
os.makedirs("model", exist_ok=True)

joblib.dump(model, "model/fake_news_model.pkl")
joblib.dump(vectorizer, "model/vectorizer.pkl")

print("Model training completed successfully!")