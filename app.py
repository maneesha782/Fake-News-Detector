from flask import Flask, render_template, request
import joblib
import requests

NEWS_API_KEY = "a95cd4c1c3394412a1eec27f72795591"

def get_news(query):
    url = (
        f"https://newsapi.org/v2/everything?"
        f"q={query}&language=en&sortBy=publishedAt&pageSize=5"
        f"&apiKey={NEWS_API_KEY}"
    )

    try:
        response = requests.get(url)
        data = response.json()

        if data["status"] == "ok":
            return data["articles"]

        return []

    except Exception:
        return []


app = Flask(__name__)

# Load model and vectorizer
model = joblib.load("model/fake_news_model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/search_news", methods=["POST"])
def search_news():

    location = request.form.get("location")
    custom_location = request.form.get("custom_location")

    # If user typed a location, use that
    if custom_location:
        location = custom_location

    local_news = get_news(location)

    return render_template(
        "index.html",
        local_news=local_news,
        location=location
    )


@app.route("/predict", methods=["POST"])
def predict():
    news = request.form["news"]

    news_vector = vectorizer.transform([news])
    prediction = model.predict(news_vector)

    # Confidence Score
    confidence = abs(model.decision_function(news_vector)[0])
    confidence = min(confidence * 20, 99.99)

    if prediction[0] == 0:
        result = "🛑 Fake News"
        color = "fake"
        message = "⚠ Please verify this news before sharing."
    else:
        result = "✅ Real News"
        color = "real"
        message = "✔ This article appears to be authentic."


    return render_template(
        "index.html",
        prediction=result,
        confidence=round(confidence, 2),
        color=color,
        message=message
    )

if __name__ == "__main__":
    app.run(debug=True)