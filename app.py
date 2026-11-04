from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load the trained model
with open("spam_model.pkl", "rb") as file:
    model = pickle.load(file)

# Load the TF-IDF vectorizer
with open("vectorizer.pkl", "rb") as file:
    vectorizer = pickle.load(file)


@app.route("/", methods=["GET", "POST"])
def home():

    result = None
    message = ""

    if request.method == "POST":

        message = request.form.get("message", "")

        if message.strip() == "":
            result = "Please enter a message."

        else:
            message_vector = vectorizer.transform([message])
            prediction = model.predict(message_vector)

            if prediction[0] == 1:
                result = "SPAM"
            else:
                result = "NOT SPAM"

    return render_template(
        "index.html",
        result=result,
        message=message
    )


if __name__ == "__main__":
    app.run(debug=True)