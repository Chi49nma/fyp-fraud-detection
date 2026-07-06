from flask import Flask, render_template, request
from predictor import predict_transaction

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    try:

        amount = float(request.form["amount"])
        bank = request.form["bank"]
        channel = request.form["channel"]
        merchant = request.form["merchant_category"]
        location = request.form["location"]
        age_group = request.form["age_group"]
        transaction_time = request.form["transaction_time"]

        prediction = predict_transaction(
            amount=amount,
            bank=bank,
            channel=channel,
            merchant_category=merchant,
            location=location,
            age_group=age_group,
            transaction_time=transaction_time
        )

        return render_template(
            "result.html",
            amount=amount,
            bank=bank,
            channel=channel,
            merchant=merchant,
            location=location,
            age_group=age_group,
            transaction_time=transaction_time,

            decision=prediction["decision"],
            probability=prediction["fraud_probability"],
            rule_flag=prediction["rule_flag"],
            behaviour_score=prediction["behaviour_score"],
            behaviour_flag=prediction["behaviour_flag"]
        )

    except Exception as e:
        return f"""
        <h2>Error</h2>
        <pre>{e}</pre>
        """


if __name__ == "__main__":
    app.run(debug=True)