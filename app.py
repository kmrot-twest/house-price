from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# load model and scaler
model = joblib.load("knn_model.pkl")
scaler = joblib.load("scaler.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    try:
        # read values safely from form
        f1 = float(request.form["f1"])
        f2 = float(request.form["f2"])
        f3 = float(request.form["f3"])
        f4 = float(request.form["f4"])

        # checkbox handling (IMPORTANT)
        f5 = request.form.get("f5")
        f5 = 1 if f5 == "1" else 0

        # combine features in correct order
        data = np.array([[f1, f2, f3, f4, f5]])

        # scale input
        data_scaled = scaler.transform(data)

        # prediction
        prediction = model.predict(data_scaled)[0]

        return render_template(
            "index.html",
            prediction=f"Predicted Price: {prediction:.2f}"
        )

    except Exception as e:
        return render_template("index.html", prediction=f"Error: {str(e)}")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)