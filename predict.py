from flask import Flask
from flask import request
from flask import jsonify
import pickle

model_file = "ridge_model_alpha=0.1.bin"

app = Flask("predict")

with open(model_file, "rb") as f_in:
    dv, model = pickle.load(f_in)

@app.route("/predict", methods=["POST"])
def predict():
    listing = request.get_json()
    X = dv.transform([listing])
    y_pred = model.predict(X)
    result = {
        "selling_price_prediction": float(y_pred)
    }
    return jsonify(result)

if __name__=="__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)