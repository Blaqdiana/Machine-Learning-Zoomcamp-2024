import pickle
from flask import Flask, request, jsonify

# Load the DictVectorizer and model (assuming they are in the same directory)
with open('dv.bin', 'rb') as f_dv:
    dv = pickle.load(f_dv)

with open('model1.bin', 'rb') as f_model:
    model = pickle.load(f_model)

app = Flask('deposit')

@app.route('/predict', methods=['POST'])
def predict():
    # json = Python dictionary
    client = request.get_json()

    X = dv.transform([client])
    y_pred = model.predict_proba(X)[0, 1]
    deposit = y_pred >= 0.5

    result = {
        'deposit_probability': float(y_pred),
        'deposit': bool(deposit)
    }
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)