from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib

# Load pre-trained model and preprocessing objects
model = joblib.load('weather_model.pkl')
scaler = joblib.load('scaler.pkl')
imputer = joblib.load('imputer.pkl')
feature_names = joblib.load('features.pkl')

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        try:
            input_data = [float(request.form.get(col, 0)) for col in feature_names[:16]]
            df_input = pd.DataFrame([dict(zip(feature_names[:16], input_data))])
            df_input = pd.get_dummies(df_input)
            df_input = df_input.reindex(columns=feature_names, fill_value=0)
            df_imputed = imputer.transform(df_input)
            df_scaled = scaler.transform(df_imputed)

            prediction_score = model.predict(df_scaled)[0]
            prediction = "Yes" if prediction_score > 0.5 else "No"
        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
