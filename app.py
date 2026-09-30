from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pandas as pd
import os
import requests
import google.generativeai as genai  # ✅ Gemini AI

app = Flask(__name__, template_folder=os.path.join('src', 'templates'))

# Load your trained ML model
model = joblib.load("src/models/best_model.joblib")

# ✅ API Keys
API_KEY_WEATHER = "a79484acfd84aba49769e1bbea411a16"  # OpenWeather
API_KEY_GEMINI = "AIzaSyCfnNi7ni0QWTG1AeR70I9FIxB277UcwXU"  

# Configure Gemini
genai.configure(api_key=API_KEY_GEMINI)


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/get_temperature', methods=['GET'])
def get_temperature():
    location = request.args.get('location')
    if not location:
        return jsonify({'error': 'Location not provided'}), 400

    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?q={location},IN&appid={API_KEY_WEATHER}&units=metric"
        response = requests.get(url)
        data = response.json()

        if response.status_code == 200 and 'main' in data:
            temp = data['main']['temp']
            return jsonify({'temperature': temp})
        else:
            return jsonify({'error': data.get('message', 'City not found')}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        Area = str(data['Area'])
        Item = str(data['Item'])
        Year = int(data['Year'])
        Rainfall = float(data['average_rain_fall_mm_per_year'])
        Pesticides = float(data['pesticides_tonnes'])
        Temp = float(data['avg_temp'])
        Location = str(data['Location'])

        input_df = pd.DataFrame([{
            'Area': Area,
            'Item': Item,
            'Year': Year,
            'average_rain_fall_mm_per_year': Rainfall,
            'pesticides_tonnes': Pesticides,
            'avg_temp': Temp
        }])

        prediction = model.predict(input_df)[0]

        # ✅ Gemini AI bilingual prompt
        prompt = f"""
        You are an agricultural expert. Based on these inputs:
        - Location: {Location}
        - Area: {Area}
        - Crop: {Item}
        - Year: {Year}
        - Average Rainfall: {Rainfall} mm/year
        - Pesticides Used: {Pesticides} tonnes
        - Average Temperature: {Temp} °C
        - Predicted Yield: {round(float(prediction), 2)} kg/ha

        Provide your suggestion in **both English and Hindi**.

        English:
        1. Is this crop suitable for this location?
        2. Suggest one or two better crop alternatives (if any).
        3. Give one short tip to improve yield.

        हिंदी:
        1. क्या यह फसल इस स्थान के लिए उपयुक्त है?
        2. यदि संभव हो तो एक या दो वैकल्पिक फसलों का सुझाव दें।
        3. उत्पादन बढ़ाने के लिए एक छोटा सुझाव दें।

        Keep the answer clear, short, and farmer-friendly.
        """

        model_ai = genai.GenerativeModel("gemini-2.5-flash")
        ai_response = model_ai.generate_content(prompt)
        suggestion = ai_response.text.strip()

        return jsonify({
            'predicted_yield': round(float(prediction), 2),
            'live_temperature': Temp,
            'location': Location,
            'ai_suggestion': suggestion
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
