from flask import Flask, request, render_template 
import pickle
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load models from pickle files
btc_model = pickle.load(open("btcmodel.pkl", "rb"))
eth_model = pickle.load(open("cryptopricemodel.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Fetching data from the form
    prediction_type = request.form['prediction_type']
    coin_type = request.form['coin_type']
    
    int_features = [float(x) for x in request.form.values() if x not in [prediction_type, coin_type, 'day']]
    day = int(request.form['day'])
    dayofweek = int(request.form['dayofweek'])
    month = int(request.form['month'])
    
    # Prepare data for the selected model
    final_features = [np.array(int_features)]

    predictions = []
    
    # Handle Bitcoin predictions
    if coin_type == "bitcoin":
        if prediction_type == "today":
            prediction = btc_model.predict(final_features)
            predictions.append(f'Predicted Bitcoin Close Price (Today): ${round(prediction[0], 2)}')
        elif prediction_type == "next_day":
            next_day_features = final_features.copy()
            next_day_dayofweek = (dayofweek + 1) % 7
            next_day_month = month if day < 31 else (month + 1) % 12
            next_day_day = (day + 1) if day < 31 else 1
            next_day_features[0][3] = next_day_dayofweek
            next_day_features[0][4] = next_day_month
            next_day_features[0][5] = next_day_day
            prediction = btc_model.predict(next_day_features)
            predictions.append(f'Predicted Bitcoin Close Price (Next Day): ${round(prediction[0], 2)}')

    # Handle Ethereum predictions
    elif coin_type == "ethereum":
        if prediction_type == "today":
            prediction = eth_model.predict(final_features)
            predictions.append(f'Predicted Ethereum Close Price (Today): ${round(prediction[0], 2)}')
        elif prediction_type == "next_day":
            next_day_features = final_features.copy()
            next_day_dayofweek = (dayofweek + 1) % 7
            next_day_month = month if day < 31 else (month + 1) % 12
            next_day_day = (day + 1) if day < 31 else 1
            next_day_features[0][3] = next_day_dayofweek
            next_day_features[0][4] = next_day_month
            next_day_features[0][5] = next_day_day
            prediction = eth_model.predict(next_day_features)
            predictions.append(f'Predicted Ethereum Close Price (Next Day): ${round(prediction[0], 2)}')

    prediction_text = "<br>".join(predictions)
    return render_template('index.html', prediction_text=prediction_text)

# Additional Routes for About Us, Services, and Contact Us
@app.route('/about')
def about():
    return render_template('about.html')  # Create about.html

@app.route('/services')
def services():
    return render_template('services.html')  # Create services.html

@app.route('/contact')
def contact():
    return render_template('contact.html')  # Create contact.html

if __name__ == "__main__":
    app.run(debug=True)
