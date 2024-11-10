from flask import Flask, request, render_template
import pickle
import numpy as np
import datetime

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

    # Fetching the day, day of the week, and month
    day = int(request.form['day'])
    dayofweek = int(request.form['dayofweek'])
    month = int(request.form['month'])
    
    # Prepare data for the selected model
    final_features = [np.array(int_features)]  # Preparing the features

    predictions = []
    
    # Handle Bitcoin predictions
    if coin_type == "bitcoin":
        if prediction_type == "today":
            prediction = btc_model.predict(final_features)  # Prediction for today
            predictions.append(f'Predicted Bitcoin Close Price (Today): ${round(prediction[0], 2)}')
        elif prediction_type == "next_day":
            # Prepare next day's features
            next_day_features = final_features.copy()
            next_day_dayofweek = (dayofweek + 1) % 7  # Increment day of week
            next_day_month = month if day < 31 else (month + 1) % 12  # Update month if day rolls over
            next_day_day = (day + 1) if day < 31 else 1  # Update day correctly
            
            # Update features for next day prediction
            next_day_features[0][3] = next_day_dayofweek  # Updated Day of Week
            next_day_features[0][4] = next_day_month      # Updated Month
            next_day_features[0][5] = next_day_day        # Updated Day

            prediction = btc_model.predict(next_day_features)  # Prediction for the next day
            predictions.append(f'Predicted Bitcoin Close Price (Next Day): ${round(prediction[0], 2)}')
        else:
            return render_template('index.html', prediction_text='Invalid prediction type.')

    # Handle Ethereum predictions
    elif coin_type == "ethereum":
        if prediction_type == "today":
            prediction = eth_model.predict(final_features)  # Prediction for today
            predictions.append(f'Predicted Ethereum Close Price (Today): ${round(prediction[0], 2)}')
        elif prediction_type == "next_day":
            # Prepare next day's features
            next_day_features = final_features.copy()
            next_day_dayofweek = (dayofweek + 1) % 7  # Increment day of week
            next_day_month = month if day < 31 else (month + 1) % 12  # Update month if day rolls over
            next_day_day = (day + 1) if day < 31 else 1  # Update day correctly
            
            # Update features for next day prediction
            next_day_features[0][3] = next_day_dayofweek  # Updated Day of Week
            next_day_features[0][4] = next_day_month      # Updated Month
            next_day_features[0][5] = next_day_day        # Updated Day

            prediction = eth_model.predict(next_day_features)  # Prediction for the next day
            predictions.append(f'Predicted Ethereum Close Price (Next Day): ${round(prediction[0], 2)}')
        else:
            return render_template('index.html', prediction_text='Invalid prediction type.')

    # Combine predictions into a single string with line breaks
    prediction_text = "<br>".join(predictions)

    return render_template('index.html', prediction_text=prediction_text)

if __name__ == "__main__":
    app.run(debug=True)
