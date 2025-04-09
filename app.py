from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the model, scaler, and feature names
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
    
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
    
# Crop names mapping
crop_names = {
    1: 'Rice',
    2: 'Maize',
    3: 'Jute',
    4: 'Cotton',
    5: 'Coconut',
    6: 'Papaya',
    7: 'Orange',
    8: 'Apple',
    9: 'Muskmelon',
    10: 'Watermelon',
    11: 'Grapes',
    12: 'Mango',
    13: 'Banana',
    14: 'Pomegranate',
    15: 'Lentil',
    16: 'Blackgram',
    17: 'Mungbean',
    18: 'Mothbeans',
    19: 'Pigeonpeas',
    20: 'Kidneybeans',
    21: 'Chickpea',
    22: 'Coffee'
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        data = {
            'nitrogen': float(request.form['Nitrogen']),
            'phosphorus': float(request.form['Phosphorus']),
            'potassium': float(request.form['Potassium']),
            'temperature': float(request.form['Temperature']),
            'humidity': float(request.form['Humidity']),
            'ph': float(request.form['pH']),
            'rainfall': float(request.form['Rainfall'])
        }
        
        # Validate input ranges
        if not (0 <= data['ph'] <= 14):
            return render_template('index.html', result="Error: pH must be between 0 and 14")
            
        # Convert to numpy array
        input_data = np.array([
            data['nitrogen'],
            data['phosphorus'],
            data['potassium'],
            data['temperature'],
            data['humidity'],
            data['ph'],
            data['rainfall']
        ]).reshape(1, -1)
        
        # Scale the data
        scaled_data = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(scaled_data)
        crop = crop_names.get(prediction[0], "Unknown Crop")
        
        return render_template('index.html', result=crop)
        
    except KeyError as e:
        return render_template('index.html', result=f"Error: Missing field - {str(e)}")
    except ValueError as e:
        return render_template('index.html', result=f"Error: Invalid input - {str(e)}")
    except Exception as e:
        return render_template('index.html', result=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)