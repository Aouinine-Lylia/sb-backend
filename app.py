from flask import Flask, request, jsonify

import joblib
import numpy as np
from flask_cors import CORS
app = Flask(__name__)
CORS(app)
import pickle
import pandas as pd
import os


# Load model and columns
def load_pickle(filename):
    with open(filename, 'rb') as f:
        return joblib.load(f)

model_laptops = load_pickle('laptop_price_regression_model.pkl')
columns_laptops = load_pickle('model_columns.pkl')
print(columns_laptops)


with open('price_classifier_improved.pkl', 'rb') as f:
    model_package = pickle.load(f)

model_food = model_package['model']
feature_cols = model_package['feature_columns']
le_category = model_package['label_encoders']['category']
le_commodity = model_package['label_encoders']['commodity']
class_map = model_package['class_map']

def haversine_distance(lat, lon, lat2=36.7538, lon2=3.0588):
    """Calculate distance in km from Algiers"""
    R = 6371
    lat, lon, lat2, lon2 = map(np.radians, [lat, lon, lat2, lon2])
    dlat = lat2 - lat
    dlon = lon2 - lon
    a = np.sin(dlat/2)**2 + np.cos(lat) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c
df = pd.read_csv('sample_data/wfp_food_prices_dza.csv', skiprows=[1])
df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
df['price'] = pd.to_numeric(df['price'], errors='coerce')

# Calculate price_per_kg (simplified - you may need to adjust)
unit_conversion = {
    'KG': 1.0, '1 KG': 1.0, '500 G': 0.5, '250 G': 0.25, 
    'Unit': 0.2, 'L': 1.0, '1 L': 1.0
}
df['unit_kg_factor'] = df['unit'].map(unit_conversion)
df['price_per_kg'] = df['price'] / df['unit_kg_factor']

@app.route('/predict-laptop', methods=['POST'])
def predict_laptops():
    data = request.get_json()
    columns = [
        'RAM_GB', 'Total_Storage', 'CPU_Gen', 'Has_GPU', 'SCREEN_SIZE',
        'days_since_posted', 'CPU_Type_Core i3', 'CPU_Type_Core i5',
        'CPU_Type_Core i7', 'CPU_Type_Core i9', 'CPU_Type_Other',
        'CPU_Type_Ryzen', 'Condition_JAMAIS UTILIS', 'Condition_MOYEN',
        'Condition_Unknown', 'model_name_ALIENWARE', 'model_name_ASPIRE',
        'model_name_BLADE', 'model_name_COMPAQ', 'model_name_DYNABOOK',
        'model_name_ELITEBOOK', 'model_name_ENVY', 'model_name_GALAXY',
        'model_name_GF', 'model_name_IDEAPAD', 'model_name_IMAC',
        'model_name_INSPIRON', 'model_name_KATANA', 'model_name_LATITUDE',
        'model_name_LEGION', 'model_name_MAC', 'model_name_MACBOOK',
        'model_name_NITRO', 'model_name_OMEN', 'model_name_OPTIPLEX',
        'model_name_PAVILION', 'model_name_PRECISION', 'model_name_PREDATOR',
        'model_name_PROBOOK', 'model_name_ROG', 'model_name_SPECTRE',
        'model_name_SPIN', 'model_name_STEALTH', 'model_name_STRIX',
        'model_name_SURFACE', 'model_name_SWIFT', 'model_name_SWORD',
        'model_name_THINKBOOK', 'model_name_THINKPAD', 'model_name_TRANSFORMER',
        'model_name_TRAVELMATE', 'model_name_TUF', 'model_name_VECTOR',
        'model_name_VICTUS', 'model_name_VIVOBOOK', 'model_name_VOSTRO',
        'model_name_XPS', 'model_name_YOGA', 'model_name_ZBOOK',
        'model_name_ZENBOOK'
    ]
    # Ensure all columns are present
    input_data = [data.get(col, 0) for col in columns]
    input_array = np.array([input_data])
    prediction = model_laptops.predict(input_array)
    inverse_log_prediction = np.exp(prediction[0])
    return jsonify({'prediction': float(inverse_log_prediction)})


@app.route('/predict-food', methods=['POST'])
def predict_food():
    """
    Predict price classification
    
    Expected JSON payload:
    {
        "commodity": "Tomatoes",
        "category": "vegetables and fruits",
        "price_per_kg": 200,
        "date": "2024-03-15",
        "latitude": 36.75,
        "longitude": 3.04
    }
    """
    try:
        data = request.json
        print(data)
        
        # Extract inputs
        commodity = data.get('commodity')
        category = data.get('category')
        price_per_kg = float(data.get('price_per_kg'))
        date_str = data.get('date')
        latitude = float(data.get('latitude', 36.75))
        longitude = float(data.get('longitude', 3.04))
        
        # Convert date
        date = pd.to_datetime(date_str)
        
        # Calculate historical statistics
        commodity_hist = df[df['commodity'] == commodity]
        
        if len(commodity_hist) > 0:
            hist_mean = commodity_hist['price_per_kg'].mean()
            hist_std = commodity_hist['price_per_kg'].std()
            hist_min = commodity_hist['price_per_kg'].min()
            hist_max = commodity_hist['price_per_kg'].max()
            days_since_first = (date - commodity_hist['date'].min()).days
        else:
            # Use category averages for unknown commodities
            category_hist = df[df['category'] == category]
            hist_mean = category_hist['price_per_kg'].mean() if len(category_hist) > 0 else 500
            hist_std = category_hist['price_per_kg'].std() if len(category_hist) > 0 else 100
            hist_min = category_hist['price_per_kg'].min() if len(category_hist) > 0 else 0
            hist_max = category_hist['price_per_kg'].max() if len(category_hist) > 0 else 1000
            days_since_first = 0
        
        # Build features
        features = {}
        
        # Temporal
        features['year'] = date.year
        features['month'] = date.month
        features['quarter'] = (date.month - 1) // 3 + 1
        features['day_of_week'] = date.dayofweek
        features['is_weekend'] = 1 if date.dayofweek >= 5 else 0
        
        # Season
        if date.month in [12, 1, 2]:
            features['season'] = 0
        elif date.month in [3, 4, 5]:
            features['season'] = 1
        elif date.month in [6, 7, 8]:
            features['season'] = 2
        else:
            features['season'] = 3
        
        features['is_harvest_season'] = 1 if 3 <= date.month <= 11 else 0
        
        # Islamic calendar (simplified - set to 0 for now)
        features['is_ramadan'] = 0
        features['is_eid_fitr'] = 0
        features['is_eid_adha'] = 0
        features['is_islamic_holiday'] = 0
        
        # Geographic
        features['distance_from_capital'] = haversine_distance(latitude, longitude)
        features['is_coastal'] = 1 if latitude > 35.5 else 0
        
        # Product characteristics
        perishable_kw = ['Potato', 'Tomato', 'Onion', 'Egg', 'Milk', 'Meat', 'Fish', 'Chicken']
        imported_kw = ['Wheat', 'Rice', 'Sugar', 'Oil', 'Tea', 'Coffee', 'Pasta']
        staple_kw = ['Bread', 'Rice', 'Potato', 'Wheat flour', 'Pasta', 'Oil', 'Sugar', 'Milk']
        
        features['is_perishable'] = 1 if any(kw in commodity for kw in perishable_kw) else 0
        features['is_imported'] = 1 if any(kw in commodity for kw in imported_kw) else 0
        features['is_staple'] = 1 if any(kw in commodity for kw in staple_kw) else 0
        
        # Encode categorical
        try:
            features['category_encoded'] = le_category.transform([category])[0]
        except:
            features['category_encoded'] = 0
        
        try:
            features['commodity_encoded'] = le_commodity.transform([commodity])[0]
        except:
            features['commodity_encoded'] = 0
        
        # Historical features
        features['hist_mean'] = hist_mean
        features['hist_std'] = hist_std if hist_std > 0 else 1
        features['hist_min'] = hist_min
        features['hist_max'] = hist_max
        features['hist_volatility'] = (hist_std / hist_mean) if hist_mean > 0 else 0
        features['days_since_first_obs'] = max(0, days_since_first)
        features['price_trend'] = 0
        
        if hist_max > hist_min:
            features['price_position_in_range'] = (price_per_kg - hist_min) / (hist_max - hist_min)
        else:
            features['price_position_in_range'] = 0.5
        
        # Create DataFrame
        X_new = pd.DataFrame([features])
        for col in feature_cols:
            if col not in X_new.columns:
                X_new[col] = 0
        X_new = X_new[feature_cols].fillna(0)
        
        # Predict
        prediction = model_food.predict(X_new)[0]
        probabilities = model_food.predict_proba(X_new)[0]
        
        # Calculate z-score
        z_score = (price_per_kg - hist_mean) / hist_std if hist_std > 0 else 0
        
        # Build response
        result = {
            'prediction': class_map[prediction],
            'prediction_code': int(prediction),
            'confidence': float(probabilities[prediction]),
            'probabilities': {
                'CHEAP': float(probabilities[0]),
                'NORMAL': float(probabilities[1]),
                'EXPENSIVE': float(probabilities[2])
            },
            'z_score': float(z_score),
            'historical_mean': float(hist_mean),
            'historical_std': float(hist_std),
            'price_deviation': float(price_per_kg - hist_mean),
            'price_deviation_percentage': float((price_per_kg - hist_mean) / hist_mean * 100) if hist_mean > 0 else 0
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/')
def home():
    return 'Laptop Price Prediction API is running.'


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
