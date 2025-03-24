from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
from geopy.geocoders import Nominatim
from src.pipeline.prediction_pipeline import CustomData, PredictPipeline
import logging
import os
from datetime import datetime
import time
from dotenv import load_dotenv  # Import dotenv

# Load environment variables from .env file
load_dotenv()

# Retrieve API keys from environment variables
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
TOMTOM_API_KEY = os.getenv("TOMTOM_API_KEY")

# Configure logging
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

logging.basicConfig(
    filename=os.path.join(log_dir, f'app_{datetime.now().strftime("%Y%m%d")}.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# [Previous helper functions (get_lat_long, get_weather, get_traffic) remain the same]

def get_lat_long(location):
    """
    Uses geopy's Nominatim to convert a location name/address to latitude and longitude.
    """
    try:
        logging.info(f"Attempting to geocode location: {location}")
        geolocator = Nominatim(user_agent="delivery_app")
        location_data = geolocator.geocode(location)
        if location_data:
            logging.info(f"Successfully geocoded {location}: ({location_data.latitude}, {location_data.longitude})")
            return location_data.latitude, location_data.longitude
        else:
            logging.error(f"Could not geocode location: {location}")
            return None, None
    except Exception as e:
        logging.error(f"Error in geocoding {location}: {str(e)}")
        return None, None


def get_weather(lat, lon):
    """
    Retrieves weather conditions using OpenWeatherMap and maps them to one of the six accepted values:
    Fog, Stormy, Sandstorms, Windy, Cloudy, Sunny.
    """
    try:
        api_key = "0b3a8bf0d84195af379def95aa9d0310"
        url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}&units=metric"
        logging.info(f"Making weather API request for coordinates: ({lat}, {lon})")
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            weather_main = data['weather'][0]['main'].lower()  # Convert to lowercase for easier comparison
            weather_description = data['weather'][0]['description'].lower()

            # Map OpenWeatherMap weather conditions to the six accepted values
            if weather_main in ['fog', 'mist', 'haze']:
                mapped_weather = "Fog"
            elif weather_main in ['thunderstorm']:
                mapped_weather = "Stormy"
            elif weather_main in ['dust', 'sand', 'ash']:
                mapped_weather = "Sandstorms"
            elif weather_main in ['squall', 'tornado']:
                mapped_weather = "Windy"
            elif weather_main in ['clouds']:
                mapped_weather = "Cloudy"
            elif weather_main in ['clear']:
                mapped_weather = "Sunny"
            else:
                # Default to "Sunny" for any unexpected weather condition
                mapped_weather = "Sunny"

            logging.info(f"Weather condition retrieved: {weather_main} -> Mapped to: {mapped_weather}")
            return mapped_weather
        else:
            logging.error(f"Weather API error: Status code {response.status_code}")
            return "Sunny"  # Default to "Sunny" if API call fails
    except Exception as e:
        logging.error(f"Error in weather API call: {str(e)}")
        return "Sunny"  # Default to "Sunny" in case of any error

def get_traffic(lat, lon):
    """
    Retrieves traffic data using TomTom Traffic API.
    """
    try:
        api_key = "l07ONLcBBYC10PGEFEHAFh0RQGLUJBSi"
        url = f"https://api.tomtom.com/traffic/services/4/flowSegmentData/relative0/10/json?point={lat},{lon}&key={TOMTOM_API_KEY}"
        logging.info(f"Making traffic API request for coordinates: ({lat}, {lon})")
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            flow_data = data.get('flowSegmentData', {})
            current_speed = flow_data.get('currentSpeed', None)
            free_flow_speed = flow_data.get('freeFlowSpeed', None)
            if current_speed and free_flow_speed:
                speed_ratio = current_speed / free_flow_speed
                traffic_status = "Jam" if speed_ratio < 0.5 else "High" if speed_ratio < 0.8 else "Low"
                logging.info(f"Traffic condition determined: {traffic_status}")
                return traffic_status
            else:
                logging.warning("Missing speed data in traffic API response")
                return "Unknown"
        else:
            logging.error(f"Traffic API error: Status code {response.status_code}")
            return "Unknown"
    except Exception as e:
        logging.error(f"Error in traffic API call: {str(e)}")
        return "Unknown"

# @app.route('/api/predict', methods=['POST'])
# def predict_datapoint():
#     try:
#         data = request.get_json()
#         logging.info(f"Received prediction request with data: {data}")
        
#         # Extract locations
#         restaurant_location = data.get('Restaurant_location')
#         delivery_location = data.get('Delivery_location')
        
#         # Get coordinates
#         restaurant_lat, restaurant_long = get_lat_long(restaurant_location)
#         delivery_lat, delivery_long = get_lat_long(delivery_location)
#         time.sleep(5)
#         if None in (restaurant_lat, restaurant_long, delivery_lat, delivery_long):
#             return jsonify({
#                 "error": "Could not resolve one of the locations. Please check your inputs."
#             }), 400
        
#         # Get conditions
#         weather = get_weather(restaurant_lat, restaurant_long)
#         traffic = get_traffic(delivery_lat, delivery_long)
        
#         # Prepare input data structure
#         input_data = {
#             "delivery_person": {
#                 "age": int(data['Delivery_person_Age']),
#                 "ratings": float(data['Delivery_person_Ratings']),
#                 "vehicle": data['Type_of_vehicle'],
#                 "vehicle_condition": int(data['Vehicle_condition'])
#             },
#             "locations": {
#                 "restaurant": restaurant_location,
#                 "delivery": delivery_location,
#                 "city": data['City']
#             },
#             "conditions": {
#                 "weather": weather,
#                 "traffic": traffic,
#                 "festival": data['Festival']
#             }
#         }
        
#         # Create CustomData instance and get predictions
#         try:
#             custom_data = CustomData(
#                 Delivery_person_Age=input_data["delivery_person"]["age"],
#                 Delivery_person_Ratings=input_data["delivery_person"]["ratings"],
#                 Restaurant_latitude=float(restaurant_lat),
#                 Restaurant_longitude=float(restaurant_long),
#                 Delivery_location_latitude=float(delivery_lat),
#                 Delivery_location_longitude=float(delivery_long),
#                 Weather_conditions=weather,
#                 multiple_deliveries=float(data['multiple_deliveries']),
#                 Festival=input_data["conditions"]["festival"],
#                 City=input_data["locations"]["city"],
#                 Road_traffic_density=traffic,
#                 Vehicle_condition=input_data["delivery_person"]["vehicle_condition"],
#                 Type_of_vehicle=input_data["delivery_person"]["vehicle"]
#             )
            
#             final_new_data = custom_data.get_data_as_dataframe()
#             predict_pipeline = PredictPipeline()
#             predictions = predict_pipeline.predict(final_new_data)
            
#             avg_prediction = round(sum(predictions.values()) / len(predictions), 2)
            
#             # Prepare response data
#             response_data = {
#                 "input_data": input_data,
#                 "weather_conditions": weather,
#                 "traffic_conditions": traffic,
#                 "model_predictions": predictions,
#                 "average_prediction": avg_prediction
#             }
            
#             logging.info(f"Prediction successful: {response_data}")
#             return jsonify(response_data)
            
#         except Exception as e:
#             logging.error(f"Error in prediction process: {str(e)}")
#             return jsonify({
#                 "error": "Error in processing your request. Please try again."
#             }), 500
            
#     except Exception as e:
#         logging.error(f"General error in predict_datapoint: {str(e)}")
#         return jsonify({
#             "error": "An unexpected error occurred. Please try again."
#         }), 500

@app.route('/api/predict', methods=['POST'])
def predict_datapoint():
    try:
        data = request.get_json()
        logging.info(f"Received prediction request with data: {data}")
        
        # Extract locations
        restaurant_location = data.get('Restaurant_location')
        delivery_location = data.get('Delivery_location')
        
        # Get coordinates
        restaurant_lat, restaurant_long = get_lat_long(restaurant_location)
        delivery_lat, delivery_long = get_lat_long(delivery_location)
        time.sleep(5)
        if None in (restaurant_lat, restaurant_long, delivery_lat, delivery_long):
            return jsonify({
                "error": "Could not resolve one of the locations. Please check your inputs."
            }), 400
        
        # Get conditions
        weather = get_weather(restaurant_lat, restaurant_long)
        traffic = get_traffic(delivery_lat, delivery_long)
        
        # Prepare input data structure
        input_data = {
            "delivery_person": {
                "age": int(data['Delivery_person_Age']),
                "ratings": float(data['Delivery_person_Ratings']),
                "vehicle": data['Type_of_vehicle'],
                "vehicle_condition": int(data['Vehicle_condition'])
            },
            "locations": {
                "restaurant": restaurant_location,
                "delivery": delivery_location,
                "city": data['City']
            },
            "conditions": {
                "weather": weather,
                "traffic": traffic,
                "festival": data['Festival']
            }
        }
        
        # Create CustomData instance and get predictions
        try:
            custom_data = CustomData(
                Delivery_person_Age=input_data["delivery_person"]["age"],
                Delivery_person_Ratings=input_data["delivery_person"]["ratings"],
                Restaurant_latitude=float(restaurant_lat),
                Restaurant_longitude=float(restaurant_long),
                Delivery_location_latitude=float(delivery_lat),
                Delivery_location_longitude=float(delivery_long),
                Weather_conditions=weather,
                multiple_deliveries=float(data['multiple_deliveries']),
                Festival=input_data["conditions"]["festival"],
                City=input_data["locations"]["city"],
                Road_traffic_density=traffic,
                Vehicle_condition=input_data["delivery_person"]["vehicle_condition"],
                Type_of_vehicle=input_data["delivery_person"]["vehicle"]
            )
            
            final_new_data = custom_data.get_data_as_dataframe()
            predict_pipeline = PredictPipeline()
            predictions = predict_pipeline.predict(final_new_data)

            # Selecting only specific models' predictions to send
            selected_models = ["LinearRegression", "GradientBoost",  "RandomForestRegressor", "CatBoost"]
            filtered_predictions = {model: predictions[model] for model in selected_models if model in predictions}
            
            avg_prediction = round(sum(filtered_predictions.values()) / len(filtered_predictions), 2) if filtered_predictions else None
            
            # Prepare response data
            response_data = {
                "input_data": input_data,
                "weather_conditions": weather,
                "traffic_conditions": traffic,
                "model_predictions": filtered_predictions,
                "average_prediction": avg_prediction
            }
            
            logging.info(f"Prediction successful: {response_data}")
            return jsonify(response_data)
            
        except Exception as e:
            logging.error(f"Error in prediction process: {str(e)}")
            return jsonify({
                "error": "Error in processing your request. Please try again."
            }), 500
            
    except Exception as e:
        logging.error(f"General error in predict_datapoint: {str(e)}")
        return jsonify({
            "error": "An unexpected error occurred. Please try again."
        }), 500

if __name__ == "__main__":
    logging.info("Starting Flask application")
    app.run(host='0.0.0.0', debug=True, port=8000)