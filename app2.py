from flask import Flask, request, render_template, jsonify
import requests
from geopy.geocoders import Nominatim
from src.pipeline.prediction_pipeline import CustomData, PredictPipeline
import logging
import os
from datetime import datetime

# Configure logging
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

logging.basicConfig(
    filename=os.path.join(log_dir, f'app_{datetime.now().strftime("%Y%m%d")}.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

application = Flask(__name__)
app = application

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
        url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"
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
        url = f"https://api.tomtom.com/traffic/services/4/flowSegmentData/relative0/10/json?point={lat},{lon}&key={api_key}"
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

@app.route('/', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        logging.info("Serving GET request - displaying form")
        return render_template('form2.html')
    else:
        try:
            logging.info("Processing POST request for delivery prediction")
            
            # Extract user-provided data
            restaurant_location = request.form.get('Restaurant_location')
            delivery_location = request.form.get('Delivery_location')
            
            logging.info(f"Processing request for restaurant: {restaurant_location} to {delivery_location}")
            
            # Convert location names to latitude and longitude
            restaurant_lat, restaurant_long = get_lat_long(restaurant_location)
            delivery_lat, delivery_long = get_lat_long(delivery_location)
            
            if None in (restaurant_lat, restaurant_long, delivery_lat, delivery_long):
                logging.error("Location geocoding failed")
                return render_template('form2.html', final_result="Error: Could not resolve one of the locations. Please check your inputs.")
            
            # Retrieve weather and traffic conditions
            weather = get_weather(restaurant_lat, restaurant_long)
            traffic = get_traffic(delivery_lat, delivery_long)
            
            # Build the data instance for prediction
            try:
                data = CustomData(
                    Delivery_person_Age=int(request.form.get('Delivery_person_Age')),
                    Delivery_person_Ratings=float(request.form.get('Delivery_person_Ratings')),
                    Restaurant_latitude=float(restaurant_lat),
                    Restaurant_longitude=float(restaurant_long),
                    Delivery_location_latitude=float(delivery_lat),
                    Delivery_location_longitude=float(delivery_long),
                    Weather_conditions=weather,
                    multiple_deliveries=float(request.form.get('multiple_deliveries')),
                    Festival=request.form.get('Festival'),
                    City=request.form.get('City'),
                    Road_traffic_density=traffic,
                    Vehicle_condition=int(request.form.get('Vehicle_condition')),
                    Type_of_vehicle=request.form.get('Type_of_vehicle')
                )
                
                # Converting data to dataframe
                final_new_data = data.get_data_as_dataframe()

                logging.info(f"Prediction final_new_data: {final_new_data} ")
                print(f"Prediction final_new_data: {final_new_data} ")
                
                # Making predictions
                predict_pipeline = PredictPipeline()
                predictions = predict_pipeline.predict(final_new_data)
                
                avg_prediction = round(sum(predictions.values()) / len(predictions), 2)
            
            # Prepare detailed results
                results = {
                    'average_prediction': avg_prediction,
                    'model_predictions': predictions
                }
                print(f"results: {results}")



                logging.info(f"Prediction successful: {results}")
                return render_template(
                'prediction.html',
                final_result=f"Average predicted time: {avg_prediction} minutes",
                model_predictions=predictions
                )                
            except Exception as e:
                logging.error(f"Error in prediction process: {str(e)}")
                return render_template('form2.html', final_result="Error in processing your request. Please try again.")
                
        except Exception as e:
            logging.error(f"General error in predict_datapoint: {str(e)}")
            return render_template('form2.html', final_result="An unexpected error occurred. Please try again.")

if __name__ == "__main__":
    logging.info("Starting Flask application")
    app.run(host='0.0.0.0', debug=True, port=8001)