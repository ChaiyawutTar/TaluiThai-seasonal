import pandas as pd
import numpy as np
import re
from math import radians, sin, cos, sqrt, atan2
import joblib
import torch
import torch.nn as nn
from flask import Flask, request, jsonify, render_template
from waitress import serve
import logging
import os
from datetime import datetime
from tensorflow.keras.preprocessing.sequence import pad_sequences
from flask_cors import CORS
import csv


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("recommender_api.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- Global Variables & Configuration for API ---
# These will be populated by load_artifacts()
device_api = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"API using device: {device_api}")

loaded_model_api = None
df_all_api = None
tokenizer_api = None
location_scaler_api = None
mlb_season_api = None
category_encoder_api = None

# Constants needed for model instantiation and preprocessing
VOCAB_SIZE_API = None
LOCATION_INPUT_DIM_API = None
ITEM_SEASON_INPUT_DIM_API = None
QUERY_SEASON_INPUT_DIM_API = None
NUM_CATEGORIES_API = None
MAX_SEQUENCE_LENGTH_API = None
MODEL_PARAMS_API = None

# Paths for loading artifacts
# Update these paths to point to your actual files
INPUT_MLFLOW_ID = str(input("Please give mlflow run id: "))
DATA_FILE_PATH_API = os.getenv('DATA_FILE_PATH', 'allattractions_with_season.csv')
MODEL_ARTIFACT_PATH = os.getenv('MODEL_ARTIFACT_PATH',f'./models_save/contextual_recommender_v4_{INPUT_MLFLOW_ID}.pth')
PREPROCESSOR_ARTIFACT_PATH = os.getenv('PREPROCESSOR_ARTIFACT_PATH', 'recommender_preprocessors.joblib')

# --- Model Definition ---
class ContextualMultiModalRecommenderPyTorch(nn.Module):
    def __init__(self, vocab_size, text_embedding_dim,
                 location_input_dim, item_season_input_dim,
                 query_season_input_dim, num_categories, category_embedding_dim,
                 conv_filters, kernel_size,
                 dense_units, shared_dense_units, dropout_rate):
        super(ContextualMultiModalRecommenderPyTorch, self).__init__()
        self.embedding_text = nn.Embedding(vocab_size, text_embedding_dim, padding_idx=0)
        self.conv_text = nn.Conv1d(in_channels=text_embedding_dim, out_channels=conv_filters, kernel_size=kernel_size)
        self.relu_conv = nn.ReLU()
        self.dense_location1 = nn.Linear(location_input_dim, dense_units)
        self.relu_loc = nn.ReLU()
        self.dense_item_season1 = nn.Linear(item_season_input_dim, dense_units)
        self.relu_item_season = nn.ReLU()
        self.embedding_category = nn.Embedding(num_categories, category_embedding_dim)
        self.dense_category1 = nn.Linear(category_embedding_dim, dense_units)
        self.relu_cat = nn.ReLU()
        self.dense_query_season1 = nn.Linear(query_season_input_dim, dense_units)
        self.relu_query_season = nn.ReLU()
        combined_feature_size = conv_filters + (dense_units * 4)
        self.fc1 = nn.Linear(combined_feature_size, shared_dense_units)
        self.relu_fc1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(shared_dense_units, shared_dense_units // 2)
        self.relu_fc2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)
        self.output_layer = nn.Linear(shared_dense_units // 2, 1)

    def forward(self, text, location, item_season_features, category, query_season_features):
        x_text = self.embedding_text(text).permute(0, 2, 1)
        x_text = self.relu_conv(self.conv_text(x_text))
        x_text = torch.max(x_text, dim=2)[0]
        x_loc = self.relu_loc(self.dense_location1(location))
        x_item_season = self.relu_item_season(self.dense_item_season1(item_season_features))
        x_cat = self.embedding_category(category)
        x_cat = self.relu_cat(self.dense_category1(x_cat))
        x_query_season = self.relu_query_season(self.dense_query_season1(query_season_features))
        combined = torch.cat((x_text, x_loc, x_item_season, x_cat, x_query_season), dim=1)
        x = self.dropout1(self.relu_fc1(self.fc1(combined)))
        x = self.dropout2(self.relu_fc2(self.fc2(x)))
        return self.output_layer(x)

# --- Helper Functions ---
def parse_location(location_str):
    """Parse coordinates from string format 'lat,lon'."""
    try:
        parts = str(location_str).split(',')
        if len(parts) == 2:
            lat, lon = float(parts[0].strip()), float(parts[1].strip())
            return lat, lon if -90 <= lat <= 90 and -180 <= lon <= 180 else (0.0, 0.0)
    except Exception:
        logger.warning(f"Failed to parse location string: {location_str}")
        return 0.0, 0.0

def clean_season_string(season_str):
    """Parse season list from string format like "['summer', 'winter']"."""
    try:
        return eval(str(season_str))
    except Exception:
        logger.warning(f"Failed to parse season string: {season_str}")
        return ['unknown']

def haversine(lat1, lon1, lat2, lon2):
    """Calculate the great circle distance between two points on earth."""
    R = 6371  # Radius of Earth in km
    if not all(isinstance(coord, (int, float)) for coord in [lat1, lon1, lat2, lon2]):
        logger.warning(f"Invalid coordinates for haversine: {lat1}, {lon1}, {lat2}, {lon2}")
        return float('inf')
    
    try:
        lat1_rad, lon1_rad, lat2_rad, lon2_rad = map(radians, [lat1, lon1, lat2, lon2])
        dlon, dlat = lon2_rad - lon1_rad, lat2_rad - lat1_rad
        a = sin(dlat/2)**2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        return R * c
    except Exception as e:
        logger.error(f"Error in haversine calculation: {e}")
        return float('inf')

# --- Flask App Setup ---
app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

def load_artifacts_for_api():
    """Load all necessary artifacts for the recommender API."""
    global loaded_model_api, df_all_api, tokenizer_api, location_scaler_api, mlb_season_api, category_encoder_api
    global VOCAB_SIZE_API, LOCATION_INPUT_DIM_API, ITEM_SEASON_INPUT_DIM_API, QUERY_SEASON_INPUT_DIM_API
    global NUM_CATEGORIES_API, MAX_SEQUENCE_LENGTH_API, MODEL_PARAMS_API

    logger.info("Loading preprocessors...")
    try:
        preprocessor_data = joblib.load(PREPROCESSOR_ARTIFACT_PATH)
        tokenizer_api = preprocessor_data['tokenizer']
        location_scaler_api = preprocessor_data['location_scaler']
        mlb_season_api = preprocessor_data['mlb_season']
        category_encoder_api = preprocessor_data['category_encoder']
      
        VOCAB_SIZE_API = preprocessor_data['VOCAB_SIZE']
        LOCATION_INPUT_DIM_API = preprocessor_data['LOCATION_INPUT_DIM']
        ITEM_SEASON_INPUT_DIM_API = preprocessor_data['ITEM_SEASON_INPUT_DIM']
        QUERY_SEASON_INPUT_DIM_API = preprocessor_data['QUERY_SEASON_INPUT_DIM']
        NUM_CATEGORIES_API = preprocessor_data['NUM_CATEGORIES']
        MAX_SEQUENCE_LENGTH_API = preprocessor_data['MAX_SEQUENCE_LENGTH']
        MODEL_PARAMS_API = preprocessor_data['MODEL_PARAMS']
        logger.info(f"Preprocessors loaded successfully from {PREPROCESSOR_ARTIFACT_PATH}")
    except FileNotFoundError:
        logger.error(f"Preprocessor file {PREPROCESSOR_ARTIFACT_PATH} not found. API cannot function.")
        return False
    except Exception as e:
        logger.error(f"Error loading preprocessors: {e}. API cannot function.")
        return False

    logger.info("Loading attractions data...")
    try:
        df_all_api = pd.read_csv(DATA_FILE_PATH_API)
        
        # Apply minimal cleaning needed for inference
        df_all_api['ATT_DETAIL_TH'] = df_all_api['ATT_DETAIL_TH'].fillna('')
        df_all_api['ATT_LOCATION'] = df_all_api['ATT_LOCATION'].fillna('0.0,0.0')
        df_all_api['ATTR_CATAGORY_TH'] = df_all_api['ATTR_CATAGORY_TH'].fillna('Unknown')
        df_all_api['SUITABLE_SEASON'] = df_all_api['SUITABLE_SEASON'].fillna("['unknown']")
        df_all_api['ATT_NAME_TH'] = df_all_api['ATT_NAME_TH'].fillna('Unnamed Attraction')
        
        # Parse and add latitude/longitude columns for distance calculations
        parsed_locs_api = df_all_api['ATT_LOCATION'].apply(lambda x: pd.Series(parse_location(x)))
        df_all_api[['ATT_LATITUDE', 'ATT_LONGITUDE']] = parsed_locs_api
        
        logger.info(f"Attraction data loaded. Shape: {df_all_api.shape}")
    except FileNotFoundError:
        logger.error(f"Data file {DATA_FILE_PATH_API} not found.")
        return False
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return False
      
    logger.info("Loading trained model...")
    try:
        global loaded_model_api
        loaded_model_api = ContextualMultiModalRecommenderPyTorch(
            vocab_size=VOCAB_SIZE_API, 
            text_embedding_dim=MODEL_PARAMS_API["text_embedding_dim"],
            location_input_dim=LOCATION_INPUT_DIM_API, 
            item_season_input_dim=ITEM_SEASON_INPUT_DIM_API,
            query_season_input_dim=QUERY_SEASON_INPUT_DIM_API, 
            num_categories=NUM_CATEGORIES_API,
            category_embedding_dim=MODEL_PARAMS_API["category_embedding_dim"], 
            conv_filters=MODEL_PARAMS_API["conv_filters"],
            kernel_size=MODEL_PARAMS_API["kernel_size"], 
            dense_units=MODEL_PARAMS_API["dense_units_module"],
            shared_dense_units=MODEL_PARAMS_API["shared_dense_units"], 
            dropout_rate=MODEL_PARAMS_API["dropout_rate"]
        ).to(device_api)
        
        loaded_model_api.load_state_dict(torch.load(MODEL_ARTIFACT_PATH, map_location=device_api))
        loaded_model_api.eval()
        logger.info(f"Trained model loaded successfully from {MODEL_ARTIFACT_PATH}")
        return True
    except FileNotFoundError:
        logger.error(f"Model weights file {MODEL_ARTIFACT_PATH} not found.")
        return False
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return False

def generate_recommendations(
    query_season_name, 
    user_query_location=None, 
    max_distance_km=None,
    top_n=10, 
    weight_suitability=0.7, 
    weight_proximity=0.3
):
    """Generate attraction recommendations based on season and location."""
    if loaded_model_api is None:
        logger.error("Model not loaded. Cannot generate recommendations.")
        return None
    
    try:
        # Convert data to tensors compatible with the model
        # Text features
        temp_text_fillna = df_all_api['ATT_DETAIL_TH'].fillna('')
        inf_text_seq = tokenizer_api.texts_to_sequences(temp_text_fillna)
        inf_X_text_np = pad_sequences(inf_text_seq, maxlen=MAX_SEQUENCE_LENGTH_API)

        # Location features
        inf_X_loc_np = location_scaler_api.transform(df_all_api[['ATT_LATITUDE', 'ATT_LONGITUDE']].values)

        # Item's own seasons
        temp_season_fillna = df_all_api['SUITABLE_SEASON'].fillna("['unknown']")
        temp_season_list_rec = temp_season_fillna.apply(clean_season_string)
        inf_X_item_season_np = mlb_season_api.transform(temp_season_list_rec)

        # Category
        temp_cat_fillna = df_all_api['ATTR_CATAGORY_TH'].fillna('Unknown')
        inf_X_cat_np = category_encoder_api.transform(temp_cat_fillna)
      
        # Convert to PyTorch tensors
        inf_X_text_t = torch.LongTensor(inf_X_text_np).to(device_api)
        inf_X_loc_t = torch.FloatTensor(inf_X_loc_np).to(device_api)
        inf_X_item_s_t = torch.FloatTensor(inf_X_item_season_np).to(device_api)
        inf_X_cat_t = torch.LongTensor(inf_X_cat_np).to(device_api)

        # Process the query season
        try:
            query_s_transformed_inf = mlb_season_api.transform([[query_season_name]])
            query_s_tensor_inf = torch.FloatTensor(query_s_transformed_inf).to(device_api)
        except ValueError as e:
            logger.warning(f"Query season '{query_season_name}' not recognized. Using zero vector. Error: {e}")
            query_s_tensor_inf = torch.zeros(1, len(mlb_season_api.classes_)).to(device_api)

        # Get model predictions in batches
        all_model_scores = []
        with torch.no_grad():
            batch_size_inf_rec = 256
            for i in range(0, len(df_all_api), batch_size_inf_rec):
                b_text = inf_X_text_t[i:i+batch_size_inf_rec]
                b_loc = inf_X_loc_t[i:i+batch_size_inf_rec]
                b_item_s = inf_X_item_s_t[i:i+batch_size_inf_rec]
                b_cat = inf_X_cat_t[i:i+batch_size_inf_rec]
                b_query_s = query_s_tensor_inf.repeat(b_text.size(0), 1)
                outputs = loaded_model_api(b_text, b_loc, b_item_s, b_cat, b_query_s)
                scores_batch = torch.sigmoid(outputs).squeeze().cpu().numpy()
                all_model_scores.extend(scores_batch.tolist() if scores_batch.ndim > 0 else [scores_batch.item()])
      
        # Process results
        df_results = df_all_api.copy()
        df_results['suitability_score'] = all_model_scores

        # Handle location filtering and scoring
        if user_query_location and isinstance(user_query_location, tuple) and len(user_query_location) == 2:
            user_lat, user_lon = user_query_location
            # Calculate distance from user to each attraction
            df_results['distance_km'] = df_results.apply(
                lambda row: haversine(user_lat, user_lon, row['ATT_LATITUDE'], row['ATT_LONGITUDE']), axis=1
            )
            
            # Normalize distances for scoring
            min_dist = df_results['distance_km'].min()
            max_dist_val = df_results[df_results['distance_km'] != float('inf')]['distance_km'].max()
            
            if pd.isna(max_dist_val) or max_dist_val == min_dist:
                df_results['normalized_distance'] = 0.0
            else:
                # Cap distance at 95th percentile to handle outliers
                cap_dist = max(1.0, df_results[df_results['distance_km'] != float('inf')]['distance_km'].quantile(0.95))
                df_results['normalized_distance'] = df_results['distance_km'].apply(
                    lambda d: min(d / cap_dist, 1.0) if d != float('inf') else 1.0
                )
            
            # Composite score combines suitability and proximity
            df_results['composite_score'] = (
                weight_suitability * df_results['suitability_score'] + 
                weight_proximity * (1 - df_results['normalized_distance'])
            )
            
            # Apply maximum distance filter if specified
            if max_distance_km:
                df_results = df_results[df_results['distance_km'] <= max_distance_km]
            
            # Sort by composite score
            df_results = df_results.sort_values(by='composite_score', ascending=False)
        else:
            # No location data, sort by suitability score only
            df_results = df_results.sort_values(by='suitability_score', ascending=False)
        
        # Return top N results
        return df_results.head(top_n)
    
    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

# --- API Routes ---
@app.route('/')
def index():
    """Render the home page with a simple UI for recommendations."""
    # This assumes you have an index.html template in the templates folder
    seasons = []
    if mlb_season_api:
        seasons = [s for s in mlb_season_api.classes_.tolist() if s != 'unknown' and s.strip() != '']
    return render_template('index.html', seasons=seasons)

@app.route('/api/seasons')
def get_seasons():
    """API endpoint to get available seasons."""
    if mlb_season_api is None:
        return jsonify({"error": "Model not loaded"}), 503
    
    try:
        seasons = [s for s in mlb_season_api.classes_.tolist() if s != 'unknown' and s.strip() != '']
        return jsonify({"seasons": seasons})
    except Exception as e:
        logger.error(f"Error retrieving seasons: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/api/recommend', methods=['POST'])
def recommend_api():
    """API endpoint for generating recommendations."""
    # Check if model and preprocessors are loaded
    if loaded_model_api is None or df_all_api is None or tokenizer_api is None:
        return jsonify({"error": "Server not ready. Model or preprocessors not loaded."}), 503
    
    try:
        # Parse request data
        data = request.get_json()
        if not data:
            return jsonify({"error": "Missing JSON payload"}), 400
        
        # Extract required parameters
        query_season_name = data.get('query_season_name')
        if not query_season_name:
            return jsonify({"error": "Missing 'query_season_name'"}), 400
        
        # Parse optional parameters
        user_query_location = None
        user_location_str = data.get('user_query_location')
        if user_location_str:
            try:
                lat, lon = map(float, user_location_str.split(','))
                user_query_location = (lat, lon)
            except ValueError:
                return jsonify({"error": "Invalid 'user_query_location' format. Use 'lat,lon'"}), 400
        
        max_distance_km = data.get('max_distance_km')
        if max_distance_km is not None:
            try:
                max_distance_km = float(max_distance_km)
            except ValueError:
                return jsonify({"error": "Invalid 'max_distance_km'"}), 400

        # Parse remaining parameters with defaults
        top_n_req = int(data.get('top_n', 5))
        weight_suitability = float(data.get('weight_suitability', 0.7))
        weight_proximity = float(data.get('weight_proximity', 0.3))

        # Generate recommendations
        start_time = datetime.now()
        recommendations_df = generate_recommendations(
            query_season_name=query_season_name,
            user_query_location=user_query_location,
            max_distance_km=max_distance_km,
            top_n=top_n_req,
            weight_suitability=weight_suitability,
            weight_proximity=weight_proximity
        )
        
        if recommendations_df is None:
            return jsonify({"error": "Failed to generate recommendations"}), 500
        
        # Prepare response
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Select columns for output
        output_cols = ['ATT_NAME_TH', 'suitability_score', 'ATTR_CATAGORY_TH', 'ATT_LATITUDE', 'ATT_LONGITUDE']
        if 'composite_score' in recommendations_df.columns:
            output_cols.append('composite_score')
        if 'distance_km' in recommendations_df.columns:
            output_cols.append('distance_km')
        
        # Convert to list of dictionaries for JSON response
        result_list = recommendations_df[output_cols].to_dict(orient='records')
        
        return jsonify({
            "query_context": data,
            "processing_time_seconds": processing_time,
            "num_results": len(result_list),
            "recommendations": result_list
        })
    
    except Exception as e:
        logger.error(f"Error processing recommendation request: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({"error": "Internal server error", "details": str(e)}), 500

@app.route('/health')
def health_check():
    """Health check endpoint for monitoring."""
    if loaded_model_api is None or df_all_api is None:
        return jsonify({"status": "not_ready", "message": "Model or data not loaded"}), 503
    return jsonify({
        "status": "healthy",
        "model_loaded": loaded_model_api is not None,
        "data_loaded": df_all_api is not None if df_all_api is not None else 0,
        "data_rows": len(df_all_api) if df_all_api is not None else 0,
        "device": str(device_api)
    })

RATINGS_CSV_PATH = 'user_ratings.csv'

@app.route('/api/rate', methods=['POST'])
def rate_attraction():
    """API endpoint for saving user ratings."""
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    required_fields = ['ATT_NAME_TH', 'ATT_LATITUDE', 'ATT_LONGITUDE', 'rating']
    if not all(field in data for field in required_fields):
        return jsonify({"error": f"Missing one of required fields: {required_fields}"}), 400

    try:
        att_name = data['ATT_NAME_TH']
        lat = float(data['ATT_LATITUDE'])
        lon = float(data['ATT_LONGITUDE'])
        rating = int(data['rating'])
        timestamp = datetime.now().isoformat()

        # Append to CSV file
        file_exists = os.path.isfile(RATINGS_CSV_PATH)
        with open(RATINGS_CSV_PATH, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['ATT_NAME_TH', 'ATT_LATITUDE', 'ATT_LONGITUDE', 'rating', 'timestamp'])
            writer.writerow([att_name, lat, lon, rating, timestamp])

        logger.info(f"Saved rating {rating} for {att_name}")
        return jsonify({"success": True, "message": "Rating saved successfully"}), 200

    except Exception as e:
        logger.error(f"Error saving rating: {e}")
        return jsonify({"error": "Failed to save rating", "details": str(e)}), 500

# --- Application Entry Point ---
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5004))
    debug_mode = os.environ.get('DEBUG', 'False').lower() == 'true'
    
    if load_artifacts_for_api():
        logger.info(f"Starting API server on port {port} (debug={debug_mode})")
        
        if debug_mode:
            # Use Flask's development server for debugging
            app.run(host='0.0.0.0', port=port, debug=True)
        else:
            # Use production WSGI server (waitress) for deployment
            logger.info("Running in production mode with waitress")
            serve(app, host='0.0.0.0', port=port, threads=4)
    else:
        logger.critical("Failed to load required artifacts. Server not started.")