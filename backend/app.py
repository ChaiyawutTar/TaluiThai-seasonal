import pandas as pd
import numpy as np
import re
from math import radians, sin, cos, sqrt, atan2
import joblib # For loading preprocessors

# Remove scikit-learn train_test_split from Flask app if not needed
# from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler, LabelEncoder # Keep these
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import torch
import torch.nn as nn
from flask import Flask, request, jsonify

# --- Global Variables & Configuration for API ---
# These will be populated by load_artifacts()
device_api = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--- API using device: {device_api} ---")

loaded_model_api = None
df_all_api = None
tokenizer_api = None
location_scaler_api = None
mlb_season_api = None
category_encoder_api = None

# Constants needed for model instantiation and preprocessing (will be loaded from preprocessor file)
VOCAB_SIZE_API = None
LOCATION_INPUT_DIM_API = None
ITEM_SEASON_INPUT_DIM_API = None
QUERY_SEASON_INPUT_DIM_API = None
NUM_CATEGORIES_API = None
MAX_SEQUENCE_LENGTH_API = None # Load this from preprocessors
MODEL_PARAMS_API = None # Load this from preprocessors

# Paths for loading artifacts
# IMPORTANT: Update this path to point to the *specific best model* you want to deploy
# For example, one saved by an MLflow run: "contextual_recommender_v4_RUN_ID.pth"
MODEL_ARTIFACT_PATH = "contextual_recommender_v4_4450d188a83148468be882cf430ffd99.pth" # Example, replace with your actual best model path
PREPROCESSOR_ARTIFACT_PATH = "recommender_preprocessors.joblib" # Saved by training script
DATA_FILE_PATH_API = 'allattractions_with_season.csv' # For df_all

# --- Model Definition (must be identical to the one used for training) ---
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
    try:
        parts = str(location_str).split(',')
        if len(parts) == 2:
            lat, lon = float(parts[0].strip()), float(parts[1].strip())
            return lat, lon if -90 <= lat <= 90 and -180 <= lon <= 180 else (0.0, 0.0)
    except: return 0.0, 0.0

def clean_season_string(season_str):
    try: return eval(str(season_str))
    except: return ['unknown']

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    if not all(isinstance(coord, (int, float)) for coord in [lat1, lon1, lat2, lon2]):
        return float('inf')
    lat1_rad, lon1_rad, lat2_rad, lon2_rad = map(radians, [lat1, lon1, lat2, lon2])
    dlon, dlat = lon2_rad - lon1_rad, lat2_rad - lat1_rad
    a = sin(dlat/2)**2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c

app = Flask(__name__)

def load_artifacts_for_api():
    global loaded_model_api, df_all_api, tokenizer_api, location_scaler_api, mlb_season_api, category_encoder_api
    global VOCAB_SIZE_API, LOCATION_INPUT_DIM_API, ITEM_SEASON_INPUT_DIM_API, QUERY_SEASON_INPUT_DIM_API, NUM_CATEGORIES_API
    global MAX_SEQUENCE_LENGTH_API, MODEL_PARAMS_API

    print("--- API: Loading Preprocessors ---")
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
        MAX_SEQUENCE_LENGTH_API = preprocessor_data['MAX_SEQUENCE_LENGTH'] # Load this
        MODEL_PARAMS_API = preprocessor_data['MODEL_PARAMS'] # Load model params
        print(f"Preprocessors loaded successfully from {PREPROCESSOR_ARTIFACT_PATH}")
    except FileNotFoundError:
        print(f"ERROR: Preprocessor file {PREPROCESSOR_ARTIFACT_PATH} not found. API cannot function.")
        return False
    except Exception as e:
        print(f"ERROR loading preprocessors: {e}. API cannot function.")
        return False

    print("--- API: Loading All Attractions Data ---")
    try:
        df_all_api = pd.read_csv(DATA_FILE_PATH_API)
        # Apply minimal cleaning needed for inference data, consistent with training
        df_all_api['ATT_LOCATION'] = df_all_api['ATT_LOCATION'].fillna('0.0,0.0')
        parsed_locs_api = df_all_api['ATT_LOCATION'].apply(lambda x: pd.Series(parse_location(x)))
        df_all_api[['ATT_LATITUDE', 'ATT_LONGITUDE']] = parsed_locs_api
        print(f"Attraction data loaded. Shape: {df_all_api.shape}")
    except FileNotFoundError:
        print(f"Error: Main data file {DATA_FILE_PATH_API} not found for API.")
        return False
        
    print("--- API: Loading Trained Model ---")
    loaded_model_api = ContextualMultiModalRecommenderPyTorch(
        vocab_size=VOCAB_SIZE_API, text_embedding_dim=MODEL_PARAMS_API["text_embedding_dim"],
        location_input_dim=LOCATION_INPUT_DIM_API, item_season_input_dim=ITEM_SEASON_INPUT_DIM_API,
        query_season_input_dim=QUERY_SEASON_INPUT_DIM_API, num_categories=NUM_CATEGORIES_API,
        category_embedding_dim=MODEL_PARAMS_API["category_embedding_dim"], conv_filters=MODEL_PARAMS_API["conv_filters"],
        kernel_size=MODEL_PARAMS_API["kernel_size"], dense_units=MODEL_PARAMS_API["dense_units_module"],
        shared_dense_units=MODEL_PARAMS_API["shared_dense_units"], dropout_rate=MODEL_PARAMS_API["dropout_rate"]
    ).to(device_api)
    
    try:
        loaded_model_api.load_state_dict(torch.load(MODEL_ARTIFACT_PATH, map_location=device_api))
        loaded_model_api.eval()
        print(f"Trained model loaded successfully from {MODEL_ARTIFACT_PATH}")
        return True
    except FileNotFoundError:
        print(f"ERROR: Model weights file {MODEL_ARTIFACT_PATH} not found. API cannot serve recommendations.")
        return False
    except Exception as e:
        print(f"ERROR loading model weights: {e}. API cannot serve recommendations.")
        return False

def generate_recommendations_for_api( # Renamed to avoid conflict if importing from training script
    trained_model, df_full_data_inf, mlb_season_inf, tokenizer_inf, location_scaler_inf, category_encoder_inf,
    query_season_name: str, user_query_location: tuple = None, max_distance_km: float = None,
    top_n=10, weight_suitability=0.7, weight_proximity=0.3
):
    trained_model.eval()
    all_model_scores = []
    
    # Preprocess df_full_data_inf using loaded preprocessors
    temp_text_fillna = df_full_data_inf['ATT_DETAIL_TH'].fillna('')
    inf_text_seq = tokenizer_inf.texts_to_sequences(temp_text_fillna)
    inf_X_text_np = pad_sequences(inf_text_seq, maxlen=MAX_SEQUENCE_LENGTH_API) # Use loaded MAX_SEQUENCE_LENGTH_API

    # Location: transform using the loaded scaler
    # ATT_LATITUDE and ATT_LONGITUDE should already be on df_full_data_inf from load_artifacts
    inf_X_loc_np = location_scaler_inf.transform(df_full_data_inf[['ATT_LATITUDE', 'ATT_LONGITUDE']].values)

    # Item's own seasons: transform using loaded mlb
    temp_season_fillna = df_full_data_inf['SUITABLE_SEASON'].fillna("['unknown']") # Use original string column
    temp_season_list_rec = temp_season_fillna.apply(clean_season_string)
    inf_X_item_season_np = mlb_season_inf.transform(temp_season_list_rec)

    # Category: transform using loaded encoder
    temp_cat_fillna = df_full_data_inf['ATTR_CATAGORY_TH'].fillna('Unknown')
    inf_X_cat_np = category_encoder_inf.transform(temp_cat_fillna)
    
    inf_X_text_t = torch.LongTensor(inf_X_text_np).to(device_api)
    inf_X_loc_t = torch.FloatTensor(inf_X_loc_np).to(device_api)
    inf_X_item_s_t = torch.FloatTensor(inf_X_item_season_np).to(device_api)
    inf_X_cat_t = torch.LongTensor(inf_X_cat_np).to(device_api)

    try:
        query_s_transformed_inf = mlb_season_inf.transform([[query_season_name]])
        query_s_tensor_inf = torch.FloatTensor(query_s_transformed_inf).to(device_api)
    except ValueError:
        print(f"Warning (API Inference): Query season '{query_season_name}' not recognized. Using zero vector.")
        query_s_tensor_inf = torch.zeros(1, len(mlb_season_inf.classes_)).to(device_api)

    with torch.no_grad():
        batch_size_inf_rec = 256
        for i in range(0, len(df_full_data_inf), batch_size_inf_rec):
            b_text = inf_X_text_t[i:i+batch_size_inf_rec]
            b_loc = inf_X_loc_t[i:i+batch_size_inf_rec]
            b_item_s = inf_X_item_s_t[i:i+batch_size_inf_rec]
            b_cat = inf_X_cat_t[i:i+batch_size_inf_rec]
            b_query_s = query_s_tensor_inf.repeat(b_text.size(0), 1)
            outputs = trained_model(b_text, b_loc, b_item_s, b_cat, b_query_s)
            scores_batch = torch.sigmoid(outputs).squeeze().cpu().numpy()
            all_model_scores.extend(scores_batch.tolist() if scores_batch.ndim > 0 else [scores_batch.item()])
    
    df_results = df_full_data_inf.copy()
    df_results['suitability_score'] = all_model_scores

    if user_query_location and isinstance(user_query_location, tuple) and len(user_query_location) == 2:
        user_lat, user_lon = user_query_location
        df_results['distance_km'] = df_results.apply(
            lambda row: haversine(user_lat, user_lon, row['ATT_LATITUDE'], row['ATT_LONGITUDE']), axis=1
        )
        min_dist = df_results['distance_km'].min() # Recalculate for current results
        max_dist_val = df_results[df_results['distance_km'] != float('inf')]['distance_km'].max() # Recalculate
        if pd.isna(max_dist_val) or max_dist_val == min_dist : df_results['normalized_distance'] = 0.0
        else:
            cap_dist = max(1.0, df_results[df_results['distance_km'] != float('inf')]['distance_km'].quantile(0.95))
            df_results['normalized_distance'] = df_results['distance_km'].apply(lambda d: min(d / cap_dist, 1.0) if d != float('inf') else 1.0)
        df_results['composite_score'] = (weight_suitability * df_results['suitability_score']) + \
                                        (weight_proximity * (1 - df_results['normalized_distance']))
        if max_distance_km: df_results = df_results[df_results['distance_km'] <= max_distance_km]
        df_results = df_results.sort_values(by='composite_score', ascending=False)
    else:
        df_results = df_results.sort_values(by='suitability_score', ascending=False)
    return df_results.head(top_n)

@app.route('/recommend', methods=['POST'])
def recommend_api():
    if loaded_model_api is None or df_all_api is None or tokenizer_api is None:
        return jsonify({"error": "Model or preprocessors not loaded. Server not ready."}), 503
    try:
        data = request.get_json()
        if not data: return jsonify({"error": "Missing JSON payload"}), 400
        query_season_name = data.get('query_season_name')
        if not query_season_name: return jsonify({"error": "Missing 'query_season_name'"}), 400
        
        user_location_str = data.get('user_query_location')
        user_query_location = None
        if user_location_str:
            try:
                lat, lon = map(float, user_location_str.split(','))
                user_query_location = (lat, lon)
            except ValueError: return jsonify({"error": "Invalid 'user_query_location'"}), 400
        
        max_distance_km = data.get('max_distance_km')
        if max_distance_km is not None:
            try: max_distance_km = float(max_distance_km)
            except ValueError: return jsonify({"error": "Invalid 'max_distance_km'"}), 400

        top_n_req = int(data.get('top_n', 5))
        weight_suitability = float(data.get('weight_suitability', 0.7))
        weight_proximity = float(data.get('weight_proximity', 0.3))

        recommendations_df = generate_recommendations_for_api(
            trained_model=loaded_model_api, df_full_data_inf=df_all_api,
            mlb_season_inf=mlb_season_api, tokenizer_inf=tokenizer_api,
            location_scaler_inf=location_scaler_api, category_encoder_inf=category_encoder_api,
            query_season_name=query_season_name, user_query_location=user_query_location,
            max_distance_km=max_distance_km, top_n=top_n_req,
            weight_suitability=weight_suitability, weight_proximity=weight_proximity
        )
        output_cols = ['ATT_NAME_TH', 'suitability_score']
        if 'composite_score' in recommendations_df.columns: output_cols.append('composite_score')
        if 'distance_km' in recommendations_df.columns: output_cols.append('distance_km')
        result_list = recommendations_df[output_cols].to_dict(orient='records')
        return jsonify({"query_context": data, "recommendations": result_list})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": "Internal server error", "details": str(e)}), 500

if __name__ == '__main__': # This will be the entry point for app.py
    if load_artifacts_for_api():
        print("--- Starting Flask API Server for Recommendations ---")
        app.run(host='0.0.0.0', port=5001, debug=False) # Use debug=False for pseudo-production
    else:
        print("--- Could not load artifacts. Flask API Server not started. ---")