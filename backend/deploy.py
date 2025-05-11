import pandas as pd
import numpy as np
import re
from math import radians, sin, cos, sqrt, atan2

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler, LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# --- MLOps and Responsible AI Libraries ---
import mlflow
import mlflow.pytorch
# import shap # Uncomment if you have SHAP installed and want to run the SHAP part
# from flask import Flask, request, jsonify # For deployment example

# --- Configuration & Constants ---
DATA_FILE_PATH = 'allattractions_with_season.csv'
MODEL_SAVE_PATH_BASE = "contextual_recommender_v4" # Base name for MLflow saving

# Preprocessing constants
MAX_WORDS = 10000
MAX_SEQUENCE_LENGTH = 200

# Training constants
BATCH_SIZE = 64
NUM_EPOCHS = 5 # Reduced for quicker demonstration with MLflow and SHAP
LEARNING_RATE = 0.001

# Model hyperparameters (these will be logged by MLflow)
MODEL_PARAMS = {
    "text_embedding_dim": 100,
    "category_embedding_dim": 20,
    "conv_filters": 128,
    "kernel_size": 5,
    "dense_units_module": 32,
    "shared_dense_units": 128,
    "dropout_rate": 0.5
}

# Recommendation constants
TOP_N_RECOMMENDATIONS = 5

# --- Device Configuration ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--- Using device: {device} ---")

# --- Model Definition (ContextualMultiModalRecommenderPyTorch) ---
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

# --- Main Script Execution ---
if __name__ == '__main__':
    print("--- Loading and Preprocessing Data ---")
    try:
        df_all = pd.read_csv(DATA_FILE_PATH)
    except FileNotFoundError:
        print(f"Error: {DATA_FILE_PATH} not found.")
        exit()

    df_all['ATT_DETAIL_TH'] = df_all['ATT_DETAIL_TH'].fillna('')
    df_all['ATT_LOCATION'] = df_all['ATT_LOCATION'].fillna('0.0,0.0')
    df_all['ATTR_CATAGORY_TH'] = df_all['ATTR_CATAGORY_TH'].fillna('Unknown')
    df_all['SUITABLE_SEASON'] = df_all['SUITABLE_SEASON'].fillna("['unknown']")
    df_all['ATT_NAME_TH'] = df_all['ATT_NAME_TH'].fillna('Unnamed Attraction')

    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<unk>")
    tokenizer.fit_on_texts(df_all['ATT_DETAIL_TH'])
    VOCAB_SIZE = len(tokenizer.word_index) + 1

    parsed_locs = df_all['ATT_LOCATION'].apply(lambda x: pd.Series(parse_location(x)))
    df_all[['ATT_LATITUDE', 'ATT_LONGITUDE']] = parsed_locs
    location_scaler = StandardScaler().fit(parsed_locs.values)
    LOCATION_INPUT_DIM = parsed_locs.shape[1]

    df_all['SUITABLE_SEASON_LIST'] = df_all['SUITABLE_SEASON'].apply(clean_season_string)
    mlb_season = MultiLabelBinarizer()
    mlb_season.fit(df_all['SUITABLE_SEASON_LIST'])
    ITEM_SEASON_INPUT_DIM = len(mlb_season.classes_)
    QUERY_SEASON_INPUT_DIM = ITEM_SEASON_INPUT_DIM
    POSSIBLE_QUERY_SEASONS = [s for s in mlb_season.classes_.tolist() if s != 'unknown' and s.strip() != '']

    category_encoder = LabelEncoder().fit(df_all['ATTR_CATAGORY_TH'])
    NUM_CATEGORIES = len(category_encoder.classes_)

    X_text_all_np = pad_sequences(tokenizer.texts_to_sequences(df_all['ATT_DETAIL_TH']), maxlen=MAX_SEQUENCE_LENGTH)
    X_location_all_np_scaled = location_scaler.transform(parsed_locs.values)
    X_item_season_all_np = mlb_season.transform(df_all['SUITABLE_SEASON_LIST'])
    X_category_encoded_all_np = category_encoder.transform(df_all['ATTR_CATAGORY_TH'])

    print("\n--- Generating Training Data with Nuanced Labels ---")
    train_samples_list = []
    for idx, item_row in df_all.iterrows():
        item_text_features = X_text_all_np[idx]
        item_loc_features = X_location_all_np_scaled[idx]
        item_season_features = X_item_season_all_np[idx]
        item_cat_features = X_category_encoded_all_np[idx]
        item_actual_suitable_seasons = set(s for s in item_row['SUITABLE_SEASON_LIST'] if s in POSSIBLE_QUERY_SEASONS)
        num_relevant_suitable_seasons_for_item = len(item_actual_suitable_seasons)
        for query_s_name in POSSIBLE_QUERY_SEASONS:
            query_s_transformed = mlb_season.transform([[query_s_name]])[0]
            label = 0.0
            if query_s_name in item_actual_suitable_seasons and num_relevant_suitable_seasons_for_item > 0:
                label = 1.0 / num_relevant_suitable_seasons_for_item
            train_samples_list.append({
                'text': item_text_features, 'location': item_loc_features,
                'item_season': item_season_features, 'category': item_cat_features,
                'query_season': query_s_transformed, 'label': label,
                'attraction_id': idx # Keep track for SHAP later
            })
    df_train_expanded = pd.DataFrame(train_samples_list)
    
    X_text_t_exp = torch.LongTensor(np.array(df_train_expanded['text'].tolist()))
    X_loc_t_exp = torch.FloatTensor(np.array(df_train_expanded['location'].tolist()))
    X_item_s_t_exp = torch.FloatTensor(np.array(df_train_expanded['item_season'].tolist()))
    X_cat_t_exp = torch.LongTensor(np.array(df_train_expanded['category'].tolist())) # This is correct for nn.Embedding
    X_query_s_t_exp = torch.FloatTensor(np.array(df_train_expanded['query_season'].tolist()))
    y_labels_t_exp = torch.FloatTensor(df_train_expanded['label'].values).unsqueeze(1)
    # Keep track of original attraction IDs for SHAP
    attraction_ids_exp = torch.LongTensor(df_train_expanded['attraction_id'].values)


    print("\n--- Splitting Data into Train/Validation ---")
    stratify_labels = y_labels_t_exp.numpy().flatten()
    unique_labels, counts = np.unique(stratify_labels, return_counts=True)
    can_stratify = len(unique_labels) > 1 and all(c >= 2 for c in counts)

    indices = np.arange(len(y_labels_t_exp))
    train_indices, val_indices = train_test_split(
        indices, test_size=0.2, random_state=42,
        stratify=(stratify_labels if can_stratify else None)
    )

    X_text_train, X_text_val = X_text_t_exp[train_indices], X_text_t_exp[val_indices]
    X_loc_train, X_loc_val = X_loc_t_exp[train_indices], X_loc_t_exp[val_indices]
    X_item_s_train, X_item_s_val = X_item_s_t_exp[train_indices], X_item_s_t_exp[val_indices]
    X_cat_train, X_cat_val = X_cat_t_exp[train_indices], X_cat_t_exp[val_indices]
    X_query_s_train, X_query_s_val = X_query_s_t_exp[train_indices], X_query_s_t_exp[val_indices]
    y_train, y_val = y_labels_t_exp[train_indices], y_labels_t_exp[val_indices]
    # Attraction IDs for SHAP background/test set
    attraction_ids_train, attraction_ids_val = attraction_ids_exp[train_indices], attraction_ids_exp[val_indices]


    train_dataset = TensorDataset(X_text_train, X_loc_train, X_item_s_train, X_cat_train, X_query_s_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataset = TensorDataset(X_text_val, X_loc_val, X_item_s_val, X_cat_val, X_query_s_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # --- MLflow Tracking (Task 1.4) ---
    mlflow.set_experiment("Attraction Recommender Experiments")
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"MLflow Run ID: {run_id}")
        MODEL_SAVE_PATH = f"{MODEL_SAVE_PATH_BASE}_{run_id}.pth"

        # Log hyperparameters
        mlflow.log_param("max_words", MAX_WORDS)
        mlflow.log_param("max_sequence_length", MAX_SEQUENCE_LENGTH)
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("num_epochs", NUM_EPOCHS)
        mlflow.log_param("learning_rate", LEARNING_RATE)
        for key, value in MODEL_PARAMS.items():
            mlflow.log_param(key, value)

        model = ContextualMultiModalRecommenderPyTorch(
            vocab_size=VOCAB_SIZE, text_embedding_dim=MODEL_PARAMS["text_embedding_dim"],
            location_input_dim=LOCATION_INPUT_DIM, item_season_input_dim=ITEM_SEASON_INPUT_DIM,
            query_season_input_dim=QUERY_SEASON_INPUT_DIM, num_categories=NUM_CATEGORIES,
            category_embedding_dim=MODEL_PARAMS["category_embedding_dim"], conv_filters=MODEL_PARAMS["conv_filters"],
            kernel_size=MODEL_PARAMS["kernel_size"], dense_units=MODEL_PARAMS["dense_units_module"],
            shared_dense_units=MODEL_PARAMS["shared_dense_units"], dropout_rate=MODEL_PARAMS["dropout_rate"]
        ).to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

        print(f"\n--- Starting Training with MLflow (Run ID: {run_id}) ---")
        best_val_loss = float('inf')
        for epoch in range(NUM_EPOCHS):
            model.train()
            total_train_loss = 0
            for b_text, b_loc, b_item_s, b_cat, b_query_s, b_labels in train_loader:
                b_text,b_loc,b_item_s,b_cat,b_query_s,b_labels = \
                    b_text.to(device),b_loc.to(device),b_item_s.to(device),b_cat.to(device),b_query_s.to(device),b_labels.to(device)
                optimizer.zero_grad()
                outputs = model(b_text, b_loc, b_item_s, b_cat, b_query_s)
                loss = criterion(outputs, b_labels)
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()
            avg_train_loss = total_train_loss / len(train_loader)
            mlflow.log_metric("train_loss", avg_train_loss, step=epoch)

            model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for b_text_v, b_loc_v, b_item_s_v, b_cat_v, b_query_s_v, b_labels_v in val_loader:
                    b_text_v,b_loc_v,b_item_s_v,b_cat_v,b_query_s_v,b_labels_v = \
                        b_text_v.to(device),b_loc_v.to(device),b_item_s_v.to(device),b_cat_v.to(device),b_query_s_v.to(device),b_labels_v.to(device)
                    outputs_val = model(b_text_v, b_loc_v, b_item_s_v, b_cat_v, b_query_s_v)
                    loss_val = criterion(outputs_val, b_labels_v)
                    total_val_loss += loss_val.item()
            avg_val_loss = total_val_loss / len(val_loader)
            mlflow.log_metric("val_loss", avg_val_loss, step=epoch)
            print(f"Epoch {epoch+1}/{NUM_EPOCHS} -> Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), MODEL_SAVE_PATH)
                print(f"  Model improved and saved to {MODEL_SAVE_PATH}")
        
        mlflow.pytorch.log_model(model, "recommender_model", registered_model_name="ContextualRecommender")
        mlflow.log_metric("best_val_loss", best_val_loss)
        print(f"\n--- Training Complete. Best model for Run ID {run_id} saved to {MODEL_SAVE_PATH} ---")

    # --- Task 1.3: Model Fairness Analysis (Conceptual Discussion) ---
    print("\n\n--- Task 1.3: Model Fairness Analysis (Conceptual) ---")
    print("Potential sources of bias in this dataset could include:")
    print("1. Popularity Bias: Popular attractions might have more detailed descriptions or appear more frequently, leading to better scores irrespective of true nuanced suitability.")
    print("2. Geographical Bias: If data is skewed towards certain regions, the model might perform better for those regions.")
    print("3. Categorical Bias: Certain ATTR_CATAGORY_TH might be over/under-represented, affecting recommendations.")
    print("To implement fairness metrics (e.g., using Fairlearn):")
    print(" - Define protected attributes (e.g., 'REGION_NAME_TH' if available, or binned 'PRICE_LEVEL' if available, or even ATTR_CATAGORY_TH).")
    print(" - Choose a fairness metric (e.g., Demographic Parity for recommendation exposure, Equalized Odds if predicting suitability as a binary task).")
    print(" - Calculate the metric across groups defined by the protected attribute.")
    print("Approaches to mitigate bias:")
    print(" - Pre-processing: Re-sample data, augment under-represented groups.")
    print(" - In-processing: Add fairness constraints or regularization terms to the model's loss function.")
    print(" - Post-processing: Re-rank recommendations to improve fairness metrics for different groups.")

    # --- Task 1.5: Model Explainability (Conceptual Discussion, SHAP is in Task 1.6) ---
    print("\n\n--- Task 1.5: Model Explainability (Conceptual) ---")
    print("Beyond SHAP (Task 1.6), other explainability methods for this model type could include:")
    print("1. Permutation Feature Importance: Randomly shuffle values of one input feature at a time (e.g., all location data, or specific category embeddings) and measure the drop in model performance (e.g., validation loss). Larger drops indicate higher importance.")
    print("2. Analyzing Embedding Layers: Visualize embeddings (e.g., text or category embeddings using t-SNE/UMAP) to see if similar items cluster together meaningfully.")
    print("3. Attention Mechanism: If an attention mechanism were added (e.g., in the text processing or feature fusion stage), visualizing attention weights would show which parts of input the model focuses on.")


    # --- Task 1.6: Prediction Reasoning (Conceptual SHAP Integration) ---
    # Note: Running SHAP can be computationally intensive, especially DeepExplainer on larger models/datasets.
    # This is a conceptual setup.
    print("\n\n--- Task 1.6: Prediction Reasoning (SHAP - Conceptual Setup) ---")
    SHAP_ENABLED = False # Set to True to run the SHAP part, ensure 'shap' is installed
    if SHAP_ENABLED:
        try:
            import shap
            print("SHAP library found. Proceeding with conceptual SHAP example.")
            # We need a background dataset for the explainer.
            # Using a subset of the training data is common. Let's take a small sample.
            # SHAP expects all model inputs to be passed to the explainer.
            # We need to wrap the model's forward pass if it expects multiple inputs.

            # Create a wrapper for the model if SHAP explainer needs a single input tensor
            # or can handle lists/tuples of tensors. DeepExplainer often expects tensors.
            # For DeepExplainer, inputs should be tensors.
            
            # Let's use a few samples from the validation set as the test instances for SHAP
            # And a few samples from the training set as background
            num_background_samples = 50 # Small number for speed
            num_test_samples_shap = 3   # Explain 3 test predictions

            # Ensure all inputs are tensors
            shap_background_text = X_text_train[:num_background_samples].to(device)
            shap_background_loc = X_loc_train[:num_background_samples].to(device)
            shap_background_item_s = X_item_s_train[:num_background_samples].to(device)
            shap_background_cat = X_cat_train[:num_background_samples].to(device) # Already LongTensor
            shap_background_query_s = X_query_s_train[:num_background_samples].to(device)
            
            shap_test_text = X_text_val[:num_test_samples_shap].to(device)
            shap_test_loc = X_loc_val[:num_test_samples_shap].to(device)
            shap_test_item_s = X_item_s_val[:num_test_samples_shap].to(device)
            shap_test_cat = X_cat_val[:num_test_samples_shap].to(device) # Already LongTensor
            shap_test_query_s = X_query_s_val[:num_test_samples_shap].to(device)
            
            # For DeepExplainer, it expects a list of tensors as background if model has multiple inputs
            background_data_list = [shap_background_text, shap_background_loc, shap_background_item_s, shap_background_cat, shap_background_query_s]
            test_data_list = [shap_test_text, shap_test_loc, shap_test_item_s, shap_test_cat, shap_test_query_s]

            # Load the best model for SHAP analysis
            best_model_for_shap = ContextualMultiModalRecommenderPyTorch(
                vocab_size=VOCAB_SIZE, text_embedding_dim=MODEL_PARAMS["text_embedding_dim"],
                location_input_dim=LOCATION_INPUT_DIM, item_season_input_dim=ITEM_SEASON_INPUT_DIM,
                query_season_input_dim=QUERY_SEASON_INPUT_DIM, num_categories=NUM_CATEGORIES,
                category_embedding_dim=MODEL_PARAMS["category_embedding_dim"], conv_filters=MODEL_PARAMS["conv_filters"],
                kernel_size=MODEL_PARAMS["kernel_size"], dense_units=MODEL_PARAMS["dense_units_module"],
                shared_dense_units=MODEL_PARAMS["shared_dense_units"], dropout_rate=MODEL_PARAMS["dropout_rate"]
            ).to(device)
            
            # Find the saved model path from the best MLflow run if possible, or use the last saved one
            final_model_path_for_shap = MODEL_SAVE_PATH # Path of the model saved from the training loop
            if mlflow.active_run() is not None: # If script is run through MLflow again for this section
                 # This assumes the MODEL_SAVE_PATH was correctly set during the training run
                 pass # Use MODEL_SAVE_PATH
            
            try:
                best_model_for_shap.load_state_dict(torch.load(final_model_path_for_shap, map_location=device))
                best_model_for_shap.eval()
                print(f"SHAP: Loaded model from {final_model_path_for_shap}")

                # DeepExplainer is suitable for PyTorch models with differentiable components
                # It expects a model and a background dataset (list of tensors for multi-input)
                explainer = shap.DeepExplainer(best_model_for_shap, background_data_list)
                
                print("SHAP: Calculating SHAP values for test samples (this might take a while)...")
                shap_values = explainer.shap_values(test_data_list)
                
                print("SHAP: SHAP values calculated.")
                # shap_values will be a list of arrays, one for each input type.
                # e.g., shap_values[0] for text input, shap_values[1] for location, etc.
                # The dimensions would be (num_test_samples, num_features_for_that_input_type)
                # For text, it might be (num_test_samples, max_seq_len, embedding_dim) if explaining embeddings,
                # or (num_test_samples, max_seq_len) if explaining input tokens (more complex setup needed for token-level).
                # DeepExplainer usually gives importance per input neuron of the first layer it can "see" (e.g. embedding output).

                print("\nExample SHAP value interpretation (Conceptual):")
                for i in range(num_test_samples_shap):
                    print(f"\n--- SHAP Explanation for Test Sample {i+1} ---")
                    # Get original attraction ID and query season for context
                    original_attraction_idx = attraction_ids_val[i].item()
                    attraction_name = df_all.iloc[original_attraction_idx]['ATT_NAME_TH']
                    
                    # The query season for this test sample
                    # We need to reconstruct which query season this sample corresponded to.
                    # This requires pairing back to df_train_expanded or passing query season names.
                    # For this conceptual example, let's just note it.
                    # query_season_for_this_sample_idx = ... (need to map back)
                    # query_season_name_for_this_sample = ...
                    
                    actual_label = y_val[i].item()
                    pred_logit = best_model_for_shap(*[td[i:i+1] for td in test_data_list]).item()
                    pred_prob = torch.sigmoid(torch.tensor(pred_logit)).item()

                    print(f"Attraction (Original Index): {original_attraction_idx} - {attraction_name}")
                    print(f"Predicted Suitability Score: {pred_prob:.4f}, Actual Label: {actual_label:.4f}")

                    # SHAP values for text features (sum over embedding dim for importance per token)
                    # This interpretation is highly dependent on SHAP's output structure for embeddings.
                    # Usually for text, you'd sum shap_values across the embedding dimension.
                    # text_shap_sum = np.sum(shap_values[0][i], axis=-1) # Sum over embedding dim
                    # print(f"Text feature SHAP values (summed over embedding dim, first 10): {text_shap_sum[:10]}")
                    
                    # For other features, SHAP values are more direct
                    print(f"Location feature SHAP values: {shap_values[1][i]}")
                    print(f"Item Season feature SHAP values: {shap_values[2][i]} (Classes: {mlb_season.classes_})")
                    # Category SHAP values would be for the embedding output of the category
                    # print(f"Category feature SHAP values (for category ID {X_cat_val[i].item()} -> {category_encoder.inverse_transform([X_cat_val[i].item()])[0]}): {shap_values[3][i]}")
                    print(f"Query Season feature SHAP values: {shap_values[4][i]} (Classes: {mlb_season.classes_})")
                    print("Positive SHAP values increase prediction, negative values decrease.")
                    print("Interpretation: The magnitude shows feature importance for *this specific prediction*.")
                    # To get feature names for item/query seasons, use mlb_season.classes_
                    # To get category name, use category_encoder.inverse_transform()

            except ImportError:
                print("SHAP library not installed. Skipping SHAP analysis. Pip install shap.")
            except Exception as e:
                print(f"Error during SHAP analysis: {e}")
                import traceback
                traceback.print_exc()
        except FileNotFoundError:
            print("SHAP analysis disabled. Set SHAP_ENABLED = True to run.")



    # --- Recommendation Generation (Inference after MLflow training) ---
    # This part can be run after the MLflow training to test the best model from that run
    print("\n\n--- Example: Generating Recommendations with Model from MLflow Run ---")
    if 'run_id' in locals() and run_id is not None: # Check if MLflow run happened
        final_model_path_from_mlflow_run = f"{MODEL_SAVE_PATH_BASE}_{run_id}.pth" # Path based on run_id
        
        loaded_model_for_inference = ContextualMultiModalRecommenderPyTorch(
            vocab_size=VOCAB_SIZE, text_embedding_dim=MODEL_PARAMS["text_embedding_dim"],
            location_input_dim=LOCATION_INPUT_DIM, item_season_input_dim=ITEM_SEASON_INPUT_DIM,
            query_season_input_dim=QUERY_SEASON_INPUT_DIM, num_categories=NUM_CATEGORIES,
            category_embedding_dim=MODEL_PARAMS["category_embedding_dim"], conv_filters=MODEL_PARAMS["conv_filters"],
            kernel_size=MODEL_PARAMS["kernel_size"], dense_units=MODEL_PARAMS["dense_units_module"],
            shared_dense_units=MODEL_PARAMS["shared_dense_units"], dropout_rate=MODEL_PARAMS["dropout_rate"]
        ).to(device)

        try:
            loaded_model_for_inference.load_state_dict(torch.load(final_model_path_from_mlflow_run, map_location=device))
            print(f"Successfully loaded trained model from MLflow run ({final_model_path_from_mlflow_run}) for inference.")
            
            # (generate_trained_recommendations_weighted function definition should be available here)
            # Re-define it or ensure it's in scope if this script is modularized
            def generate_trained_recommendations_weighted(
                trained_model, df_full_data_inf, mlb_season_inf, tokenizer_inf, location_scaler_inf, category_encoder_inf,
                query_season_name: str, user_query_location: tuple = None, max_distance_km: float = None,
                top_n=10, weight_suitability=0.7, weight_proximity=0.3
            ):
                trained_model.eval()
                all_model_scores = []
                temp_text_fillna = df_full_data_inf['ATT_DETAIL_TH'].fillna('')
                inf_text_seq = tokenizer_inf.texts_to_sequences(temp_text_fillna)
                inf_X_text_np = pad_sequences(inf_text_seq, maxlen=MAX_SEQUENCE_LENGTH)
                temp_loc_fillna = df_full_data_inf['ATT_LOCATION'].fillna('0.0,0.0')
                inf_parsed_locs_rec = temp_loc_fillna.apply(lambda x: pd.Series(parse_location(x)))

                # Critical: Use the df_full_data_inf that has ATT_LATITUDE, ATT_LONGITUDE from global scope or ensure it's passed correctly
                # If df_full_data_inf inside this function is a fresh copy without these, Haversine will fail
                # Assuming df_full_data_inf is the global df_all which has these columns added
                
                inf_X_loc_np = location_scaler_inf.transform(inf_parsed_locs_rec.values)
                temp_season_fillna = df_full_data_inf['SUITABLE_SEASON'].fillna("['unknown']") # Uses 'SUITABLE_SEASON' from df_full_data_inf
                temp_season_list_rec = temp_season_fillna.apply(clean_season_string)
                inf_X_item_season_np = mlb_season_inf.transform(temp_season_list_rec)

                temp_cat_fillna = df_full_data_inf['ATTR_CATAGORY_TH'].fillna('Unknown')
                inf_X_cat_np = category_encoder_inf.transform(temp_cat_fillna)
                
                inf_X_text_t = torch.LongTensor(inf_X_text_np).to(device)
                inf_X_loc_t = torch.FloatTensor(inf_X_loc_np).to(device)
                inf_X_item_s_t = torch.FloatTensor(inf_X_item_season_np).to(device)
                inf_X_cat_t = torch.LongTensor(inf_X_cat_np).to(device)

                try:
                    query_s_transformed_inf = mlb_season_inf.transform([[query_season_name]])
                    query_s_tensor_inf = torch.FloatTensor(query_s_transformed_inf).to(device)
                except ValueError:
                    query_s_tensor_inf = torch.zeros(1, len(mlb_season_inf.classes_)).to(device)

                with torch.no_grad():
                    batch_size_inf_rec = 256
                    for i in range(0, len(df_full_data_inf), batch_size_inf_rec):
                        b_text,b_loc,b_item_s,b_cat = inf_X_text_t[i:i+batch_size_inf_rec],inf_X_loc_t[i:i+batch_size_inf_rec],inf_X_item_s_t[i:i+batch_size_inf_rec],inf_X_cat_t[i:i+batch_size_inf_rec]
                        b_query_s = query_s_tensor_inf.repeat(b_text.size(0), 1)
                        outputs = trained_model(b_text, b_loc, b_item_s, b_cat, b_query_s)
                        scores_batch = torch.sigmoid(outputs).squeeze().cpu().numpy()
                        all_model_scores.extend(scores_batch.tolist() if scores_batch.ndim > 0 else [scores_batch.item()])
                
                df_results = df_full_data_inf.copy()
                df_results['suitability_score'] = all_model_scores

                if user_query_location and isinstance(user_query_location, tuple) and len(user_query_location) == 2:
                    user_lat, user_lon = user_query_location
                    df_results['distance_km'] = df_results.apply(
                        lambda row: haversine(user_lat, user_lon, row['ATT_LATITUDE'], row['ATT_LONGITUDE']), axis=1 # Assumes ATT_LATITUDE, ATT_LONGITUDE exist
                    )
                    min_dist = df_results['distance_km'].min()
                    max_dist_val = df_results[df_results['distance_km'] != float('inf')]['distance_km'].max()
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

            user_q_season = "rainy"
            user_q_loc = (13.7563, 100.5018)
            user_max_dist = 200.0

            recommendations = generate_trained_recommendations_weighted(
                loaded_model_for_inference, df_all, mlb_season, tokenizer, location_scaler, category_encoder,
                query_season_name=user_q_season, user_query_location=user_q_loc,
                max_distance_km=user_max_dist, top_n=TOP_N_RECOMMENDATIONS,
                weight_suitability=0.6, weight_proximity=0.4
            )
            if not recommendations.empty:
                print(f"\nTop {TOP_N_RECOMMENDATIONS} recommendations for '{user_q_season}' from MLflow run model:")
                output_cols = ['ATT_NAME_TH', 'suitability_score']
                if 'composite_score' in recommendations.columns: output_cols.append('composite_score')
                if 'distance_km' in recommendations.columns: output_cols.append('distance_km')
                print(recommendations[output_cols])
        except FileNotFoundError:
            print(f"Model weights file {final_model_path_from_mlflow_run} not found. Cannot generate recommendations.")
        except Exception as e:
            print(f"An error occurred during recommendation generation with MLflow run model: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("MLflow run did not complete or run_id not available, skipping inference example with MLflow model.")