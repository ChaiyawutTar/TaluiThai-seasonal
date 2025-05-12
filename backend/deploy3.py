import pandas as pd
import numpy as np
import re
from math import radians, sin, cos, sqrt, atan2
import joblib # For saving preprocessors

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler, LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from mlflow.models.signature import infer_signature

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import mlflow
import mlflow.pytorch
# import shap # Uncomment if you have SHAP installed and want to run the SHAP part

# Import the evaluation metrics module
from eval import add_evaluation_to_training

# --- Configuration & Constants ---
DATA_FILE_PATH = 'allattractions_with_season.csv'
MODEL_SAVE_PATH_BASE = "contextual_recommender_v4" # Base name for MLflow saving
PREPROCESSOR_SAVE_PATH = "recommender_preprocessors.joblib" # Path to save fitted preprocessors

# Preprocessing constants
MAX_WORDS = 10000
MAX_SEQUENCE_LENGTH = 200

# Training constants
BATCH_SIZE = 64
NUM_EPOCHS = 10 # As per your last script version
LEARNING_RATE = 0.001

# Evaluation constants
EVALUATION_K_VALUES = [1, 5, 10, 20]  # k values for top-k metrics

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
            if -90 <= lat <= 90 and -180 <= lon <= 180:
                return lat, lon
    except:
        pass
    return 0.0, 0.0

def clean_season_string(season_str):
    try: return eval(str(season_str))
    except: return ['unknown']

# --- Main Script Execution ---
if __name__ == '__main__':
    print("--- Loading and Preprocessing Data ---")
    try:
        df_all = pd.read_csv(DATA_FILE_PATH)
        print(f"Successfully loaded {DATA_FILE_PATH}. Shape: {df_all.shape}")
    except FileNotFoundError:
        print(f"Error: {DATA_FILE_PATH} not found.")
        exit()

    df_all['ATT_DETAIL_TH'] = df_all['ATT_DETAIL_TH'].fillna('')
    df_all['ATT_LOCATION'] = df_all['ATT_LOCATION'].fillna('0.0,0.0')
    df_all['ATTR_CATAGORY_TH'] = df_all['ATTR_CATAGORY_TH'].fillna('Unknown')
    df_all['SUITABLE_SEASON'] = df_all['SUITABLE_SEASON'].fillna("['unknown']")
    df_all['ATT_NAME_TH'] = df_all['ATT_NAME_TH'].fillna('Unnamed Attraction')

    # Fit Text Tokenizer
    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<unk>")
    tokenizer.fit_on_texts(df_all['ATT_DETAIL_TH'])
    VOCAB_SIZE = len(tokenizer.word_index) + 1
    print(f"Text Vocab Size: {VOCAB_SIZE}")

    # Fit Location Scaler
    parsed_locs = df_all['ATT_LOCATION'].apply(lambda x: pd.Series(parse_location(x)))
    # df_all[['ATT_LATITUDE', 'ATT_LONGITUDE']] = parsed_locs # Keep for reference if needed by other parts of df_all
    location_scaler = StandardScaler().fit(parsed_locs.values)
    LOCATION_INPUT_DIM = parsed_locs.shape[1]

    # Fit Season MultiLabelBinarizer
    df_all['SUITABLE_SEASON_LIST'] = df_all['SUITABLE_SEASON'].apply(clean_season_string)
    mlb_season = MultiLabelBinarizer()
    mlb_season.fit(df_all['SUITABLE_SEASON_LIST'])
    ITEM_SEASON_INPUT_DIM = len(mlb_season.classes_)
    QUERY_SEASON_INPUT_DIM = ITEM_SEASON_INPUT_DIM
    POSSIBLE_QUERY_SEASONS = [s for s in mlb_season.classes_.tolist() if s != 'unknown' and s.strip() != '']
    print(f"Season Classes: {mlb_season.classes_}, Possible Query Seasons for Training: {POSSIBLE_QUERY_SEASONS}")

    # Fit Category LabelEncoder
    category_encoder = LabelEncoder().fit(df_all['ATTR_CATAGORY_TH'])
    NUM_CATEGORIES = len(category_encoder.classes_)
    print(f"Num Categories: {NUM_CATEGORIES}")

    # --- Save fitted preprocessors ---
    preprocessors = {
        'tokenizer': tokenizer,
        'location_scaler': location_scaler,
        'mlb_season': mlb_season,
        'category_encoder': category_encoder,
        'VOCAB_SIZE': VOCAB_SIZE, # Save derived constants too
        'LOCATION_INPUT_DIM': LOCATION_INPUT_DIM,
        'ITEM_SEASON_INPUT_DIM': ITEM_SEASON_INPUT_DIM,
        'QUERY_SEASON_INPUT_DIM': QUERY_SEASON_INPUT_DIM,
        'NUM_CATEGORIES': NUM_CATEGORIES,
        'MAX_SEQUENCE_LENGTH': MAX_SEQUENCE_LENGTH, # Save this for consistency
        'MODEL_PARAMS': MODEL_PARAMS # Save model structure params
    }
    joblib.dump(preprocessors, PREPROCESSOR_SAVE_PATH)
    print(f"Preprocessors saved to {PREPROCESSOR_SAVE_PATH}")


    # Transform all data using fitted preprocessors (for generating expanded training set)
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
                'attraction_id': idx
            })
    df_train_expanded = pd.DataFrame(train_samples_list)
    print(f"Expanded training dataset size: {len(df_train_expanded)}")

    X_text_t_exp = torch.LongTensor(np.array(df_train_expanded['text'].tolist()))
    X_loc_t_exp = torch.FloatTensor(np.array(df_train_expanded['location'].tolist()))
    X_item_s_t_exp = torch.FloatTensor(np.array(df_train_expanded['item_season'].tolist()))
    X_cat_t_exp = torch.LongTensor(np.array(df_train_expanded['category'].tolist()))
    X_query_s_t_exp = torch.FloatTensor(np.array(df_train_expanded['query_season'].tolist()))
    y_labels_t_exp = torch.FloatTensor(df_train_expanded['label'].values).unsqueeze(1)
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
    if not can_stratify: print("Warning: Splitting without stratify.")
    
    X_text_train, X_text_val = X_text_t_exp[train_indices], X_text_t_exp[val_indices]
    X_loc_train, X_loc_val = X_loc_t_exp[train_indices], X_loc_t_exp[val_indices]
    X_item_s_train, X_item_s_val = X_item_s_t_exp[train_indices], X_item_s_t_exp[val_indices]
    X_cat_train, X_cat_val = X_cat_t_exp[train_indices], X_cat_t_exp[val_indices]
    X_query_s_train, X_query_s_val = X_query_s_t_exp[train_indices], X_query_s_t_exp[val_indices]
    y_train, y_val = y_labels_t_exp[train_indices], y_labels_t_exp[val_indices]
    attraction_ids_train = attraction_ids_exp[train_indices]
    attraction_ids_val = attraction_ids_exp[val_indices]  # For evaluation metrics

    train_dataset = TensorDataset(X_text_train, X_loc_train, X_item_s_train, X_cat_train, X_query_s_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataset = TensorDataset(X_text_val, X_loc_val, X_item_s_val, X_cat_val, X_query_s_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    mlflow.set_experiment("Attraction Recommender Training")
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"MLflow Run ID: {run_id}")
        current_model_save_path = f"./models_save/{MODEL_SAVE_PATH_BASE}_{run_id}.pth"

        # Log parameters
        mlflow.log_param("data_file_path", DATA_FILE_PATH)
        mlflow.log_param("max_words", MAX_WORDS)
        mlflow.log_param("max_sequence_length", MAX_SEQUENCE_LENGTH)
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("num_epochs", NUM_EPOCHS)
        mlflow.log_param("learning_rate", LEARNING_RATE)
        for key, value in MODEL_PARAMS.items(): mlflow.log_param(key, value)
        mlflow.log_artifact(PREPROCESSOR_SAVE_PATH) # Log the saved preprocessors

        # Initialize model
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
            # Training phase
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

            # Validation phase
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
                torch.save(model.state_dict(), current_model_save_path) # Save model with run_id
                print(f"  Model improved and saved to {current_model_save_path}")

        # Load the best model for evaluation
        print("\n--- Loading best model for evaluation ---")
        model.load_state_dict(torch.load(current_model_save_path))
        model.eval()
        
        # Run evaluation metrics
        add_evaluation_to_training(
            model, 
            X_text_val, X_loc_val, X_item_s_val, X_cat_val, X_query_s_val, 
            y_val, attraction_ids_val,
            device, 
            batch_size=BATCH_SIZE,
            run_id=run_id
        )

        # Log the model to MLflow
        input_example = (
            torch.randint(0, VOCAB_SIZE, (1, MAX_SEQUENCE_LENGTH)).to(device),       
            torch.rand(1, LOCATION_INPUT_DIM).to(device),                            # b_loc
            torch.rand(1, ITEM_SEASON_INPUT_DIM).to(device),                         # b_item_s
            torch.randint(0, NUM_CATEGORIES, (1,)).to(device),                       # b_cat
            torch.rand(1, QUERY_SEASON_INPUT_DIM).to(device)                         # b_query_s
        )

        example_output = model(*input_example)
        signature = infer_signature(input_example, example_output)
        mlflow.pytorch.log_model(model, "recommender_model", registered_model_name="ContextualRecommenderPyTorch", signature=signature)
        mlflow.log_metric("best_val_loss", best_val_loss)
        print(f"\n--- Training Complete. Best model for Run ID {run_id} saved to {current_model_save_path} ---")
        
        # Create a test set for final evaluation if needed
        # Here you could create a separate test set or use a portion of validation data
        
        print("\n--- Final Model Evaluation ---")
        print("The model has been evaluated using the following metrics:")
        print("- Precision@k: Proportion of recommended items that are relevant")
        print("- Recall@k: Proportion of relevant items that are recommended")
        print("- NDCG@k: Normalized Discounted Cumulative Gain (measures ranking quality)")
        print("- MRR: Mean Reciprocal Rank (position of first relevant item)")
        print("These metrics provide a complete picture of recommendation quality.")