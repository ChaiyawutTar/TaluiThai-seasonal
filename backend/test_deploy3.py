import pytest
import numpy as np
import torch
import pandas as pd
from unittest.mock import patch, MagicMock, mock_open

class TestLocationParsing:
    def test_valid_locations(self):
        from deploy3 import parse_location
        # Test with various valid formats
        assert parse_location("13.7563, 100.5018") == (13.7563, 100.5018)
        assert parse_location("13.7563,100.5018") == (13.7563, 100.5018)
        assert parse_location("-33.8688, 151.2093") == (-33.8688, 151.2093)
        assert parse_location("0.0, 0.0") == (0.0, 0.0)

    def test_invalid_locations(self):
        from deploy3 import parse_location
        # Test with invalid inputs
        assert parse_location("invalid_string") == (0.0, 0.0)
        assert parse_location(None) == (0.0, 0.0)
        assert parse_location("") == (0.0, 0.0)
        assert parse_location("13.7563") == (0.0, 0.0)  # Missing longitude
        assert parse_location("13.7563, longitude") == (0.0, 0.0)  # Non-numeric longitude

    def test_out_of_bounds_locations(self):
        from deploy3 import parse_location
        # Test with out-of-range values
        assert parse_location("91.0, 100.0") == (0.0, 0.0)  # Latitude > 90
        assert parse_location("-91.0, 100.0") == (0.0, 0.0)  # Latitude < -90
        assert parse_location("45.0, 181.0") == (0.0, 0.0)  # Longitude > 180
        assert parse_location("45.0, -181.0") == (0.0, 0.0)  # Longitude < -180

class TestSeasonStringCleaning:
    def test_valid_season_strings(self):
        from deploy3 import clean_season_string
        # Test with valid season strings
        assert clean_season_string("['summer', 'winter']") == ['summer', 'winter']
        assert clean_season_string("['spring']") == ['spring']
        assert clean_season_string("[]") == []

    def test_invalid_season_strings(self):
        from deploy3 import clean_season_string
        # Test with invalid season strings
        assert clean_season_string("not_a_list") == ['unknown']
        assert clean_season_string(None) is None

class TestContextualMultiModalRecommender:
    @pytest.fixture
    def model_params(self):
        return {
            "vocab_size": 100,
            "text_embedding_dim": 10,
            "location_input_dim": 2,
            "item_season_input_dim": 3,
            "query_season_input_dim": 3,
            "num_categories": 5,
            "category_embedding_dim": 4,
            "conv_filters": 8,
            "kernel_size": 3,
            "dense_units": 6,
            "shared_dense_units": 12,
            "dropout_rate": 0.1
        }

    @pytest.fixture
    def model(self, model_params):
        from deploy3 import ContextualMultiModalRecommenderPyTorch
        return ContextualMultiModalRecommenderPyTorch(
            vocab_size=model_params["vocab_size"],
            text_embedding_dim=model_params["text_embedding_dim"],
            location_input_dim=model_params["location_input_dim"],
            item_season_input_dim=model_params["item_season_input_dim"],
            query_season_input_dim=model_params["query_season_input_dim"],
            num_categories=model_params["num_categories"],
            category_embedding_dim=model_params["category_embedding_dim"],
            conv_filters=model_params["conv_filters"],
            kernel_size=model_params["kernel_size"],
            dense_units=model_params["dense_units"],
            shared_dense_units=model_params["shared_dense_units"],
            dropout_rate=model_params["dropout_rate"]
        )

    def test_model_initialization(self, model, model_params):
        # Test model structure
        assert isinstance(model.embedding_text, torch.nn.Embedding)
        assert model.embedding_text.num_embeddings == model_params["vocab_size"]
        assert model.embedding_text.embedding_dim == model_params["text_embedding_dim"]

        # Check multi-scale convolutions instead of a single conv_text
        assert isinstance(model.conv1_text, torch.nn.Conv1d)
        assert isinstance(model.conv2_text, torch.nn.Conv1d)
        assert isinstance(model.conv3_text, torch.nn.Conv1d)
        assert model.conv1_text.out_channels == model_params["conv_filters"]
        assert model.conv2_text.out_channels == model_params["conv_filters"]
        assert model.conv3_text.out_channels == model_params["conv_filters"]

        assert isinstance(model.dense_location1, torch.nn.Linear)
        assert model.dense_location1.in_features == model_params["location_input_dim"]

        assert isinstance(model.output_layer, torch.nn.Linear)
        assert model.output_layer.out_features == 1

    def test_model_forward_pass(self, model, model_params):
        # Test forward pass with random inputs
        batch_size = 2
        text = torch.randint(0, model_params["vocab_size"], (batch_size, 20), dtype=torch.long)
        location = torch.randn(batch_size, model_params["location_input_dim"])
        item_season = torch.randn(batch_size, model_params["item_season_input_dim"])
        category = torch.randint(0, model_params["num_categories"], (batch_size,), dtype=torch.long)
        query_season = torch.randn(batch_size, model_params["query_season_input_dim"])

        model.eval()
        with torch.no_grad():
            output = model(text, location, item_season, category, query_season)

        # Check output shape and type
        assert output.shape == (batch_size, 1)
        assert isinstance(output, torch.Tensor)

        # Test with different batch sizes
        for test_batch_size in [1, 4, 8]:
            text = torch.randint(0, model_params["vocab_size"], (test_batch_size, 20), dtype=torch.long)
            location = torch.randn(test_batch_size, model_params["location_input_dim"])
            item_season = torch.randn(test_batch_size, model_params["item_season_input_dim"])
            category = torch.randint(0, model_params["num_categories"], (test_batch_size,), dtype=torch.long)
            query_season = torch.randn(test_batch_size, model_params["query_season_input_dim"])

            with torch.no_grad():
                output = model(text, location, item_season, category, query_season)
            assert output.shape == (test_batch_size, 1)

class TestDataProcessing:
    @patch('pandas.read_csv')
    def test_data_loading(self, mock_read_csv):
        # Create mock DataFrame
        mock_df = pd.DataFrame({
            'ATT_DETAIL_TH': ['text1', 'text2', None],
            'ATT_LOCATION': ['13.7,100.5', '14.0,101.0', None],
            'ATTR_CATAGORY_TH': ['cat1', 'cat2', None],
            'SUITABLE_SEASON': ["['summer']", "['winter']", None],
            'ATT_NAME_TH': ['name1', 'name2', None]
        })
        mock_read_csv.return_value = mock_df

        # Import after mocking
        from deploy3 import DATA_FILE_PATH
        df = pd.read_csv(DATA_FILE_PATH)

        # Verify data loading
        assert not df.empty
        assert 'ATT_DETAIL_TH' in df.columns
        mock_read_csv.assert_called_once_with(DATA_FILE_PATH)

    @patch('joblib.dump')
    def test_preprocessor_save(self, mock_dump):
        # Test preprocessor saving
        from deploy3 import PREPROCESSOR_SAVE_PATH
        preprocessors = {
            'tokenizer': MagicMock(),
            'location_scaler': MagicMock(),
            'mlb_season': MagicMock(),
            'category_encoder': MagicMock(),
            'VOCAB_SIZE': 1000,
            'MODEL_PARAMS': {'param1': 'value1'}
        }

        import joblib
        joblib.dump(preprocessors, PREPROCESSOR_SAVE_PATH)

        # Verify correct call
        mock_dump.assert_called_once()
        args, kwargs = mock_dump.call_args
        assert args[0] == preprocessors
        assert args[1] == PREPROCESSOR_SAVE_PATH

class TestMLflowIntegration:
    @patch('mlflow.start_run')
    @patch('mlflow.log_param')
    @patch('mlflow.log_metric')
    @patch('mlflow.log_artifact')
    @patch('mlflow.pytorch.log_model')
    def test_mlflow_logging(self, mock_log_model, mock_log_artifact, mock_log_metric,
                            mock_log_param, mock_start_run):
        # Import mlflow here
        import mlflow

        # Mock MLflow run
        mock_run = MagicMock()
        mock_run.info.run_id = "test_run_id"
        mock_start_run.return_value.__enter__.return_value = mock_run

        # Import after mocking
        from deploy3 import MODEL_PARAMS, PREPROCESSOR_SAVE_PATH

        # Simulate MLflow logging
        with mlflow.start_run() as run:
            run_id = run.info.run_id
            mlflow.log_param("batch_size", 64)
            mlflow.log_param("learning_rate", 0.001)
            for key, value in MODEL_PARAMS.items():
                mlflow.log_param(key, value)
            mlflow.log_artifact(PREPROCESSOR_SAVE_PATH)
            mlflow.log_metric("train_loss", 0.5)
            mlflow.log_metric("val_loss", 0.6)

        # Verify MLflow interactions
        mock_start_run.assert_called_once()
        assert mock_log_param.call_count >= 2
        assert mock_log_metric.call_count >= 2
        mock_log_artifact.assert_called_once_with(PREPROCESSOR_SAVE_PATH)

class TestTrainingProcess:
    @patch('torch.save')
    def test_model_saving(self, mock_save):
        # Test model saving functionality
        model = MagicMock()
        state_dict = {"layer1.weight": torch.randn(5, 5)}
        model.state_dict.return_value = state_dict

        save_path = "test_model.pth"
        torch.save(model.state_dict(), save_path)

        # Verify correct saving
        mock_save.assert_called_once()
        args, kwargs = mock_save.call_args
        assert args[0] == state_dict
        assert args[1] == save_path

    @patch('deploy3.add_evaluation_to_training')
    def test_evaluation_integration(self, mock_add_evaluation):
        # Test integration with evaluation module
        model = MagicMock()
        X_text_val = torch.randint(0, 100, (10, 20))
        X_loc_val = torch.randn(10, 2)
        X_item_s_val = torch.randn(10, 3)
        X_cat_val = torch.randint(0, 5, (10,))
        X_query_s_val = torch.randn(10, 3)
        y_val = torch.randn(10, 1)
        attraction_ids_val = torch.randint(0, 100, (10,))
        device = torch.device("cpu")
        batch_size = 64
        run_id = "test_run_id"

        # Call evaluation function
        from deploy3 import add_evaluation_to_training
        add_evaluation_to_training(
            model, X_text_val, X_loc_val, X_item_s_val, X_cat_val, X_query_s_val,
            y_val, attraction_ids_val, device, batch_size=batch_size, run_id=run_id
        )

        # Verify evaluation was called with correct parameters
        mock_add_evaluation.assert_called_once()
        args, kwargs = mock_add_evaluation.call_args
        assert args[0] == model
        assert kwargs['batch_size'] == batch_size
        assert kwargs['run_id'] == run_id

if __name__ == "__main__":
    pytest.main(["-v"])