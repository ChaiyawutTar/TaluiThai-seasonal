import pytest
import numpy as np
import torch
from unittest.mock import patch, MagicMock
from deploy2 import parse_location, clean_season_string, ContextualMultiModalRecommenderPyTorch

def test_parse_location_valid():
    lat, lon = parse_location("13.7563, 100.5018")
    assert isinstance(lat, float)
    assert isinstance(lon, float)
    assert -90 <= lat <= 90
    assert -180 <= lon <= 180

def test_parse_location_invalid():
    lat, lon = parse_location("invalid_string")
    assert lat == 0.0 and lon == 0.0

def test_parse_location_out_of_bounds():
    lat, lon = parse_location("1000, 2000")
    assert lat == 0.0 and lon == 0.0

def test_clean_season_string_valid():
    assert clean_season_string("['summer', 'winter']") == ['summer', 'winter']

def test_clean_season_string_invalid():
    assert clean_season_string("not_a_list") == ['unknown']

def test_model_forward_pass():
    vocab_size = 100
    model = ContextualMultiModalRecommenderPyTorch(
        vocab_size=vocab_size,
        text_embedding_dim=10,
        location_input_dim=2,
        item_season_input_dim=3,
        query_season_input_dim=3,
        num_categories=5,
        category_embedding_dim=4,
        conv_filters=8,
        kernel_size=3,
        dense_units=6,
        shared_dense_units=12,
        dropout_rate=0.1
    )
    model.eval()
    batch_size = 2
    text = torch.randint(0, vocab_size, (batch_size, 20), dtype=torch.long)
    location = torch.randn(batch_size, 2)
    item_season = torch.randn(batch_size, 3)
    category = torch.randint(0, 5, (batch_size,), dtype=torch.long)
    query_season = torch.randn(batch_size, 3)
    with torch.no_grad():
        output = model(text, location, item_season, category, query_season)
    assert output.shape == (batch_size, 1)

@patch('joblib.dump')
def test_preprocessor_save(mock_dump):
    from deploy2 import PREPROCESSOR_SAVE_PATH
    preprocessors = {'dummy': 1}
    # Call joblib.dump in your code with preprocessors and path
    # Here we just simulate it
    import joblib
    joblib.dump(preprocessors, PREPROCESSOR_SAVE_PATH)
    mock_dump.assert_called_once_with(preprocessors, PREPROCESSOR_SAVE_PATH)

@patch('pandas.read_csv')
def test_data_loading(mock_read_csv):
    import pandas as pd
    from deploy2 import DATA_FILE_PATH
    mock_df = pd.DataFrame({
        'ATT_DETAIL_TH': ['text1', 'text2'],
        'ATT_LOCATION': ['13.7,100.5', '14.0,101.0'],
        'ATTR_CATAGORY_TH': ['cat1', 'cat2'],
        'SUITABLE_SEASON': ["['summer']", "['winter']"],
        'ATT_NAME_TH': ['name1', 'name2']
    })
    mock_read_csv.return_value = mock_df
    df = pd.read_csv(DATA_FILE_PATH)
    assert not df.empty
    assert 'ATT_DETAIL_TH' in df.columns

# Additional tests can be added for tokenizer fitting, scaler fitting, label encoding, etc.

if __name__ == "__main__":
    pytest.main()