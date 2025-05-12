# TaluiThai Seasonal Reccomendation

## Setup Instructions

1. Install required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Create a directory for model storage:
   ```
   mkdir -p backend/models_save
   ```

## Project Structure

### Backend

- **Data Processing**: 
  - `backend/preprocessing_pipeline.ipynb`: Processes the initial dataset (`allattractions.csv`) to generate `allattractions_with_season.csv`

- **Model Training**:
  - `backend/deploy3.py`: Handles model training, MLflow tracking and evaluate model

- **API Service**:
  - `backend/app2.py`: Flask API endpoint (runs on localhost:5004)
  - Requires MLflow ID input for model selection (after training model MLflow will generate ID you must input ID into it)

- **Model Storage**:
  - Models are saved as .pth files in `backend/models_save/`

### Frontend

- React application running on localhost:3000

## Running the Application

### Backend

1. **Data Preprocessing**:
   - Run the preprocessing notebook to prepare your data
   - Input: `allattractions.csv`
   - Output: `allattractions_with_season.csv`

2. **Model Training**:
   - Run the deployment script to train models
   - Models will be tracked with MLflow
   ```
   cd backend
   python deploy3.py
   
   // This will get MLflow ID 
   ```

3. **Start API Server**:
   ```
   cd backend
   python app2.py <MLflow ID>
   ```
   - Access the API at http://localhost:5004
   - You'll need to input your MLflow model ID

4. **MLflow Tracking**:
   ```
   mlflow ui --port <desired_port>
   ```
   - View model versions and performance metrics

### Frontend

1. Install dependencies:
   ```
   cd frontend
   npm install
   ```

2. Start development server:
   ```
   npm run dev
   ```
   - Access the application at http://localhost:3000

## Testing

Run unit tests with:
```
cd backend
pytest test_deploy2.py
```
