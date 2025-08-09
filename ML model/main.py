# main.py - FastAPI Iris Species Prediction Service
# This is the main FastAPI application file that serves ML model predictions via REST API

# ========== IMPORTS ==========
from fastapi import FastAPI, HTTPException  # FastAPI framework and HTTP exception handling
from pydantic import BaseModel, Field       # Data validation and serialization
import joblib                               # For loading saved ML models
import numpy as np                          # Numerical operations for model input
from typing import List                     # Type hints for lists
import os                                   # Operating system interface for file operations

# ========== MODEL VALIDATION ==========
# Check if the trained model file exists before starting the application
if not os.path.exists('iris_model.pkl'):
    raise FileNotFoundError("Model file 'iris_model.pkl' not found. Run MLmodel.py first!")

# Check if the target names file exists (contains species names)
if not os.path.exists('iris_target_names.pkl'):
    raise FileNotFoundError("Target names file not found. Run MLmodel.py first!")

# ========== MODEL LOADING ==========
# Load the trained model and target names at application startup (not on every request)
print("Loading trained model...")
model = joblib.load('iris_model.pkl')              # Load the Random Forest model
target_names = joblib.load('iris_target_names.pkl') # Load species names ['setosa', 'versicolor', 'virginica']
print(f"Model loaded successfully! Species: {list(target_names)}")

# ========== FASTAPI APPLICATION SETUP ==========
# Create the FastAPI application instance with metadata
app = FastAPI(
    title="🌸 Iris Species Prediction API",                    # API title shown in docs
    description="Predict iris species from flower measurements using Random Forest",  # API description
    version="1.0.0",                                           # API version
    docs_url="/docs",                                          # Swagger UI documentation URL
    redoc_url="/redoc"                                         # ReDoc documentation URL
)

# ========== PYDANTIC MODELS (DATA SCHEMAS) ==========

class IrisFeatures(BaseModel):
    """
    Input features for iris species prediction
    This model defines the structure and validation rules for incoming prediction requests
    """
    sepal_length: float = Field(
        ...,                                    # Required field (ellipsis means required)
        ge=0.1,                                # Greater than or equal to 0.1
        le=10.0,                               # Less than or equal to 10.0
        description="Sepal length in centimeters",  # Field description for API docs
        example=5.1                            # Example value shown in API docs
    )
    sepal_width: float = Field(
        ..., 
        ge=0.1, 
        le=10.0,
        description="Sepal width in centimeters", 
        example=3.5
    )
    petal_length: float = Field(
        ..., 
        ge=0.1, 
        le=10.0,
        description="Petal length in centimeters",
        example=1.4
    )
    petal_width: float = Field(
        ..., 
        ge=0.1, 
        le=10.0,
        description="Petal width in centimeters",
        example=0.2
    )

class PredictionResponse(BaseModel):
    """
    Response model for predictions
    Defines the structure of the API response for single predictions
    """
    species: str = Field(description="Predicted iris species")           # e.g., "setosa"
    confidence: float = Field(description="Prediction confidence (0-1)") # e.g., 0.95
    probabilities: dict = Field(description="Probability for each species")  # e.g., {"setosa": 0.95, ...}
    input_features: dict = Field(description="Input features used for prediction")  # Echo back the input

class BatchPredictionRequest(BaseModel):
    """
    Request model for batch predictions
    Allows multiple flower samples to be predicted in one API call
    """
    samples: List[IrisFeatures] = Field(description="List of iris flower measurements")

class BatchPredictionResponse(BaseModel):
    """
    Response model for batch predictions
    Contains results for all samples in the batch request
    """
    predictions: List[PredictionResponse]  # List of individual prediction responses
    total_samples: int                     # Total number of samples processed

# ========== API ENDPOINTS ==========

# Root endpoint - provides API information and available endpoints
@app.get("/")
def root():
    """
    Welcome message and API information
    This endpoint provides an overview of the API and its capabilities
    """
    return {
        "message": "🌸 Welcome to Iris Species Prediction API!",
        "description": "Predict iris species from flower measurements",
        "endpoints": {
            "predict": "/predict - Single prediction",
            "predict_batch": "/predict-batch - Multiple predictions",
            "health": "/health - API health check",
            "docs": "/docs - Interactive API documentation"
        },
        "model_info": {
            "species": list(target_names),  # Available species the model can predict
            "features_required": ["sepal_length", "sepal_width", "petal_length", "petal_width"]
        }
    }

# Health check endpoint - verifies that the API and model are working correctly
@app.get("/health")
def health_check():
    """
    Check if the API and model are working
    Performs a test prediction to ensure the model is loaded and functional
    """
    try:
        # Test prediction with sample data (typical setosa measurements)
        test_features = np.array([[5.1, 3.5, 1.4, 0.2]])
        prediction = model.predict(test_features)
        return {
            "status": "healthy",                        # API status
            "model_loaded": True,                       # Confirms model is loaded
            "test_prediction": target_names[prediction[0]]  # Test prediction result
        }
    except Exception as e:
        # If health check fails, return HTTP 500 error
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

# Single prediction endpoint - the main prediction functionality
@app.post("/predict", response_model=PredictionResponse)
def predict_species(features: IrisFeatures):
    """
    Predict iris species from flower measurements
    
    Takes flower measurements and returns the predicted species with confidence scores
    This is the core functionality of the API
    """
    try:
        # Convert Pydantic model to numpy array for scikit-learn model
        input_array = np.array([[
            features.sepal_length,
            features.sepal_width, 
            features.petal_length,
            features.petal_width
        ]])
        
        # Make prediction using the loaded Random Forest model
        prediction = model.predict(input_array)           # Returns class index (0, 1, or 2)
        prediction_proba = model.predict_proba(input_array)  # Returns probabilities for all classes
        
        # Get the predicted species name from the class index
        predicted_species = target_names[prediction[0]]
        
        # Get confidence (highest probability among all classes)
        confidence = float(np.max(prediction_proba[0]))
        
        # Create probabilities dictionary mapping species names to their probabilities
        probabilities = {
            species: float(prob) 
            for species, prob in zip(target_names, prediction_proba[0])
        }
        
        # Create input features dictionary for response (echo back what was sent)
        input_features_dict = {
            "sepal_length": features.sepal_length,
            "sepal_width": features.sepal_width,
            "petal_length": features.petal_length,
            "petal_width": features.petal_width
        }
        
        # Return structured response using Pydantic model
        return PredictionResponse(
            species=predicted_species,
            confidence=confidence,
            probabilities=probabilities,
            input_features=input_features_dict
        )
        
    except Exception as e:
        # Handle any errors during prediction
        raise HTTPException(
            status_code=500, 
            detail=f"Prediction error: {str(e)}"
        )

# Batch prediction endpoint - handles multiple samples at once
@app.post("/predict-batch", response_model=BatchPredictionResponse)
def predict_batch(request: BatchPredictionRequest):
    """
    Predict iris species for multiple flower samples
    
    Useful for processing multiple measurements at once
    More efficient than making multiple individual API calls
    """
    try:
        predictions = []
        
        # Process each sample in the batch
        for sample in request.samples:
            # Reuse the single prediction logic for each sample
            prediction_response = predict_species(sample)
            predictions.append(prediction_response)
        
        # Return batch response with all predictions
        return BatchPredictionResponse(
            predictions=predictions,
            total_samples=len(predictions)
        )
        
    except Exception as e:
        # Handle batch prediction errors
        raise HTTPException(
            status_code=500,
            detail=f"Batch prediction error: {str(e)}"
        )

# Model information endpoint - provides insights about the trained model
@app.get("/model-info")
def get_model_info():
    """
    Get information about the trained model
    Returns model metadata, feature importance, and other useful information
    """
    try:
        # Define feature names in the same order as training data
        feature_names = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
        
        # Get feature importance from the Random Forest model
        feature_importance = model.feature_importances_
        
        # Create dictionary mapping feature names to their importance scores
        importance_dict = {
            name: float(importance) 
            for name, importance in zip(feature_names, feature_importance)
        }
        
        # Return comprehensive model information
        return {
            "model_type": "Random Forest Classifier",       # Type of ML model
            "n_estimators": model.n_estimators,            # Number of trees in the forest
            "species": list(target_names),                  # Species the model can predict
            "feature_importance": importance_dict,          # Importance score for each feature
            "most_important_features": sorted(              # Features ranked by importance
                importance_dict.items(), 
                key=lambda x: x[1],                        # Sort by importance value
                reverse=True                               # Highest importance first
            )
        }
    except Exception as e:
        # Handle model info retrieval errors
        raise HTTPException(
            status_code=500,
            detail=f"Model info error: {str(e)}"
        )

# ========== APPLICATION STARTUP ==========
# This block runs only when the script is executed directly (not imported)
if __name__ == "__main__":
    import uvicorn  # ASGI server for running FastAPI
    print("🚀 Starting Iris Species Prediction API...")
    print("📖 Visit http://localhost:8000/docs for interactive documentation")
    # Start the server with configuration
    uvicorn.run(
        app,                    # FastAPI application instance
        host="0.0.0.0",        # Listen on all network interfaces
        port=8000              # Port number for the API
    )