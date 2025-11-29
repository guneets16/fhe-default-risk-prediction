"""
FastAPI Backend for FHE Credit Default Risk Prediction

This module provides the REST API for the FHE credit default risk prediction application.
It handles encrypted data processing and secure inference.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List
import base64
import os
import time
from fhe_model import FHEModel

# Initialize FastAPI app
app = FastAPI(
    title="FHE Credit Default Risk Prediction API",
    description="Privacy-preserving credit card default risk prediction using Fully Homomorphic Encryption",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize FHE model
try:
    fhe_model = FHEModel("models/credit_model.pkl")
    print("✅ FHE model loaded successfully")
except FileNotFoundError:
    print("⚠️  Model file not found. Creating sample model...")
    from fhe_model import create_sample_model
    create_sample_model()
    fhe_model = FHEModel("models/credit_model.pkl")
    print("✅ Sample FHE model created and loaded")


# Pydantic models for API
class PredictionRequest(BaseModel):
    """Request model for credit default risk prediction."""
    features: List[float]
    
    class Config:
        json_schema_extra = {
            "example": {
                "features": [50000, 30, 24, 0.3]  # income, age, credit_history, debt_ratio
            }
        }


class PredictionResponse(BaseModel):
    """Response model for credit default risk prediction."""
    prediction: int
    probability: float
    confidence: str
    processing_time: float
    model_info: dict
    
    class Config:
        json_schema_extra = {
            "example": {
                "prediction": 1,
                "probability": 0.75,
                "confidence": "High",
                "processing_time": 2.5,
                "model_info": {
                    "model_type": "LogisticRegression",
                    "fhe_scheme": "CKKS"
                }
            }
        }


class ContextResponse(BaseModel):
    """Response model for public context."""
    public_context: str
    context_info: dict
    
    class Config:
        json_schema_extra = {
            "example": {
                "public_context": "base64_encoded_context",
                "context_info": {
                    "scheme": "CKKS",
                    "poly_modulus_degree": 8192
                }
            }
        }


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    timestamp: float
    model_loaded: bool
    fhe_ready: bool


# API Endpoints
@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint with health check."""
    return HealthResponse(
        status="healthy",
        timestamp=time.time(),
        model_loaded=fhe_model is not None,
        fhe_ready=fhe_model.context is not None
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp=time.time(),
        model_loaded=fhe_model is not None,
        fhe_ready=fhe_model.context is not None
    )


@app.get("/context", response_model=ContextResponse)
async def get_public_context():
    """
    Get public context for client-side encryption.
    
    This endpoint provides the public context needed for clients
    to encrypt their data before sending it to the server.
    """
    try:
        public_context = fhe_model.get_public_context()
        context_b64 = base64.b64encode(public_context).decode()
        
        return ContextResponse(
            public_context=context_b64,
            context_info={
                "scheme": "CKKS",
                "poly_modulus_degree": 8192,
                "coeff_mod_bit_sizes": [60, 40, 40, 60]
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get context: {str(e)}")


@app.post("/predict", response_model=PredictionResponse)
async def predict_default_risk(request: PredictionRequest):
    """
    Predict credit default risk using FHE.
    
    This endpoint performs encrypted inference on the provided features.
    The server never sees the raw data - only encrypted computations.
    """
    start_time = time.time()
    
    try:
        # Validate input - UCI dataset has 23 features
        if len(request.features) != 19:
            raise HTTPException(
                status_code=400, 
                detail=f"Expected exactly 19 features (demographic features and age excluded for fairness) (UCI dataset format), got {len(request.features)}"
            )
        
        # Features are: X1-X23
        # X1: Credit Limit, X2: Gender, X3: Education, X4: Marriage, X5: Age
        # X6-X11: Payment History, X12-X17: Bill Statements, X18-X23: Previous Payments
        
        # Perform FHE prediction
        encrypted_features = fhe_model.encrypt_features(request.features)
        encrypted_result = fhe_model.predict_encrypted(encrypted_features)
        probability, prediction = fhe_model.decrypt_result(encrypted_result)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Determine confidence level
        confidence_level = "High" if abs(probability - 0.5) > 0.3 else "Medium" if abs(probability - 0.5) > 0.1 else "Low"
        
        # Get model info
        model_info = fhe_model.get_model_info()
        
        return PredictionResponse(
            prediction=prediction,
            probability=probability,
            confidence=confidence_level,
            processing_time=processing_time,
            model_info=model_info
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/clear", response_model=PredictionResponse)
async def predict_default_risk_clear(request: PredictionRequest):
    """
    Predict credit default risk using clear (non-encrypted) data.
    
    This endpoint is for comparison and testing purposes.
    In production, this should be disabled for security.
    """
    start_time = time.time()
    
    try:
        # Validate input - UCI dataset has 23 features
        if len(request.features) != 19:
            raise HTTPException(
                status_code=400, 
                detail=f"Expected exactly 19 features (demographic features and age excluded for fairness) (UCI dataset format), got {len(request.features)}"
            )
        
        # Perform clear prediction
        probability, prediction = fhe_model.predict_clear(request.features)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Determine confidence level
        confidence_level = "High" if abs(probability - 0.5) > 0.3 else "Medium" if abs(probability - 0.5) > 0.1 else "Low"
        
        # Get model info
        model_info = fhe_model.get_model_info()
        
        return PredictionResponse(
            prediction=prediction,
            probability=probability,
            confidence=confidence_level,
            processing_time=processing_time,
            model_info=model_info
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Clear prediction failed: {str(e)}")


@app.get("/model/info")
async def get_model_info():
    """Get information about the loaded model."""
    try:
        return fhe_model.get_model_info()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")


@app.get("/models/list")
async def list_all_models():
    """
    Get list of all available models with their metadata.
    
    Returns information about all trained models including:
    - Performance metrics (accuracy, F1, ROC AUC)
    - Model parameters
    - FHE compatibility
    """
    try:
        metadata_path = "models/all_models_metadata.json"
        if os.path.exists(metadata_path):
            import json
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            return metadata
        else:
            raise HTTPException(status_code=404, detail="Model metadata not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load models list: {str(e)}")


@app.post("/predict/model/{model_name}")
async def predict_with_specific_model(model_name: str, request: PredictionRequest):
    """
    Make prediction using a specific model.
    
    Args:
        model_name: Name of the model (e.g., "Logistic Regression", "Random Forest", "Gradient Boosting")
        request: PredictionRequest with 23 features
    
    Returns:
        Prediction result with probability and model information
    
    Note: Only Logistic Regression supports FHE encryption.
    """
    start_time = time.time()
    
    try:
        # Validate input
        if len(request.features) != 19:
            raise HTTPException(
                status_code=400,
                detail=f"Expected exactly 20 features (demographic features excluded for fairness), got {len(request.features)}"
            )
        
        # Load the requested model
        model_filename = model_name.lower().replace(' ', '_') + '.pkl'
        model_path = f"models/{model_filename}"
        
        if not os.path.exists(model_path):
            raise HTTPException(
                status_code=404,
                detail=f"Model '{model_name}' not found. Available models: Logistic Regression, Random Forest, Gradient Boosting"
            )
        
        import joblib
        import numpy as np
        
        model = joblib.load(model_path)
        scaler = joblib.load("models/scaler.pkl")
        
        # Scale features
        features_array = np.array(request.features).reshape(1, -1)
        features_scaled = scaler.transform(features_array)
        
        # Make prediction
        prediction = int(model.predict(features_scaled)[0])
        
        if hasattr(model, 'predict_proba'):
            probability = float(model.predict_proba(features_scaled)[0][1])
        else:
            probability = 0.5  # Fallback
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Determine confidence
        confidence_level = "High" if abs(probability - 0.5) > 0.3 else "Medium" if abs(probability - 0.5) > 0.1 else "Low"
        
        # Check if FHE compatible
        fhe_compatible = (model_name == "Logistic Regression")
        
        return PredictionResponse(
            prediction=prediction,
            probability=probability,
            confidence=confidence_level,
            processing_time=processing_time,
            model_info={
                "model_name": model_name,
                "model_type": type(model).__name__,
                "fhe_compatible": fhe_compatible,
                "fhe_used": False  # Clear prediction, no FHE
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/compare")
async def compare_all_models(request: PredictionRequest):
    """
    Make predictions using all available models for comparison.
    
    Args:
        request: PredictionRequest with 23 features
    
    Returns:
        Predictions from all models with their probabilities
    """
    start_time = time.time()
    
    try:
        # Validate input
        if len(request.features) != 19:
            raise HTTPException(
                status_code=400,
                detail=f"Expected exactly 20 features (demographic features excluded for fairness), got {len(request.features)}"
            )
        
        import joblib
        import numpy as np
        
        scaler = joblib.load("models/scaler.pkl")
        features_array = np.array(request.features).reshape(1, -1)
        features_scaled = scaler.transform(features_array)
        
        # Get all model names from metadata
        metadata_path = "models/all_models_metadata.json"
        import json
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        results = {}
        
        for model_name, model_info in metadata['models'].items():
            model_start = time.time()
            
            # For Logistic Regression, run BOTH plaintext and FHE
            if model_name == "Logistic Regression":
                # First: Plaintext LR
                model_path = f"models/{model_info['filename']}"
                model = joblib.load(model_path)
                
                plain_start = time.time()
                prediction_plain = int(model.predict(features_scaled)[0])
                probability_plain = float(model.predict_proba(features_scaled)[0][1])
                plain_time = time.time() - plain_start
                
                # Add plaintext result
                results["Logistic Regression (Plain)"] = {
                    "prediction": int(prediction_plain),
                    "probability": float(probability_plain),
                    "prediction_label": "DEFAULT" if prediction_plain == 1 else "NO DEFAULT",
                    "fhe_compatible": True,
                    "accuracy": model_info['accuracy'],
                    "f1_score": model_info['f1_score'],
                    "roc_auc": model_info['roc_auc'],
                    "processing_time": plain_time,
                    "used_fhe": False
                }
                
                # Second: FHE encrypted LR
                if fhe_model is not None:
                    fhe_start = time.time()
                    encrypted_features = fhe_model.encrypt_features(request.features)
                    encrypted_result = fhe_model.predict_encrypted(encrypted_features)
                    probability, prediction = fhe_model.decrypt_result(encrypted_result)
                    fhe_time = time.time() - fhe_start
                    
                    # Add FHE result
                    results["Logistic Regression (FHE)"] = {
                        "prediction": int(prediction),
                        "probability": float(probability),
                        "prediction_label": "DEFAULT" if prediction == 1 else "NO DEFAULT",
                        "fhe_compatible": True,
                        "accuracy": model_info['accuracy'],
                        "f1_score": model_info['f1_score'],
                        "roc_auc": model_info['roc_auc'],
                        "processing_time": fhe_time,
                        "used_fhe": True
                    }
            else:
                # Other models: just plaintext
                model_path = f"models/{model_info['filename']}"
                model = joblib.load(model_path)
                
                prediction = int(model.predict(features_scaled)[0])
                
                if hasattr(model, 'predict_proba'):
                    probability = float(model.predict_proba(features_scaled)[0][1])
                else:
                    probability = 0.5
                
                model_time = time.time() - model_start
                
                results[model_name] = {
                    "prediction": int(prediction),
                    "probability": float(probability),
                    "prediction_label": "DEFAULT" if prediction == 1 else "NO DEFAULT",
                    "fhe_compatible": model_info['fhe_compatible'],
                    "accuracy": model_info['accuracy'],
                    "f1_score": model_info['f1_score'],
                    "roc_auc": model_info['roc_auc'],
                    "processing_time": model_time,
                    "used_fhe": False
                }
        
        processing_time = time.time() - start_time
        
        # Check if all models agree
        predictions = [r["prediction"] for r in results.values()]
        all_agree = len(set(predictions)) == 1
        
        return {
            "models": results,
            "all_agree": all_agree,
            "consensus": "DEFAULT" if predictions[0] == 1 else "NO DEFAULT" if all_agree else "MIXED",
            "processing_time": processing_time
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Comparison failed: {str(e)}")


@app.get("/stats")
async def get_stats():
    """Get application statistics."""
    return {
        "model_loaded": fhe_model is not None,
        "fhe_ready": fhe_model.context is not None,
        "uptime": time.time(),
        "endpoints": [
            "/",
            "/health", 
            "/context",
            "/predict",
            "/predict/clear",
            "/predict/model/{model_name}",
            "/predict/compare",
            "/model/info",
            "/models/list",
            "/stats"
        ]
    }


if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Starting FHE Credit Default Risk Prediction API...")
    print("📚 API Documentation: http://localhost:8000/docs")
    print("🔍 Health Check: http://localhost:8000/health")
    print("🔐 Context Endpoint: http://localhost:8000/context")
    
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8000,
        reload=True,  # Enable auto-reload for development
        log_level="info"
    )
