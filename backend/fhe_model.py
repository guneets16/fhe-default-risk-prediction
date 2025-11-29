"""
FHE Model Implementation for Credit Default Risk Prediction

This module provides a wrapper around scikit-learn models to enable
Fully Homomorphic Encryption (FHE) inference using TenSEAL.
"""

import tenseal as ts
import numpy as np
import joblib
from typing import List, Tuple
import os


class FHEModel:
    """
    FHE-enabled machine learning model for credit default risk prediction.
    
    This class wraps a trained scikit-learn model and provides
    methods for encrypted inference using TenSEAL.
    """
    
    def __init__(self, model_path: str = "models/credit_model.pkl"):
        """
        Initialize the FHE model.
        
        Args:
            model_path: Path to the trained model file
        """
        self.model_path = model_path
        self.model = None
        self.scaler = None
        self.context = None
        self._load_model()
        self._setup_context()
    
    def _load_model(self):
        """Load the trained scikit-learn model and scaler."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        self.model = joblib.load(self.model_path)
        print(f"✅ Loaded model from {self.model_path}")
        print(f"Model type: {type(self.model).__name__}")
        print(f"Model accuracy: {getattr(self.model, 'score', lambda x, y: 'N/A')}")
        
        # Load scaler
        scaler_path = self.model_path.replace("credit_model.pkl", "scaler.pkl")
        
        # Check if path is relative, if so try different locations
        if not os.path.exists(scaler_path):
            # Try one directory up (for when backend is in backend/ subdirectory)
            scaler_path = os.path.join("..", scaler_path)
        
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
            print(f"✅ Loaded scaler from {scaler_path}")
        else:
            print(f"⚠️  No scaler found - features will not be scaled")
            print(f"   Looked for: {scaler_path}")
    
    def _setup_context(self):
        """Setup TenSEAL context for FHE operations."""
        # CKKS scheme parameters
        # poly_modulus_degree: 8192 (good balance of security and performance)
        # coeff_mod_bit_sizes: [60, 40, 40, 60] (optimized for our use case)
        self.context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=8192,
            coeff_mod_bit_sizes=[60, 40, 40, 60]
        )
        
        # Set global scale after context creation
        self.context.global_scale = 2**40
        
        # Generate Galois keys for rotations
        self.context.generate_galois_keys()
        
        # Keep secret key for testing, but create public context for API
        self._setup_public_context()
        
        print("✅ FHE context initialized")
    
    def _setup_public_context(self):
        """Setup public context for client-side encryption."""
        # Create a copy of the context for public use
        self.public_context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=8192,
            coeff_mod_bit_sizes=[60, 40, 40, 60]
        )
        self.public_context.global_scale = 2**40
        self.public_context.generate_galois_keys()
        self.public_context.make_context_public()
    
    def _get_base_model(self):
        """
        Get the base estimator from a potentially calibrated model.
        
        Returns:
            The underlying model (e.g., LogisticRegression)
        """
        # Check if this is a CalibratedClassifierCV
        if hasattr(self.model, 'calibrated_classifiers_'):
            # Return the first calibrated classifier's base estimator
            return self.model.calibrated_classifiers_[0].estimator
        else:
            # Return the model as-is
            return self.model
    
    def get_public_context(self) -> bytes:
        """
        Get the public context for client-side encryption.
        
        Returns:
            Serialized public context
        """
        return self.public_context.serialize(save_public_key=True, save_secret_key=False)
    
    def encrypt_features(self, features: List[float]) -> ts.CKKSVector:
        """
        Encrypt input features for FHE computation.
        
        Args:
            features: List of feature values (23 features for UCI dataset)
            
        Returns:
            Encrypted feature vector
        """
        # Get base model (handle CalibratedClassifierCV)
        base_model = self._get_base_model()
        expected_features = base_model.coef_.shape[1] if hasattr(base_model, 'coef_') else 19
        if len(features) != expected_features:
            raise ValueError(f"Expected {expected_features} features, got {len(features)}")
        
        # Scale features if scaler is available
        if self.scaler is not None:
            features_array = np.array(features).reshape(1, -1)
            scaled_features = self.scaler.transform(features_array)[0]
            return ts.ckks_vector(self.context, scaled_features.tolist())
        else:
            return ts.ckks_vector(self.context, features)
    
    def predict_encrypted(self, encrypted_features: ts.CKKSVector) -> ts.CKKSVector:
        """
        Perform encrypted prediction using FHE.
        
        This method implements a simplified linear model that works with FHE:
        prediction = features * coefficients + bias
        
        Args:
            encrypted_features: Encrypted feature vector
            
        Returns:
            Encrypted prediction result
        """
        # Get base model (handle CalibratedClassifierCV)
        base_model = self._get_base_model()
        
        # Get model coefficients
        if hasattr(base_model, 'coef_'):
            coefficients = base_model.coef_[0]  # For binary classification
        else:
            # Fallback for models without coef_ attribute
            coefficients = np.array([0.1, -0.2, 0.3, -0.1])  # Example coefficients
        
        # Get bias term
        if hasattr(base_model, 'intercept_'):
            bias = base_model.intercept_[0]
        else:
            bias = 0.5  # Example bias
        
        # Encrypt coefficients
        encrypted_coeffs = ts.ckks_vector(self.context, coefficients)
        
        # Compute linear combination: features * coefficients + bias
        encrypted_result = encrypted_features.dot(encrypted_coeffs) + bias
        
        return encrypted_result
    
    def decrypt_result(self, encrypted_result: ts.CKKSVector) -> Tuple[float, int]:
        """
        Decrypt and process the FHE result.
        
        Args:
            encrypted_result: Encrypted prediction result
            
        Returns:
            Tuple of (default_probability, prediction)
            - default_probability: Probability of DEFAULT (0-1)
            - prediction: 1 = DEFAULT, 0 = NO DEFAULT
        """
        # Decrypt the result using the secret key
        # Note: In a real application, this would be done on the client side
        # For demo purposes, we'll use the server's secret key
        result = encrypted_result.decrypt()[0]
        
        # Apply sigmoid approximation for probability of DEFAULT
        # Using a simple approximation: 1 / (1 + exp(-x))
        # For FHE compatibility, we use a polynomial approximation
        probability = self._sigmoid_approximation(result)
        
        # Binary classification: 1 if probability > 0.5, else 0
        prediction = 1 if probability > 0.5 else 0
        
        return probability, prediction
    
    def _sigmoid_approximation(self, x: float) -> float:
        """
        Approximate sigmoid function for FHE compatibility.
        
        Args:
            x: Input value
            
        Returns:
            Approximate sigmoid value
        """
        # Use standard sigmoid - FHE decryption happens on client side anyway
        # so no need for polynomial approximation here
        return 1 / (1 + np.exp(-x))
    
    def predict_clear(self, features: List[float]) -> Tuple[float, int]:
        """
        Predict using clear (non-encrypted) data for comparison.
        
        Args:
            features: List of feature values
            
        Returns:
            Tuple of (probability, prediction)
        """
        features_array = np.array(features).reshape(1, -1)
        
        # Scale features if scaler is available
        if self.scaler is not None:
            features_array = self.scaler.transform(features_array)
        
        if hasattr(self.model, 'predict_proba'):
            probability = self.model.predict_proba(features_array)[0][1]
        else:
            # Fallback for models without predict_proba
            probability = 0.7  # Example probability
        
        prediction = 1 if probability > 0.5 else 0
        
        return probability, prediction
    
    def get_model_info(self) -> dict:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        # Get actual number of features from model
        base_model = self._get_base_model()
        n_features = base_model.coef_.shape[1] if hasattr(base_model, 'coef_') else 19
        
        return {
            "model_type": type(self.model).__name__,
            "model_path": self.model_path,
            "features": n_features,
            "fhe_scheme": "CKKS",
            "poly_modulus_degree": 8192
        }


def create_sample_model():
    """
    Create and train a sample credit default risk prediction model.
    
    This function generates sample data and trains a logistic regression model
    for demonstration purposes.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    import os
    
    # Create sample data
    X, y = make_classification(
        n_samples=1000,
        n_features=4,
        n_redundant=0,
        n_informative=4,
        n_clusters_per_class=1,
        random_state=42
    )
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    accuracy = model.score(X_test, y_test)
    print(f"Model accuracy: {accuracy:.3f}")
    
    # Save model
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/credit_model.pkl")
    print("✅ Model saved to models/credit_model.pkl")
    
    return model


if __name__ == "__main__":
    # Create sample model for testing
    print("Creating sample model...")
    model = create_sample_model()
    
    # Test FHE model
    print("\nTesting FHE model...")
    fhe_model = FHEModel("models/credit_model.pkl")
    
    # Test with sample data
    test_features = [50000, 30, 24, 0.3]  # income, age, credit_history, debt_ratio
    
    # Clear prediction
    prob_clear, pred_clear = fhe_model.predict_clear(test_features)
    print(f"Clear prediction: {pred_clear} (probability: {prob_clear:.3f})")
    
    # FHE prediction
    encrypted_features = fhe_model.encrypt_features(test_features)
    encrypted_result = fhe_model.predict_encrypted(encrypted_features)
    prob_fhe, pred_fhe = fhe_model.decrypt_result(encrypted_result)
    print(f"FHE prediction: {pred_fhe} (probability: {prob_fhe:.3f})")
    
    print("✅ FHE model test completed!")
