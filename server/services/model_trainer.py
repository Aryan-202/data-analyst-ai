import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import uuid
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, mean_squared_error, silhouette_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings

warnings.filterwarnings('ignore')

from server.utils.logger import setup_logger

logger = setup_logger()


class ModelTrainer:
    """Handles automated machine learning model training"""

    def __init__(self):
        self.trained_models = {}
        self.available_models = {
            'classification': {
                'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
                # Add more classifiers as needed
            },
            'regression': {
                'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
                # Add more regressors as needed
            },
            'clustering': {
                'kmeans': KMeans(n_clusters=3, random_state=42),
                # Add more clustering algorithms as needed
            }
        }

    async def train_model(self, df: pd.DataFrame, target_column: str, problem_type: str,
                          model_type: str = "auto", test_size: float = 0.2) -> Dict[str, Any]:
        """Train a machine learning model"""

        try:
            # Prepare data
            X, y, feature_names = self._prepare_features(df, target_column, problem_type)

            # Handle missing values
            X = self._handle_missing_values(X)

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )

            # Select model
            model = self._select_model(problem_type, model_type)

            # Train model
            model.fit(X_train, y_train)

            # Evaluate model
            performance = self._evaluate_model(model, X_test, y_test, problem_type)

            # Get feature importance
            feature_importance = self._get_feature_importance(model, feature_names, problem_type)

            # Generate predictions sample
            predictions_sample = self._get_predictions_sample(model, X_test, y_test, problem_type)

            # Store model
            model_id = str(uuid.uuid4())
            self.trained_models[model_id] = {
                'model': model,
                'problem_type': problem_type,
                'target_column': target_column,
                'feature_names': feature_names,
                'performance': performance
            }

            return {
                'model_id': model_id,
                'performance': performance,
                'feature_importance': feature_importance,
                'predictions_sample': predictions_sample,
                'model_type': type(model).__name__
            }

        except Exception as e:
            logger.error(f"Model training failed: {str(e)}")
            raise

    def _prepare_features(self, df: pd.DataFrame, target_column: str, problem_type: str):
        """Prepare features and target variable"""
        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]

        # Handle categorical features
        X = self._encode_categorical_features(X)

        # Handle target variable based on problem type
        if problem_type == 'classification' and y.dtype == 'object':
            le = LabelEncoder()
            y = le.fit_transform(y)

        feature_names = X.columns.tolist()

        return X, y, feature_names

    def _encode_categorical_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features"""
        X_encoded = X.copy()

        for column in X_encoded.select_dtypes(include=['object']).columns:
            # For simplicity, use label encoding
            # In production, consider one-hot encoding for low cardinality
            le = LabelEncoder()
            X_encoded[column] = le.fit_transform(X_encoded[column].astype(str))

        return X_encoded

    def _handle_missing_values(self, X: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in features"""
        X_clean = X.copy()

        for column in X_clean.columns:
            if X_clean[column].isnull().any():
                if pd.api.types.is_numeric_dtype(X_clean[column]):
                    X_clean[column] = X_clean[column].fillna(X_clean[column].median())
                else:
                    X_clean[column] = X_clean[column].fillna(X_clean[column].mode()[0])

        return X_clean

    def _select_model(self, problem_type: str, model_type: str):
        """Select appropriate model"""
        if model_type == 'auto':
            # Auto-select first available model for the problem type
            available = self.available_models.get(problem_type, {})
            if available:
                model_name = list(available.keys())[0]
                return available[model_name]
            else:
                raise ValueError(f"No models available for problem type: {problem_type}")
        else:
            # Select specific model
            model = self.available_models.get(problem_type, {}).get(model_type)
            if not model:
                raise ValueError(f"Model {model_type} not available for {problem_type}")
            return model

    def _evaluate_model(self, model, X_test, y_test, problem_type: str) -> Dict[str, Any]:
        """Evaluate model performance"""
        y_pred = model.predict(X_test)

        metrics = {}

        if problem_type == 'classification':
            metrics['accuracy'] = round(accuracy_score(y_test, y_pred), 4)
            # Add more classification metrics as needed

        elif problem_type == 'regression':
            metrics['mse'] = round(mean_squared_error(y_test, y_pred), 4)
            metrics['rmse'] = round(np.sqrt(metrics['mse']), 4)

        elif problem_type == 'clustering':
            if len(np.unique(y_pred)) > 1:  # Need at least 2 clusters for silhouette score
                metrics['silhouette_score'] = round(silhouette_score(X_test, y_pred), 4)
            metrics['n_clusters'] = len(np.unique(y_pred))

        return metrics

    def _get_feature_importance(self, model, feature_names: List[str], problem_type: str) -> Optional[Dict[str, float]]:
        """Get feature importance if available"""
        if hasattr(model, 'feature_importances_'):
            importance_dict = dict(zip(feature_names, model.feature_importances_))
            # Sort by importance
            return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:10])  # Top 10
        return None

    def _get_predictions_sample(self, model, X_test, y_test, problem_type: str) -> List[Any]:
        """Get sample predictions for demonstration"""
        try:
            y_pred = model.predict(X_test)

            # Return first 5 predictions
            sample_size = min(5, len(y_pred))

            if problem_type == 'classification':
                return [
                    {'actual': int(y_test.iloc[i]), 'predicted': int(y_pred[i])}
                    for i in range(sample_size)
                ]
            elif problem_type == 'regression':
                return [
                    {'actual': float(y_test.iloc[i]), 'predicted': float(y_pred[i])}
                    for i in range(sample_size)
                ]
            else:
                return [int(pred) for pred in y_pred[:sample_size]]

        except Exception as e:
            logger.warning(f"Could not generate predictions sample: {str(e)}")
            return []

    def get_classification_models(self) -> List[str]:
        """Get available classification models"""
        return list(self.available_models['classification'].keys())

    def get_regression_models(self) -> List[str]:
        """Get available regression models"""
        return list(self.available_models['regression'].keys())

    def get_clustering_models(self) -> List[str]:
        """Get available clustering models"""
        return list(self.available_models['clustering'].keys())

    def get_auto_ml_info(self) -> Dict[str, Any]:
        """Get AutoML capabilities information"""
        return {
            'supported_problem_types': ['classification', 'regression', 'clustering'],
            'auto_feature_engineering': True,
            'auto_hyperparameter_tuning': False,  # Basic version
            'cross_validation': False,
            'ensemble_methods': False
        }

    def store_model(self, dataset_id: str, model_results: Dict[str, Any]):
        """Store model results"""
        # In production, you might want to save models to disk/database
        pass

    def predict(self, model_id: str, data: pd.DataFrame) -> np.ndarray:
        """Make predictions using a trained model"""
        if model_id not in self.trained_models:
            raise ValueError(f"Model {model_id} not found")

        model_info = self.trained_models[model_id]
        model = model_info['model']

        # Preprocess data same as training
        X = self._encode_categorical_features(data)
        X = self._handle_missing_values(X)

        return model.predict(X)