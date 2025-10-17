import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle
import os
import uuid
from typing import Dict, Any, List, Optional
import warnings

warnings.filterwarnings('ignore')


class ModelTrainer:
    def __init__(self, settings):
        self.settings = settings
        self.models_storage = "data/models"
        os.makedirs(self.models_storage, exist_ok=True)

        self.algorithm_map = {
            'regression': {
                'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
                'linear_regression': LinearRegression()
            },
            'classification': {
                'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
                'logistic_regression': LogisticRegression(random_state=42)
            }
        }

    async def train_model(self, file_path: str, target_column: str, problem_type: str,
                          model_type: str = "auto", test_size: float = 0.2,
                          time_column: Optional[str] = None) -> Dict[str, Any]:
        """Train machine learning model"""
        df = pd.read_csv(file_path)

        # Validate inputs
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset")

        # Prepare data
        X, y, feature_names, preprocessing_info = await self._prepare_features(df, target_column)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        # Select model
        model = await self._select_model(problem_type, model_type)

        # Train model
        model.fit(X_train, y_train)

        # Evaluate model
        performance = await self._evaluate_model(model, X_test, y_test, problem_type)

        # Generate model info
        model_id = str(uuid.uuid4())
        model_info = {
            'model_id': model_id,
            'problem_type': problem_type,
            'algorithm': type(model).__name__,
            'target_column': target_column,
            'feature_columns': feature_names,
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'preprocessing': preprocessing_info
        }

        # Save model
        model_path = os.path.join(self.models_storage, f"{model_id}.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump({
                'model': model,
                'model_info': model_info,
                'preprocessing': preprocessing_info
            }, f)

        return {
            'model_id': model_id,
            'model_info': model_info,
            'performance': performance
        }

    async def predict(self, model_id: str, input_data: List[Dict[str, Any]]) -> List[Any]:
        """Make predictions using trained model"""
        model_path = os.path.join(self.models_storage, f"{model_id}.pkl")

        if not os.path.exists(model_path):
            raise ValueError(f"Model {model_id} not found")

        # Load model
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)

        model = model_data['model']
        preprocessing_info = model_data['preprocessing']

        # Convert input data to DataFrame
        input_df = pd.DataFrame(input_data)

        # Preprocess input data
        processed_input = await self._preprocess_predict_data(input_df, preprocessing_info)

        # Make predictions
        predictions = model.predict(processed_input)

        return predictions.tolist()

    async def get_model_suggestions(self, file_path: str, target_column: str) -> List[Dict[str, Any]]:
        """Get suggested models for the dataset"""
        df = pd.read_csv(file_path)

        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found")

        # Determine problem type
        problem_type = await self._determine_problem_type(df[target_column])

        suggestions = []

        if problem_type == 'regression':
            suggestions = [
                {
                    'algorithm': 'Random Forest',
                    'type': 'regression',
                    'description': 'Good for complex relationships, handles non-linearity well',
                    'suitability': 'high',
                    'training_time': 'medium'
                },
                {
                    'algorithm': 'Linear Regression',
                    'type': 'regression',
                    'description': 'Fast and interpretable, good for linear relationships',
                    'suitability': 'medium',
                    'training_time': 'low'
                }
            ]
        elif problem_type == 'classification':
            suggestions = [
                {
                    'algorithm': 'Random Forest',
                    'type': 'classification',
                    'description': 'Robust, handles non-linearity, good for complex patterns',
                    'suitability': 'high',
                    'training_time': 'medium'
                },
                {
                    'algorithm': 'Logistic Regression',
                    'type': 'classification',
                    'description': 'Fast, interpretable, good for binary classification',
                    'suitability': 'medium',
                    'training_time': 'low'
                }
            ]

        # Add dataset-specific recommendations
        dataset_info = await self._analyze_dataset_for_modeling(df, target_column)

        for suggestion in suggestions:
            suggestion.update({
                'dataset_compatibility': dataset_info,
                'recommended': suggestion['suitability'] == 'high'
            })

        return suggestions

    async def _prepare_features(self, df: pd.DataFrame, target_column: str) -> tuple:
        """Prepare features for modeling"""
        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]

        preprocessing_info = {}
        feature_names = list(X.columns)

        # Handle categorical variables
        categorical_columns = X.select_dtypes(include=['object']).columns
        label_encoders = {}

        for col in categorical_columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = {
                'classes': le.classes_.tolist(),
                'type': 'label_encoder'
            }

        preprocessing_info['label_encoders'] = label_encoders

        # Handle missing values
        numeric_columns = X.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if X[col].isnull().any():
                median_val = X[col].median()
                X[col].fillna(median_val, inplace=True)
                preprocessing_info[f'missing_value_{col}'] = {
                    'method': 'median',
                    'value': median_val
                }

        # Scale numeric features
        scaler = StandardScaler()
        X[numeric_columns] = scaler.fit_transform(X[numeric_columns])
        preprocessing_info['scaler'] = {
            'method': 'standard_scaler',
            'features': numeric_columns.tolist()
        }

        # Encode target variable if classification
        if y.dtype == 'object' or len(y.unique()) < 20:  # Classification
            le_target = LabelEncoder()
            y = le_target.fit_transform(y)
            preprocessing_info['target_encoder'] = {
                'classes': le_target.classes_.tolist(),
                'type': 'label_encoder'
            }

        return X.values, y, feature_names, preprocessing_info

    async def _select_model(self, problem_type: str, model_type: str):
        """Select appropriate model based on problem type"""
        if model_type == "auto":
            # Default to Random Forest for auto selection
            if problem_type == 'regression':
                return self.algorithm_map['regression']['random_forest']
            elif problem_type == 'classification':
                return self.algorithm_map['classification']['random_forest']
            else:
                raise ValueError(f"Unsupported problem type: {problem_type}")
        else:
            # Specific model selection
            if problem_type in self.algorithm_map and model_type in self.algorithm_map[problem_type]:
                return self.algorithm_map[problem_type][model_type]
            else:
                raise ValueError(f"Unsupported model type: {model_type} for problem type: {problem_type}")

    async def _evaluate_model(self, model, X_test, y_test, problem_type: str) -> Dict[str, Any]:
        """Evaluate model performance"""
        y_pred = model.predict(X_test)

        performance = {}

        if problem_type == 'regression':
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)

            performance.update({
                'mean_squared_error': mse,
                'root_mean_squared_error': rmse,
                'r_squared': model.score(X_test, y_test),
                'mean_absolute_error': np.mean(np.abs(y_test - y_pred))
            })

        elif problem_type == 'classification':
            accuracy = accuracy_score(y_test, y_pred)

            performance.update({
                'accuracy': accuracy,
                'precision': classification_report(y_test, y_pred, output_dict=True)['weighted avg']['precision'],
                'recall': classification_report(y_test, y_pred, output_dict=True)['weighted avg']['recall'],
                'f1_score': classification_report(y_test, y_pred, output_dict=True)['weighted avg']['f1-score']
            })

        # Feature importance for tree-based models
        if hasattr(model, 'feature_importances_'):
            performance['feature_importance'] = {
                'features': [f"Feature_{i}" for i in range(len(model.feature_importances_))],
                'importance': model.feature_importances_.tolist()
            }

        return performance

    async def _determine_problem_type(self, target_series: pd.Series) -> str:
        """Determine if problem is regression or classification"""
        if target_series.dtype in [np.number]:
            # Check if it's actually classification with numeric labels
            unique_values = target_series.nunique()
            if unique_values <= 10:  # Likely classification
                return 'classification'
            else:
                return 'regression'
        else:
            return 'classification'

    async def _analyze_dataset_for_modeling(self, df: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        """Analyze dataset characteristics for modeling recommendations"""
        analysis = {
            'total_samples': len(df),
            'total_features': len(df.columns) - 1,  # excluding target
            'numeric_features': len(df.select_dtypes(include=[np.number]).columns) - (
                1 if df[target_column].dtype in [np.number] else 0),
            'categorical_features': len(df.select_dtypes(include=['object']).columns),
            'missing_values': df.isnull().sum().sum(),
            'class_imbalance': None
        }

        # Check for class imbalance in classification
        if df[target_column].dtype == 'object' or df[target_column].nunique() < 20:
            value_counts = df[target_column].value_counts()
            if len(value_counts) > 1:
                imbalance_ratio = value_counts.iloc[0] / value_counts.iloc[1]
                analysis['class_imbalance'] = {
                    'ratio': imbalance_ratio,
                    'severity': 'high' if imbalance_ratio > 5 else 'medium' if imbalance_ratio > 2 else 'low'
                }

        return analysis

    async def _preprocess_predict_data(self, input_df: pd.DataFrame, preprocessing_info: Dict) -> np.ndarray:
        """Preprocess input data for prediction using saved preprocessing info"""
        X = input_df.copy()

        # Apply label encoding
        if 'label_encoders' in preprocessing_info:
            for col, encoder_info in preprocessing_info['label_encoders'].items():
                if col in X.columns:
                    le = LabelEncoder()
                    le.classes_ = np.array(encoder_info['classes'])

                    # Handle unseen categories
                    X[col] = X[col].astype(str)
                    unseen_mask = ~X[col].isin(encoder_info['classes'])
                    if unseen_mask.any():
                        # Replace unseen categories with most frequent
                        X.loc[unseen_mask, col] = encoder_info['classes'][0]

                    X[col] = le.transform(X[col])

        # Handle missing values
        numeric_columns = X.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if f'missing_value_{col}' in preprocessing_info:
                fill_value = preprocessing_info[f'missing_value_{col}']['value']
                X[col].fillna(fill_value, inplace=True)

        # Apply scaling
        if 'scaler' in preprocessing_info:
            scaler_info = preprocessing_info['scaler']
            X[scaler_info['features']] = (X[scaler_info['features']] - np.mean(X[scaler_info['features']],
                                                                               axis=0)) / np.std(
                X[scaler_info['features']], axis=0)

        return X.values