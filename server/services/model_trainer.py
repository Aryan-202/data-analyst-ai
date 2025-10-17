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
import traceback

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
        try:
            print(f"Starting prediction for model_id: {model_id}")
            print(f"Input data received: {input_data}")

            model_path = os.path.join(self.models_storage, f"{model_id}.pkl")

            print(f"Looking for model at: {model_path}")
            print(f"Models directory exists: {os.path.exists(self.models_storage)}")
            print(f"Model file exists: {os.path.exists(model_path)}")

            if not os.path.exists(model_path):
                # List available models for debugging
                available_models = []
                if os.path.exists(self.models_storage):
                    available_models = os.listdir(self.models_storage)
                print(f"Available models: {available_models}")
                raise ValueError(f"Model {model_id} not found at {model_path}. Available models: {available_models}")

            # Load model
            print("Loading model...")
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)

            model = model_data['model']
            preprocessing_info = model_data['preprocessing']
            model_info = model_data['model_info']

            print(f"Model loaded successfully: {model_info.get('algorithm', 'Unknown')}")
            print(f"Model info: {model_info}")
            print(f"Preprocessing info: {preprocessing_info.keys()}")

        except Exception as e:
            print(f"Error loading model: {str(e)}")
            print(f"Traceback: {traceback.format_exc()}")
            raise ValueError(f"Failed to load model: {str(e)}")

        try:
            # Convert input data to DataFrame
            print("Converting input data to DataFrame...")
            input_df = pd.DataFrame(input_data)

            print(f"Input DataFrame shape: {input_df.shape}")
            print(f"Input DataFrame columns: {list(input_df.columns)}")
            print(f"Input DataFrame dtypes: {input_df.dtypes.to_dict()}")
            print(f"Input data sample: {input_df.head().to_dict()}")

            # Check if we have the required columns
            expected_features = model_info.get('feature_columns', [])
            print(f"Expected features: {expected_features}")

            missing_columns = set(expected_features) - set(input_df.columns)
            if missing_columns:
                print(f"Warning: Missing columns in input: {missing_columns}")
                # Add missing columns with default values
                for col in missing_columns:
                    input_df[col] = 0  # or np.nan, depending on your needs

            # Ensure column order matches training
            input_df = input_df.reindex(columns=expected_features, fill_value=0)

            # Preprocess input data
            print("Preprocessing input data...")
            processed_input = await self._preprocess_predict_data(input_df, preprocessing_info)

            print(f"Processed input shape: {processed_input.shape}")
            print(f"Processed input type: {type(processed_input)}")

            # Make predictions
            print("Making predictions...")
            predictions = model.predict(processed_input)

            print(f"Raw predictions: {predictions}")
            print(f"Predictions type: {type(predictions)}")
            print(f"Predictions shape: {predictions.shape if hasattr(predictions, 'shape') else 'No shape'}")

            # Convert predictions to native Python types
            if hasattr(predictions, 'tolist'):
                predictions_list = predictions.tolist()
            else:
                predictions_list = list(predictions)

            # Convert numpy types to native Python types
            predictions_list = self._convert_numpy_types(predictions_list)

            print(f"Final predictions: {predictions_list}")

            return predictions_list

        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            print(f"Traceback: {traceback.format_exc()}")
            raise ValueError(f"Prediction failed: {str(e)}")

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
                    'value': float(median_val)  # Convert to Python float
                }

        # Scale numeric features
        scaler = StandardScaler()
        X[numeric_columns] = scaler.fit_transform(X[numeric_columns])
        preprocessing_info['scaler'] = {
            'method': 'standard_scaler',
            'features': numeric_columns.tolist(),
            'mean': {col: float(X[col].mean()) for col in numeric_columns},  # Convert to Python float
            'std': {col: float(X[col].std()) for col in numeric_columns}  # Convert to Python float
        }

        # Encode target variable if classification
        if y.dtype == 'object' or len(y.unique()) < 20:  # Classification
            le_target = LabelEncoder()
            y = le_target.fit_transform(y)
            preprocessing_info['target_encoder'] = {
                'classes': le_target.classes_.tolist(),
                'type': 'label_encoder'
            }

        # Convert all preprocessing info to native Python types
        preprocessing_info = self._convert_numpy_types(preprocessing_info)

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
                'mean_squared_error': float(mse),  # Convert to Python float
                'root_mean_squared_error': float(rmse),  # Convert to Python float
                'r_squared': float(model.score(X_test, y_test)),  # Convert to Python float
                'mean_absolute_error': float(np.mean(np.abs(y_test - y_pred)))  # Convert to Python float
            })

        elif problem_type == 'classification':
            accuracy = accuracy_score(y_test, y_pred)

            performance.update({
                'accuracy': float(accuracy),  # Convert to Python float
                'precision': float(
                    classification_report(y_test, y_pred, output_dict=True)['weighted avg']['precision']),
                # Convert to Python float
                'recall': float(classification_report(y_test, y_pred, output_dict=True)['weighted avg']['recall']),
                # Convert to Python float
                'f1_score': float(classification_report(y_test, y_pred, output_dict=True)['weighted avg']['f1-score'])
                # Convert to Python float
            })

        # Feature importance for tree-based models
        if hasattr(model, 'feature_importances_'):
            performance['feature_importance'] = {
                'features': [f"Feature_{i}" for i in range(len(model.feature_importances_))],
                'importance': [float(imp) for imp in model.feature_importances_]  # Convert to Python float
            }

        # Convert all performance metrics to native Python types
        performance = self._convert_numpy_types(performance)

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
            'missing_values': int(df.isnull().sum().sum()),  # Convert to Python int
            'class_imbalance': None
        }

        # Check for class imbalance in classification
        if df[target_column].dtype == 'object' or df[target_column].nunique() < 20:
            value_counts = df[target_column].value_counts()
            if len(value_counts) > 1:
                imbalance_ratio = float(value_counts.iloc[0] / value_counts.iloc[1])  # Convert to Python float
                analysis['class_imbalance'] = {
                    'ratio': imbalance_ratio,
                    'severity': 'high' if imbalance_ratio > 5 else 'medium' if imbalance_ratio > 2 else 'low'
                }

        # Convert all analysis data to native Python types
        analysis = self._convert_numpy_types(analysis)

        return analysis

    async def _preprocess_predict_data(self, input_df: pd.DataFrame, preprocessing_info: Dict) -> np.ndarray:
        """Preprocess input data for prediction using saved preprocessing info"""
        X = input_df.copy()

        print("Starting preprocessing...")
        print(f"Input shape before preprocessing: {X.shape}")

        # Apply label encoding
        if 'label_encoders' in preprocessing_info:
            print("Applying label encoding...")
            for col, encoder_info in preprocessing_info['label_encoders'].items():
                if col in X.columns:
                    print(f"Encoding column: {col}")
                    le = LabelEncoder()
                    le.classes_ = np.array(encoder_info['classes'])

                    # Handle unseen categories
                    X[col] = X[col].astype(str)
                    unseen_mask = ~X[col].isin(encoder_info['classes'])
                    if unseen_mask.any():
                        print(f"Found {unseen_mask.sum()} unseen categories in {col}")
                        # Replace unseen categories with most frequent
                        X.loc[unseen_mask, col] = encoder_info['classes'][0]

                    X[col] = le.transform(X[col])

        # Handle missing values
        numeric_columns = X.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if f'missing_value_{col}' in preprocessing_info:
                fill_value = preprocessing_info[f'missing_value_{col}']['value']
                missing_count = X[col].isnull().sum()
                if missing_count > 0:
                    print(f"Filling {missing_count} missing values in {col} with {fill_value}")
                    X[col].fillna(fill_value, inplace=True)

        # Apply scaling
        if 'scaler' in preprocessing_info:
            print("Applying scaling...")
            scaler_info = preprocessing_info['scaler']
            scaling_features = [f for f in scaler_info['features'] if f in X.columns]

            if scaling_features:
                # Use saved mean and std for scaling
                for col in scaling_features:
                    if col in scaler_info.get('mean', {}) and col in scaler_info.get('std', {}):
                        mean_val = scaler_info['mean'][col]
                        std_val = scaler_info['std'][col]
                        if std_val != 0:  # Avoid division by zero
                            X[col] = (X[col] - mean_val) / std_val
                        else:
                            X[col] = 0  # If std is 0, set to 0 (all values were same during training)

        print(f"Output shape after preprocessing: {X.shape}")
        return X.values

    def _convert_numpy_types(self, obj: Any) -> Any:
        """Recursively convert numpy types to native Python types for JSON serialization"""
        if isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif pd.isna(obj):
            return None
        else:
            return obj