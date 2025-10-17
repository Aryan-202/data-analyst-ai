import pandas as pd
import os
import json
from typing import Dict, Any, Optional, Tuple
from server.config import settings
from server.models.dataset_schema import FileType
from server.utils.logger import setup_logger

logger = setup_logger()


class DataLoader:
    """Handles dataset loading and basic validation"""

    def __init__(self):
        self.supported_formats = {
            '.csv': self._load_csv,
            '.xlsx': self._load_excel,
            '.xls': self._load_excel,
            '.json': self._load_json
        }
        self.datasets_metadata = {}

    async def load_dataset(self, file_path: str, file_extension: str) -> Tuple[pd.DataFrame, FileType]:
        """Load dataset based on file type"""
        if file_extension not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {file_extension}")

        try:
            loader_function = self.supported_formats[file_extension]
            df = await loader_function(file_path)

            # Determine file type
            if file_extension == '.csv':
                file_type = FileType.CSV
            elif file_extension in ['.xlsx', '.xls']:
                file_type = FileType.EXCEL
            else:
                file_type = FileType.JSON

            logger.info(f"Dataset loaded: {file_path}, Shape: {df.shape}")
            return df, file_type

        except Exception as e:
            logger.error(f"Error loading dataset {file_path}: {str(e)}")
            raise

    async def _load_csv(self, file_path: str) -> pd.DataFrame:
        """Load CSV file with smart parsing"""
        try:
            # Try different encodings
            encodings = ['utf-8', 'latin-1', 'windows-1252', 'iso-8859-1']

            for encoding in encodings:
                try:
                    df = pd.read_csv(file_path, encoding=encoding)
                    if not df.empty:
                        return df
                except UnicodeDecodeError:
                    continue

            # If all encodings fail, try without specified encoding
            return pd.read_csv(file_path)

        except Exception as e:
            logger.error(f"CSV loading failed: {str(e)}")
            raise

    async def _load_excel(self, file_path: str) -> pd.DataFrame:
        """Load Excel file"""
        try:
            # Read first sheet by default
            df = pd.read_excel(file_path, sheet_name=0)
            return df
        except Exception as e:
            logger.error(f"Excel loading failed: {str(e)}")
            raise

    async def _load_json(self, file_path: str) -> pd.DataFrame:
        """Load JSON file"""
        try:
            df = pd.read_json(file_path)
            return df
        except Exception as e:
            logger.error(f"JSON loading failed: {str(e)}")
            raise

    def get_columns_info(self, df: pd.DataFrame) -> list[Dict[str, Any]]:
        """Get detailed information about DataFrame columns"""
        columns_info = []

        for column in df.columns:
            col_info = {
                'name': column,
                'dtype': str(df[column].dtype),
                'non_null_count': df[column].count(),
                'null_count': df[column].isnull().sum(),
                'null_percentage': round((df[column].isnull().sum() / len(df)) * 100, 2),
                'unique_count': df[column].nunique(),
                'sample_data': df[column].head(5).tolist()
            }

            # Add type-specific info
            if pd.api.types.is_numeric_dtype(df[column]):
                col_info.update({
                    'type': 'numeric',
                    'min': float(df[column].min()) if not df[column].isnull().all() else None,
                    'max': float(df[column].max()) if not df[column].isnull().all() else None,
                    'mean': float(df[column].mean()) if not df[column].isnull().all() else None
                })
            elif pd.api.types.is_datetime64_any_dtype(df[column]):
                col_info.update({
                    'type': 'datetime',
                    'min': df[column].min().isoformat() if not df[column].isnull().all() else None,
                    'max': df[column].max().isoformat() if not df[column].isnull().all() else None
                })
            else:
                col_info.update({
                    'type': 'categorical',
                    'top_values': df[column].value_counts().head(5).to_dict()
                })

            columns_info.append(col_info)

        return columns_info

    def load_dataset_by_id(self, dataset_id: str) -> Optional[pd.DataFrame]:
        """Load dataset by ID from stored files"""
        try:
            file_manager = FileManager()
            file_path = file_manager.get_file_path(dataset_id, "raw")

            if not file_path or not os.path.exists(file_path):
                return None

            file_extension = os.path.splitext(file_path)[1].lower()
            df, _ = self.load_dataset(file_path, file_extension)
            return df

        except Exception as e:
            logger.error(f"Error loading dataset {dataset_id}: {str(e)}")
            return None

    def store_dataset_metadata(self, dataset_id: str, filename: str, file_type: FileType, df: pd.DataFrame):
        """Store dataset metadata"""
        self.datasets_metadata[dataset_id] = {
            'filename': filename,
            'file_type': file_type.value,
            'rows': len(df),
            'columns': len(df.columns),
            'columns_info': self.get_columns_info(df),
            'loaded_at': pd.Timestamp.now().isoformat()
        }

    def list_available_datasets(self) -> list[Dict[str, Any]]:
        """List all available datasets"""
        datasets = []
        for dataset_id, metadata in self.datasets_metadata.items():
            datasets.append({
                'dataset_id': dataset_id,
                **metadata
            })
        return datasets

    def get_dataset_info(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific dataset"""
        return self.datasets_metadata.get(dataset_id)