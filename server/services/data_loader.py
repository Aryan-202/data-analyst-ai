import pandas as pd
import aiofiles
import json
from typing import Dict, Any
import io


class DataLoader:
    def __init__(self):
        self.supported_formats = ['.csv', '.xlsx', '.xls', '.json']

    async def load_data(self, file_path: str) -> pd.DataFrame:
        """Load data from various file formats"""
        try:
            if file_path.endswith('.csv'):
                return await self._load_csv(file_path)
            elif file_path.endswith(('.xlsx', '.xls')):
                return await self._load_excel(file_path)
            elif file_path.endswith('.json'):
                return await self._load_json(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path}")
        except Exception as e:
            raise Exception(f"Error loading data from {file_path}: {str(e)}")

    async def _load_csv(self, file_path: str) -> pd.DataFrame:
        """Load CSV file"""
        async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
            content = await f.read()

        # Try different encodings if utf-8 fails
        try:
            return pd.read_csv(io.StringIO(content))
        except UnicodeDecodeError:
            async with aiofiles.open(file_path, 'r', encoding='latin-1') as f:
                content = await f.read()
            return pd.read_csv(io.StringIO(content))

    async def _load_excel(self, file_path: str) -> pd.DataFrame:
        """Load Excel file"""
        return pd.read_excel(file_path)

    async def _load_json(self, file_path: str) -> pd.DataFrame:
        """Load JSON file"""
        async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
            content = await f.read()

        data = json.loads(content)

        # Handle different JSON structures
        if isinstance(data, list):
            return pd.DataFrame(data)
        elif isinstance(data, dict):
            # Assume records are in first key
            first_key = list(data.keys())[0]
            if isinstance(data[first_key], list):
                return pd.DataFrame(data[first_key])
            else:
                return pd.DataFrame([data])
        else:
            raise ValueError("Unsupported JSON structure")

    def get_dataset_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get basic information about the dataset"""
        return {
            "shape": df.shape,
            "columns": list(df.columns),
            "data_types": df.dtypes.astype(str).to_dict(),
            "memory_usage": df.memory_usage(deep=True).sum(),
            "missing_values": df.isnull().sum().to_dict()
        }