import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
from server.utils.logger import setup_logger

logger = setup_logger()


class DataCleaner:
    """Handles data cleaning and preprocessing operations"""

    def __init__(self):
        self.available_operations = {
            'handle_missing_values': self._handle_missing_values,
            'remove_duplicates': self._remove_duplicates,
            'fix_data_types': self._fix_data_types,
            'remove_outliers': self._remove_outliers,
            'standardize_text': self._standardize_text,
            'normalize_data': self._normalize_data,
            'encode_categorical': self._encode_categorical
        }

    async def clean_dataframe(self, df: pd.DataFrame, operations: List[str], options: Dict[str, Any]) -> Tuple[
        pd.DataFrame, List[str], Dict[str, Any]]:
        """Perform cleaning operations on DataFrame"""
        cleaning_report = {
            'operations_performed': [],
            'changes_made': {},
            'rows_before': len(df),
            'rows_after': len(df)
        }

        cleaned_df = df.copy()
        performed_operations = []

        for operation in operations:
            if operation in self.available_operations:
                try:
                    cleaned_df = await self.available_operations[operation](
                        cleaned_df,
                        options.get(operation, {})
                    )
                    performed_operations.append(operation)
                    cleaning_report['operations_performed'].append(operation)
                except Exception as e:
                    logger.warning(f"Operation {operation} failed: {str(e)}")
                    cleaning_report['changes_made'][operation] = f"Failed: {str(e)}"
            else:
                logger.warning(f"Unknown cleaning operation: {operation}")

        cleaning_report['rows_after'] = len(cleaned_df)
        cleaning_report['rows_removed'] = cleaning_report['rows_before'] - cleaning_report['rows_after']

        return cleaned_df, performed_operations, cleaning_report

    async def _handle_missing_values(self, df: pd.DataFrame, options: Dict[str, Any]) -> pd.DataFrame:
        """Handle missing values in DataFrame"""
        strategy = options.get('strategy', 'auto')
        threshold = options.get('threshold', 0.5)

        # Calculate missing percentages
        missing_percentages = df.isnull().sum() / len(df)

        if strategy == 'auto':
            # Auto strategy: drop columns with >50% missing, fill others
            columns_to_drop = missing_percentages[missing_percentages > threshold].index
            df_cleaned = df.drop(columns=columns_to_drop)

            # Fill remaining missing values
            for column in df_cleaned.columns:
                if df_cleaned[column].isnull().any():
                    if pd.api.types.is_numeric_dtype(df_cleaned[column]):
                        df_cleaned[column] = df_cleaned[column].fillna(df_cleaned[column].median())
                    else:
                        df_cleaned[column] = df_cleaned[column].fillna(
                            df_cleaned[column].mode()[0] if not df_cleaned[column].mode().empty else 'Unknown')

        elif strategy == 'drop':
            df_cleaned = df.dropna()
        elif strategy == 'fill':
            fill_value = options.get('fill_value')
            if fill_value:
                df_cleaned = df.fillna(fill_value)
            else:
                df_cleaned = df.fillna(method='ffill').fillna(method='bfill')
        else:
            df_cleaned = df.copy()

        return df_cleaned

    async def _remove_duplicates(self, df: pd.DataFrame, options: Dict[str, Any]) -> pd.DataFrame:
        """Remove duplicate rows"""
        subset = options.get('subset')
        keep = options.get('keep', 'first')

        if subset:
            return df.drop_duplicates(subset=subset, keep=keep)
        else:
            return df.drop_duplicates(keep=keep)

    async def _fix_data_types(self, df: pd.DataFrame, options: Dict[str, Any]) -> pd.DataFrame:
        """Fix data types automatically"""
        df_cleaned = df.copy()

        for column in df_cleaned.columns:
            # Try to convert to numeric
            if not pd.api.types.is_numeric_dtype(df_cleaned[column]):
                df_cleaned[column] = pd.to_numeric(df_cleaned[column], errors='ignore')

            # Try to convert to datetime
            if not pd.api.types.is_datetime64_any_dtype(df_cleaned[column]):
                df_cleaned[column] = pd.to_datetime(df_cleaned[column], errors='ignore')

        return df_cleaned

    async def _remove_outliers(self, df: pd.DataFrame, options: Dict[str, Any]) -> pd.DataFrame:
        """Remove outliers using IQR method"""
        method = options.get('method', 'iqr')
        columns = options.get('columns', [])

        if not columns:
            # Auto-detect numeric columns
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        df_cleaned = df.copy()

        for column in columns:
            if column in df_cleaned.columns and pd.api.types.is_numeric_dtype(df_cleaned[column]):
                Q1 = df_cleaned[column].quantile(0.25)
                Q3 = df_cleaned[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                # Keep only non-outliers
                df_cleaned = df_cleaned[
                    (df_cleaned[column] >= lower_bound) &
                    (df_cleaned[column] <= upper_bound)
                    ]

        return df_cleaned

    async def _standardize_text(self, df: pd.DataFrame, options: Dict[str, Any]) -> pd.DataFrame:
        """Standardize text columns"""
        df_cleaned = df.copy()

        for column in df_cleaned.columns:
            if df_cleaned[column].dtype == 'object':
                df_cleaned[column] = df_cleaned[column].astype(str).str.strip().str.lower()

        return df_cleaned

    async def _normalize_data(self, df: pd.DataFrame, options: Dict[str, Any]) -> pd.DataFrame:
        """Normalize numeric data"""
        method = options.get('method', 'standard')
        columns = options.get('columns', [])

        if not columns:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        df_cleaned = df.copy()

        for column in columns:
            if column in df_cleaned.columns and pd.api.types.is_numeric_dtype(df_cleaned[column]):
                if method == 'standard':
                    df_cleaned[column] = (df_cleaned[column] - df_cleaned[column].mean()) / df_cleaned[column].std()
                elif method == 'minmax':
                    df_cleaned[column] = (df_cleaned[column] - df_cleaned[column].min()) / (
                                df_cleaned[column].max() - df_cleaned[column].min())

        return df_cleaned

    async def _encode_categorical(self, df: pd.DataFrame, options: Dict[str, Any]) -> pd.DataFrame:
        """Encode categorical variables"""
        method = options.get('method', 'label')
        columns = options.get('columns', [])

        if not columns:
            columns = df.select_dtypes(include=['object']).columns.tolist()

        df_cleaned = df.copy()

        for column in columns:
            if column in df_cleaned.columns and df_cleaned[column].dtype == 'object':
                if method == 'label':
                    df_cleaned[column] = pd.Categorical(df_cleaned[column]).codes
                elif method == 'onehot':
                    # For one-hot encoding, you might want to handle this differently
                    dummies = pd.get_dummies(df_cleaned[column], prefix=column)
                    df_cleaned = pd.concat([df_cleaned, dummies], axis=1)
                    df_cleaned = df_cleaned.drop(columns=[column])

        return df_cleaned

    def get_available_operations(self) -> List[str]:
        """Get list of available cleaning operations"""
        return list(self.available_operations.keys())

    def get_default_options(self) -> Dict[str, Any]:
        """Get default options for each operation"""
        return {
            'handle_missing_values': {'strategy': 'auto', 'threshold': 0.5},
            'remove_duplicates': {'keep': 'first'},
            'fix_data_types': {},
            'remove_outliers': {'method': 'iqr'},
            'standardize_text': {},
            'normalize_data': {'method': 'standard'},
            'encode_categorical': {'method': 'label'}
        }