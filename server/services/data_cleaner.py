import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
import re
import json


class DataCleaner:
    def __init__(self):
        self.cleaning_strategies = {
            'missing_values': self._handle_missing_values,
            'duplicates': self._remove_duplicates,
            'data_types': self._fix_data_types,
            'outliers': self._handle_outliers
        }

    async def clean_dataset(self, file_path: str, options: Dict[str, Any]) -> Dict[str, Any]:
        """Clean dataset based on provided options"""
        df = pd.read_csv(file_path)

        cleaning_report = {
            'original_shape': df.shape,
            'cleaning_steps': [],
            'issues_found': {}
        }

        # Apply cleaning strategies
        if options.get('remove_duplicates', True):
            df, report = self._remove_duplicates(df)
            cleaning_report['cleaning_steps'].append('duplicates_removal')
            cleaning_report['issues_found']['duplicates'] = report

        if options.get('handle_missing_values', 'auto') != 'skip':
            df, report = self._handle_missing_values(df, options['handle_missing_values'])
            cleaning_report['cleaning_steps'].append('missing_values_handling')
            cleaning_report['issues_found']['missing_values'] = report

        if options.get('fix_data_types', True):
            df, report = self._fix_data_types(df)
            cleaning_report['cleaning_steps'].append('data_type_fixing')
            cleaning_report['issues_found']['data_types'] = report

        if options.get('remove_outliers', False):
            df, report = self._handle_outliers(df, options.get('outlier_method', 'iqr'))
            cleaning_report['cleaning_steps'].append('outlier_handling')
            cleaning_report['issues_found']['outliers'] = report

        cleaning_report['final_shape'] = df.shape
        cleaning_report['rows_removed'] = cleaning_report['original_shape'][0] - cleaning_report['final_shape'][0]

        # FIX: Convert numpy types to native Python types for JSON serialization
        cleaning_report = self._convert_numpy_types(cleaning_report)

        return {
            'cleaned_data': df,
            'report': cleaning_report
        }

    async def get_data_quality_report(self, file_path: str) -> Dict[str, Any]:
        """Generate data quality assessment report"""
        df = pd.read_csv(file_path)

        quality_report = {
            'basic_info': {
                'total_rows': len(df),
                'total_columns': len(df.columns),
                'memory_usage': df.memory_usage(deep=True).sum()
            },
            'completeness': {
                'missing_values': df.isnull().sum().to_dict(),
                'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
                'complete_rows': len(df.dropna()),
                'completeness_score': (1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
            },
            'uniqueness': {
                'duplicate_rows': df.duplicated().sum(),
                'unique_values_per_column': df.nunique().to_dict()
            },
            'data_types': df.dtypes.astype(str).to_dict(),
            'potential_issues': self._identify_potential_issues(df)
        }

        # FIX: Convert numpy types to native Python types
        quality_report = self._convert_numpy_types(quality_report)

        return quality_report

    def _remove_duplicates(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Remove duplicate rows"""
        original_count = len(df)
        df_cleaned = df.drop_duplicates()
        removed_count = original_count - len(df_cleaned)

        report = {
            'duplicates_found': int(removed_count),  # Convert to Python int
            'remaining_rows': int(len(df_cleaned))  # Convert to Python int
        }

        return df_cleaned, report

    def _handle_missing_values(self, df: pd.DataFrame, strategy: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Handle missing values based on strategy"""
        missing_before = int(df.isnull().sum().sum())  # Convert to Python int

        if strategy == 'auto':
            # Auto strategy: use different methods for different columns
            for column in df.columns:
                if df[column].dtype in ['float64', 'int64']:
                    # Numeric columns: fill with median
                    df[column].fillna(df[column].median(), inplace=True)
                else:
                    # Categorical columns: fill with mode
                    if not df[column].mode().empty:
                        df[column].fillna(df[column].mode()[0], inplace=True)
                    else:
                        df[column].fillna('Unknown', inplace=True)
        elif strategy == 'drop':
            df = df.dropna()
        elif strategy == 'fill_mean':
            df = df.fillna(df.mean(numeric_only=True))
        elif strategy == 'fill_median':
            df = df.fillna(df.median(numeric_only=True))
        elif strategy == 'fill_mode':
            for column in df.columns:
                if not df[column].mode().empty:
                    df[column].fillna(df[column].mode()[0], inplace=True)

        missing_after = int(df.isnull().sum().sum())  # Convert to Python int

        report = {
            'missing_before': missing_before,
            'missing_after': missing_after,
            'strategy_used': strategy
        }

        return df, report

    def _fix_data_types(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Automatically fix data types"""
        conversions = {}

        for column in df.columns:
            original_dtype = str(df[column].dtype)

            # Try to convert to numeric
            if df[column].dtype == 'object':
                # Check if it's actually numeric
                numeric_version = pd.to_numeric(df[column], errors='coerce')
                if numeric_version.notna().sum() > len(df) * 0.8:  # 80% can be converted
                    df[column] = numeric_version
                    conversions[column] = f'{original_dtype} -> numeric'

                # Check if it's datetime
                elif pd.to_datetime(df[column], errors='coerce').notna().sum() > len(df) * 0.8:
                    df[column] = pd.to_datetime(df[column], errors='coerce')
                    conversions[column] = f'{original_dtype} -> datetime'

        report = {
            'conversions_made': conversions
        }

        return df, report

    def _handle_outliers(self, df: pd.DataFrame, method: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Handle outliers using specified method"""
        numeric_df = df.select_dtypes(include=[np.number])
        outliers_report = {}

        for column in numeric_df.columns:
            col_data = df[column].dropna()

            if method == 'iqr':
                Q1 = float(col_data.quantile(0.25))  # Convert to Python float
                Q3 = float(col_data.quantile(0.75))  # Convert to Python float
                IQR = float(Q3 - Q1)  # Convert to Python float
                lower_bound = float(Q1 - 1.5 * IQR)  # Convert to Python float
                upper_bound = float(Q3 + 1.5 * IQR)  # Convert to Python float

                outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
                outliers_count = int(len(outliers))  # Convert to Python int

                # Cap outliers instead of removing
                df[column] = np.where(df[column] < lower_bound, lower_bound, df[column])
                df[column] = np.where(df[column] > upper_bound, upper_bound, df[column])

            elif method == 'zscore':
                from scipy import stats
                z_scores = np.abs(stats.zscore(col_data))
                outliers_count = int(len(z_scores[z_scores > 3]))  # Convert to Python int

                # Cap outliers
                mean = float(col_data.mean())  # Convert to Python float
                std = float(col_data.std())  # Convert to Python float
                df[column] = np.where(
                    np.abs((df[column] - mean) / std) > 3,
                    np.sign(df[column] - mean) * 3 * std + mean,
                    df[column]
                )

            outliers_report[column] = {
                'outliers_count': outliers_count,
                'outliers_percentage': float((outliers_count / len(col_data)) * 100)  # Convert to Python float
            }

        report = {
            'method_used': method,
            'outliers_found': outliers_report
        }

        return df, report

    def _identify_potential_issues(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Identify potential data quality issues"""
        issues = {}

        # Check for constant columns
        constant_columns = []
        for column in df.columns:
            if df[column].nunique() == 1:
                constant_columns.append(column)

        if constant_columns:
            issues['constant_columns'] = constant_columns

        # Check for high cardinality categorical columns
        high_cardinality = []
        for column in df.select_dtypes(include=['object']).columns:
            if df[column].nunique() > len(df) * 0.5:  # More than 50% unique values
                high_cardinality.append(column)

        if high_cardinality:
            issues['high_cardinality_columns'] = high_cardinality

        # Check for skewed distributions
        skewed_columns = []
        for column in df.select_dtypes(include=[np.number]).columns:
            skew_val = df[column].skew()
            if not np.isnan(skew_val) and abs(skew_val) > 2:  # Highly skewed
                skewed_columns.append({
                    'column': column,
                    'skewness': float(skew_val)  # Convert to Python float
                })

        if skewed_columns:
            issues['skewed_columns'] = skewed_columns

        return issues

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