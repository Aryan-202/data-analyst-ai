import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
import re


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
        df = pd.read_csv(file_path)  # Assuming CSV for simplicity

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

        return quality_report

    def _remove_duplicates(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Remove duplicate rows"""
        original_count = len(df)
        df_cleaned = df.drop_duplicates()
        removed_count = original_count - len(df_cleaned)

        report = {
            'duplicates_found': removed_count,
            'remaining_rows': len(df_cleaned)
        }

        return df_cleaned, report

    def _handle_missing_values(self, df: pd.DataFrame, strategy: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Handle missing values based on strategy"""
        missing_before = df.isnull().sum().sum()

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

        missing_after = df.isnull().sum().sum()

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
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        outliers_report = {}

        for column in numeric_columns:
            if method == 'iqr':
                Q1 = df[column].quantile(0.25)
                Q3 = df[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
                outliers_count = len(outliers)

                # Cap outliers instead of removing
                df[column] = np.where(df[column] < lower_bound, lower_bound, df[column])
                df[column] = np.where(df[column] > upper_bound, upper_bound, df[column])

            elif method == 'zscore':
                from scipy import stats
                z_scores = np.abs(stats.zscore(df[column].dropna()))
                outliers_count = len(z_scores[z_scores > 3])

                # Cap outliers
                mean = df[column].mean()
                std = df[column].std()
                df[column] = np.where(
                    np.abs((df[column] - mean) / std) > 3,
                    np.sign(df[column] - mean) * 3 * std + mean,
                    df[column]
                )

            outliers_report[column] = outliers_count

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
            if abs(df[column].skew()) > 2:  # Highly skewed
                skewed_columns.append(column)

        if skewed_columns:
            issues['skewed_columns'] = skewed_columns

        return issues