import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Any, List
import warnings
from utils.logger import setup_logger

warnings.filterwarnings('ignore')
logger = setup_logger(__name__)


class EDAEngine:
    def __init__(self):
        self.analysis_methods = {
            'summary': self._generate_summary,
            'correlation': self._analyze_correlations,
            'outliers': self._detect_outliers,
            'distributions': self._analyze_distributions,
            'trends': self._analyze_trends
        }

    async def analyze_dataset(self, df: pd.DataFrame, analysis_types: List[str]) -> Dict[str, Any]:
        """Perform comprehensive EDA on dataset"""
        results = {}

        # Validate analysis types
        valid_analysis_types = [at for at in analysis_types if at in self.analysis_methods]
        invalid_analysis_types = [at for at in analysis_types if at not in self.analysis_methods]

        if invalid_analysis_types:
            logger.warning(f"Invalid analysis types requested: {invalid_analysis_types}")

        # Perform requested analyses
        for analysis_type in valid_analysis_types:
            try:
                logger.info(f"Performing {analysis_type} analysis...")
                results[analysis_type] = await self.analysis_methods[analysis_type](df)
            except Exception as e:
                logger.error(f"Analysis '{analysis_type}' failed: {str(e)}")
                results[analysis_type] = {
                    'error': f"Analysis failed: {str(e)}",
                    'traceback': str(e)
                }

        # Always generate basic dataset info
        if 'summary' not in results:
            try:
                results['summary'] = await self._generate_summary(df)
            except Exception as e:
                results['summary'] = {'error': f"Summary generation failed: {str(e)}"}

        return results

    async def get_basic_summary(self, file_path: str) -> Dict[str, Any]:
        """Get basic dataset summary"""
        df = pd.read_csv(file_path)
        summary = await self._generate_summary(df)

        # FIX: Convert all numpy types to native Python types
        summary = self._convert_numpy_types(summary)

        return summary

    async def _generate_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive summary statistics"""
        summary = {
            'dataset_info': {
                'rows': int(len(df)),  # Convert to Python int
                'columns': int(len(df.columns)),  # Convert to Python int
                'memory_usage': float(df.memory_usage(deep=True).sum())  # Convert to Python float
            },
            'column_stats': {},
            'missing_values': df.isnull().sum().to_dict(),
            'data_types': df.dtypes.astype(str).to_dict()
        }

        for column in df.columns:
            col_data = df[column]
            col_stats = {
                'data_type': str(col_data.dtype),
                'unique_values': int(col_data.nunique()),  # Convert to Python int
                'missing_values': int(col_data.isnull().sum()),  # Convert to Python int
                'missing_percentage': float((col_data.isnull().sum() / len(col_data)) * 100)  # Convert to Python float
            }

            if pd.api.types.is_numeric_dtype(col_data):
                col_stats.update({
                    'mean': float(col_data.mean()) if not pd.isna(col_data.mean()) else None,
                    'median': float(col_data.median()) if not pd.isna(col_data.median()) else None,
                    'std': float(col_data.std()) if not pd.isna(col_data.std()) else None,
                    'min': float(col_data.min()) if not pd.isna(col_data.min()) else None,
                    'max': float(col_data.max()) if not pd.isna(col_data.max()) else None,
                    'skewness': float(col_data.skew()) if not pd.isna(col_data.skew()) else None,
                    'kurtosis': float(col_data.kurtosis()) if not pd.isna(col_data.kurtosis()) else None,
                    'quartiles': {
                        'q1': float(col_data.quantile(0.25)) if not pd.isna(col_data.quantile(0.25)) else None,
                        'q2': float(col_data.quantile(0.5)) if not pd.isna(col_data.quantile(0.5)) else None,
                        'q3': float(col_data.quantile(0.75)) if not pd.isna(col_data.quantile(0.75)) else None
                    }
                })
            else:
                # For categorical columns, convert value counts to native types
                value_counts = col_data.value_counts().head(5)
                top_categories = {}
                for idx, count in value_counts.items():
                    top_categories[str(idx)] = int(count)  # Convert to Python int

                col_stats.update({
                    'top_categories': top_categories,
                    'most_frequent': str(col_data.mode().iloc[0]) if not col_data.mode().empty else None
                })

            summary['column_stats'][column] = col_stats

        return summary

    async def _analyze_correlations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze correlations between numeric variables"""
        numeric_df = df.select_dtypes(include=[np.number])

        if numeric_df.empty:
            return {'message': 'No numeric columns for correlation analysis'}

        correlation_matrix = numeric_df.corr()

        # Convert correlation matrix to native Python types
        corr_dict = {}
        for col in correlation_matrix.columns:
            corr_dict[col] = {}
            for idx in correlation_matrix.index:
                corr_dict[col][idx] = float(correlation_matrix.loc[idx, col])  # Convert to Python float

        # Find strong correlations
        strong_correlations = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i + 1, len(correlation_matrix.columns)):
                corr_value = correlation_matrix.iloc[i, j]
                if abs(corr_value) > 0.5:  # Threshold for strong correlation
                    strong_correlations.append({
                        'variable1': correlation_matrix.columns[i],
                        'variable2': correlation_matrix.columns[j],
                        'correlation': float(corr_value),  # Convert to Python float
                        'strength': 'strong' if abs(corr_value) > 0.7 else 'moderate'
                    })

        return {
            'correlation_matrix': corr_dict,
            'strong_correlations': strong_correlations,
            'heatmap_data': self._prepare_heatmap_data(correlation_matrix)
        }

    async def _detect_outliers(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect outliers in numeric columns using IQR method"""
        outliers_report = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            try:
                # Remove null values for this column
                col_data = df[col].dropna()

                # Skip if no data or only one value
                if len(col_data) < 2:
                    outliers_report[col] = {
                        'outlier_count': 0,
                        'outliers_percentage': 0.0,
                        'message': 'Insufficient data for outlier detection'
                    }
                    continue

                # Calculate IQR
                Q1 = col_data.quantile(0.25)
                Q3 = col_data.quantile(0.75)
                IQR = Q3 - Q1

                # Skip if IQR is zero (all values are same)
                if IQR == 0:
                    outliers_report[col] = {
                        'outlier_count': 0,
                        'outliers_percentage': 0.0,
                        'message': 'No variability in data (IQR=0)'
                    }
                    continue

                # Define bounds
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                # Detect outliers
                outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]

                # Calculate percentage safely
                total_count = len(col_data)
                outlier_count = len(outliers)
                outlier_percentage = (outlier_count / total_count) * 100 if total_count > 0 else 0

                outliers_report[col] = {
                    'outlier_count': outlier_count,
                    'outliers_percentage': round(outlier_percentage, 2),
                    'lower_bound': float(lower_bound),
                    'upper_bound': float(upper_bound),
                    'total_values': total_count,
                    'outlier_values': outliers.tolist()[:10]  # First 10 outliers
                }

            except Exception as e:
                logger.warning(f"Error detecting outliers for column '{col}': {str(e)}")
                outliers_report[col] = {
                    'outlier_count': 0,
                    'outliers_percentage': 0.0,
                    'error': f"Outlier detection failed: {str(e)}"
                }

        return outliers_report

    async def _analyze_distributions(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze distributions of numeric variables"""
        numeric_df = df.select_dtypes(include=[np.number])
        distributions = {}

        for column in numeric_df.columns:
            col_data = numeric_df[column].dropna()

            distributions[column] = {
                'histogram_data': self._prepare_histogram_data(col_data),
                'is_normal': bool(stats.normaltest(col_data).pvalue > 0.05),  # Convert to Python bool
                'skewness': float(col_data.skew()) if not pd.isna(col_data.skew()) else None,  # Convert to Python float
                'kurtosis': float(col_data.kurtosis()) if not pd.isna(col_data.kurtosis()) else None
                # Convert to Python float
            }

        return distributions

    async def _analyze_trends(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze trends in time series data"""
        # Identify potential datetime columns
        datetime_columns = []
        for column in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[column]):
                datetime_columns.append(column)
            else:
                # Try to convert to datetime
                try:
                    pd.to_datetime(df[column])
                    datetime_columns.append(column)
                except:
                    continue

        trends = {}
        for date_col in datetime_columns[:1]:  # Analyze first datetime column
            try:
                df_sorted = df.sort_values(date_col)
                numeric_cols = df_sorted.select_dtypes(include=[np.number]).columns

                for num_col in numeric_cols[:3]:  # Analyze first 3 numeric columns
                    trend_data = df_sorted[[date_col, num_col]].dropna()
                    if len(trend_data) > 1:
                        # Calculate trend
                        x = np.arange(len(trend_data))
                        y = trend_data[num_col].values
                        slope, intercept = np.polyfit(x, y, 1)

                        trends[f'{num_col}_over_{date_col}'] = {
                            'slope': float(slope),  # Convert to Python float
                            'trend_direction': 'increasing' if slope > 0 else 'decreasing',
                            'trend_strength': float(abs(slope)),  # Convert to Python float
                            'data_points': int(len(trend_data))  # Convert to Python int
                        }
            except:
                continue

        return trends

    def _prepare_heatmap_data(self, correlation_matrix: pd.DataFrame) -> List[Dict]:
        """Prepare correlation matrix data for heatmap visualization"""
        heatmap_data = []
        columns = correlation_matrix.columns

        for i, col1 in enumerate(columns):
            for j, col2 in enumerate(columns):
                heatmap_data.append({
                    'x': col1,
                    'y': col2,
                    'value': float(correlation_matrix.iloc[i, j])  # Convert to Python float
                })

        return heatmap_data

    def _prepare_histogram_data(self, data: pd.Series, bins: int = 20) -> Dict:
        """Prepare histogram data for visualization"""
        counts, bin_edges = np.histogram(data.dropna(), bins=bins)

        return {
            'counts': [int(count) for count in counts.tolist()],  # Convert to Python int
            'bin_edges': [float(edge) for edge in bin_edges.tolist()],  # Convert to Python float
            'bin_centers': [float(center) for center in ((bin_edges[:-1] + bin_edges[1:]) / 2).tolist()]
            # Convert to Python float
        }

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

    def _validate_dataframe(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate dataframe before analysis"""
        validation_report = {
            'is_valid': True,
            'issues': [],
            'warnings': []
        }

        # Check if dataframe is empty
        if df.empty:
            validation_report['is_valid'] = False
            validation_report['issues'].append("DataFrame is empty")
            return validation_report

        # Check for all null columns
        all_null_cols = df.columns[df.isnull().all()].tolist()
        if all_null_cols:
            validation_report['warnings'].append(f"Columns with all null values: {all_null_cols}")

        # Check for single value columns
        single_value_cols = []
        for col in df.columns:
            if df[col].nunique() <= 1:
                single_value_cols.append(col)

        if single_value_cols:
            validation_report['warnings'].append(f"Columns with single value: {single_value_cols}")

        # Check for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            validation_report['warnings'].append("No numeric columns found for statistical analysis")

        return validation_report