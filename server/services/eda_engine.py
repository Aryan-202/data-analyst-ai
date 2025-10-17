import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Any, List
import warnings

warnings.filterwarnings('ignore')


class EDAEngine:
    def __init__(self):
        self.analysis_methods = {
            'summary': self._generate_summary,
            'correlation': self._analyze_correlations,
            'outliers': self._detect_outliers,
            'distributions': self._analyze_distributions,
            'trends': self._analyze_trends
        }

    async def analyze_dataset(self, file_path: str, analysis_types: List[str]) -> Dict[str, Any]:
        """Perform comprehensive EDA"""
        df = pd.read_csv(file_path)
        results = {}

        for analysis_type in analysis_types:
            if analysis_type in self.analysis_methods:
                results[analysis_type] = await self.analysis_methods[analysis_type](df)

        return results

    async def get_basic_summary(self, file_path: str) -> Dict[str, Any]:
        """Get basic dataset summary"""
        df = pd.read_csv(file_path)
        return await self._generate_summary(df)

    async def _generate_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive summary statistics"""
        summary = {
            'dataset_info': {
                'rows': len(df),
                'columns': len(df.columns),
                'memory_usage': df.memory_usage(deep=True).sum()
            },
            'column_stats': {},
            'missing_values': df.isnull().sum().to_dict(),
            'data_types': df.dtypes.astype(str).to_dict()
        }

        for column in df.columns:
            col_data = df[column]
            col_stats = {
                'data_type': str(col_data.dtype),
                'unique_values': col_data.nunique(),
                'missing_values': col_data.isnull().sum(),
                'missing_percentage': (col_data.isnull().sum() / len(col_data)) * 100
            }

            if pd.api.types.is_numeric_dtype(col_data):
                col_stats.update({
                    'mean': col_data.mean(),
                    'median': col_data.median(),
                    'std': col_data.std(),
                    'min': col_data.min(),
                    'max': col_data.max(),
                    'skewness': col_data.skew(),
                    'kurtosis': col_data.kurtosis(),
                    'quartiles': {
                        'q1': col_data.quantile(0.25),
                        'q2': col_data.quantile(0.5),
                        'q3': col_data.quantile(0.75)
                    }
                })
            else:
                col_stats.update({
                    'top_categories': col_data.value_counts().head(5).to_dict(),
                    'most_frequent': col_data.mode().iloc[0] if not col_data.mode().empty else None
                })

            summary['column_stats'][column] = col_stats

        return summary

    async def _analyze_correlations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze correlations between numeric variables"""
        numeric_df = df.select_dtypes(include=[np.number])

        if numeric_df.empty:
            return {'message': 'No numeric columns for correlation analysis'}

        correlation_matrix = numeric_df.corr()

        # Find strong correlations
        strong_correlations = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i + 1, len(correlation_matrix.columns)):
                corr_value = correlation_matrix.iloc[i, j]
                if abs(corr_value) > 0.5:  # Threshold for strong correlation
                    strong_correlations.append({
                        'variable1': correlation_matrix.columns[i],
                        'variable2': correlation_matrix.columns[j],
                        'correlation': corr_value,
                        'strength': 'strong' if abs(corr_value) > 0.7 else 'moderate'
                    })

        return {
            'correlation_matrix': correlation_matrix.to_dict(),
            'strong_correlations': strong_correlations,
            'heatmap_data': self._prepare_heatmap_data(correlation_matrix)
        }

    async def _detect_outliers(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect outliers in numeric columns"""
        numeric_df = df.select_dtypes(include=[np.number])
        outliers_report = {}

        for column in numeric_df.columns:
            col_data = numeric_df[column].dropna()

            # IQR method
            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]

            # Z-score method
            z_scores = np.abs(stats.zscore(col_data))
            z_outliers = col_data[z_scores > 3]

            outliers_report[column] = {
                'iqr_method': {
                    'outliers_count': len(outliers),
                    'outliers_percentage': (len(outliers) / len(col_data)) * 100,
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound
                },
                'zscore_method': {
                    'outliers_count': len(z_outliers),
                    'outliers_percentage': (len(z_outliers) / len(col_data)) * 100
                }
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
                'is_normal': stats.normaltest(col_data).pvalue > 0.05,
                'skewness': col_data.skew(),
                'kurtosis': col_data.kurtosis()
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
                            'slope': slope,
                            'trend_direction': 'increasing' if slope > 0 else 'decreasing',
                            'trend_strength': abs(slope),
                            'data_points': len(trend_data)
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
                    'value': correlation_matrix.iloc[i, j]
                })

        return heatmap_data

    def _prepare_histogram_data(self, data: pd.Series, bins: int = 20) -> Dict:
        """Prepare histogram data for visualization"""
        counts, bin_edges = np.histogram(data.dropna(), bins=bins)

        return {
            'counts': counts.tolist(),
            'bin_edges': bin_edges.tolist(),
            'bin_centers': ((bin_edges[:-1] + bin_edges[1:]) / 2).tolist()
        }