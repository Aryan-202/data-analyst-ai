import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, Any, List, Optional
import base64
import io
from server.utils.logger import setup_logger

logger = setup_logger()


class VisualizationEngine:
    """Generates automatic visualizations and charts"""

    def __init__(self):
        self.chart_templates = {
            'distribution': self._create_distribution_chart,
            'correlation': self._create_correlation_chart,
            'timeseries': self._create_timeseries_chart,
            'categorical': self._create_categorical_chart,
            'scatter': self._create_scatter_plot,
            'box': self._create_box_plot
        }

    async def auto_generate_charts(self, df: pd.DataFrame, max_charts: int = 8) -> List[Dict[str, Any]]:
        """Automatically generate relevant charts based on data types"""
        charts = []
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        datetime_cols = df.select_dtypes(include=['datetime']).columns.tolist()

        # 1. Distribution charts for numeric columns
        for col in numeric_cols[:3]:  # First 3 numeric columns
            chart = await self._create_distribution_chart(df, col)
            if chart:
                charts.append(chart)

        # 2. Correlation heatmap if enough numeric columns
        if len(numeric_cols) >= 2:
            chart = await self._create_correlation_chart(df)
            if chart:
                charts.append(chart)

        # 3. Categorical charts
        for col in categorical_cols[:2]:  # First 2 categorical columns
            chart = await self._create_categorical_chart(df, col)
            if chart:
                charts.append(chart)

        # 4. Timeseries if datetime column exists
        if datetime_cols and numeric_cols:
            for date_col in datetime_cols[:1]:
                for num_col in numeric_cols[:1]:
                    chart = await self._create_timeseries_chart(df, date_col, num_col)
                    if chart:
                        charts.append(chart)

        # 5. Scatter plot for correlated numeric columns
        if len(numeric_cols) >= 2:
            chart = await self._create_scatter_plot(df, numeric_cols[0], numeric_cols[1])
            if chart:
                charts.append(chart)

        return charts[:max_charts]

    async def generate_specific_chart(self, df: pd.DataFrame, chart_type: str,
                                      x_axis: Optional[str] = None, y_axis: Optional[str] = None,
                                      color_by: Optional[str] = None) -> List[Dict[str, Any]]:
        """Generate specific chart type"""
        if chart_type in self.chart_templates:
            chart_func = self.chart_templates[chart_type]
            chart = await chart_func(df, x_axis, y_axis, color_by)
            return [chart] if chart else []
        else:
            logger.warning(f"Unknown chart type: {chart_type}")
            return []

    async def _create_distribution_chart(self, df: pd.DataFrame, column: str, **kwargs) -> Dict[str, Any]:
        """Create distribution chart (histogram)"""
        if column not in df.columns or not pd.api.types.is_numeric_dtype(df[column]):
            return None

        try:
            fig = px.histogram(df, x=column, title=f"Distribution of {column}",
                               labels={column: column}, opacity=0.7)
            fig.update_layout(showlegend=False)

            return self._format_chart_response(fig, "distribution", f"histogram_{column}")
        except Exception as e:
            logger.error(f"Error creating distribution chart: {str(e)}")
            return None

    async def _create_correlation_chart(self, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Create correlation heatmap"""
        numeric_df = df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) < 2:
            return None

        try:
            corr_matrix = numeric_df.corr()

            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdBu',
                zmid=0,
                text=corr_matrix.round(2).values,
                texttemplate='%{text}',
                hoverinfo='text'
            ))

            fig.update_layout(
                title="Correlation Heatmap",
                xaxis_title="Columns",
                yaxis_title="Columns"
            )

            return self._format_chart_response(fig, "correlation", "correlation_heatmap")
        except Exception as e:
            logger.error(f"Error creating correlation chart: {str(e)}")
            return None

    async def _create_timeseries_chart(self, df: pd.DataFrame, date_column: str, value_column: str, **kwargs) -> Dict[
        str, Any]:
        """Create timeseries chart"""
        if date_column not in df.columns or value_column not in df.columns:
            return None

        if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
            return None

        try:
            fig = px.line(df, x=date_column, y=value_column,
                          title=f"{value_column} over Time")

            return self._format_chart_response(fig, "timeseries", f"timeseries_{value_column}")
        except Exception as e:
            logger.error(f"Error creating timeseries chart: {str(e)}")
            return None

    async def _create_categorical_chart(self, df: pd.DataFrame, column: str, **kwargs) -> Dict[str, Any]:
        """Create categorical bar chart"""
        if column not in df.columns or df[column].dtype != 'object':
            return None

        try:
            value_counts = df[column].value_counts().head(10)  # Top 10 categories

            fig = px.bar(x=value_counts.index, y=value_counts.values,
                         title=f"Top Categories in {column}",
                         labels={'x': column, 'y': 'Count'})

            return self._format_chart_response(fig, "categorical", f"bar_{column}")
        except Exception as e:
            logger.error(f"Error creating categorical chart: {str(e)}")
            return None

    async def _create_scatter_plot(self, df: pd.DataFrame, x_col: str, y_col: str, color_by: Optional[str] = None,
                                   **kwargs) -> Dict[str, Any]:
        """Create scatter plot"""
        if x_col not in df.columns or y_col not in df.columns:
            return None

        if not (pd.api.types.is_numeric_dtype(df[x_col]) and pd.api.types.is_numeric_dtype(df[y_col])):
            return None

        try:
            if color_by and color_by in df.columns:
                fig = px.scatter(df, x=x_col, y=y_col, color=color_by,
                                 title=f"{y_col} vs {x_col}")
            else:
                fig = px.scatter(df, x=x_col, y=y_col,
                                 title=f"{y_col} vs {x_col}")

            return self._format_chart_response(fig, "scatter", f"scatter_{x_col}_{y_col}")
        except Exception as e:
            logger.error(f"Error creating scatter plot: {str(e)}")
            return None

    async def _create_box_plot(self, df: pd.DataFrame, column: str, **kwargs) -> Dict[str, Any]:
        """Create box plot"""
        if column not in df.columns or not pd.api.types.is_numeric_dtype(df[column]):
            return None

        try:
            fig = px.box(df, y=column, title=f"Box Plot of {column}")
            return self._format_chart_response(fig, "box", f"box_{column}")
        except Exception as e:
            logger.error(f"Error creating box plot: {str(e)}")
            return None

    def _format_chart_response(self, fig, chart_type: str, chart_id: str) -> Dict[str, Any]:
        """Format chart response with base64 encoding"""
        try:
            # Convert to base64 for easy transmission
            img_bytes = fig.to_image(format="png", width=800, height=500)
            img_base64 = base64.b64encode(img_bytes).decode('utf-8')

            # Get chart data as JSON for frontend rendering
            chart_json = fig.to_json()

            return {
                'chart_id': chart_id,
                'chart_type': chart_type,
                'title': fig.layout.title.text if fig.layout.title else chart_id,
                'image_base64': img_base64,
                'chart_json': chart_json,
                'config': {
                    'responsive': True,
                    'displayModeBar': True
                }
            }
        except Exception as e:
            logger.error(f"Error formatting chart response: {str(e)}")
            return None

    def extract_insights_from_charts(self, charts: List[Dict[str, Any]], df: pd.DataFrame) -> List[str]:
        """Extract basic insights from generated charts"""
        insights = []

        for chart in charts:
            chart_type = chart.get('chart_type')

            if chart_type == 'distribution':
                insights.append("Check distributions for normality and outliers")
            elif chart_type == 'correlation':
                insights.append("Review strong correlations for potential relationships")
            elif chart_type == 'timeseries':
                insights.append("Look for trends and seasonal patterns over time")
            elif chart_type == 'categorical':
                insights.append("Identify dominant categories and their frequencies")

        # Add general insights
        if len(df.select_dtypes(include=[np.number]).columns) >= 3:
            insights.append("Multiple numeric variables available for multivariate analysis")

        if len(df.select_dtypes(include=['object']).columns) >= 2:
            insights.append("Categorical variables can be used for segmentation analysis")

        return insights[:5]  # Return top 5 insights

    def get_available_chart_types(self) -> Dict[str, Any]:
        """Get available chart types and their requirements"""
        return {
            'distribution': {
                'description': 'Histogram showing value distribution',
                'requirements': ['One numeric column'],
                'best_for': ['Understanding data spread', 'Identifying outliers']
            },
            'correlation': {
                'description': 'Heatmap showing correlations between numeric columns',
                'requirements': ['At least 2 numeric columns'],
                'best_for': ['Finding relationships', 'Feature selection']
            },
            'timeseries': {
                'description': 'Line chart showing trends over time',
                'requirements': ['One datetime column', 'One numeric column'],
                'best_for': ['Trend analysis', 'Seasonality detection']
            },
            'categorical': {
                'description': 'Bar chart showing category frequencies',
                'requirements': ['One categorical column'],
                'best_for': ['Category comparison', 'Dominant category identification']
            },
            'scatter': {
                'description': 'Scatter plot showing relationship between two variables',
                'requirements': ['Two numeric columns'],
                'best_for': ['Correlation visualization', 'Cluster identification']
            },
            'box': {
                'description': 'Box plot showing distribution statistics',
                'requirements': ['One numeric column'],
                'best_for': ['Outlier detection', 'Distribution comparison']
            }
        }

    def get_chart_selection_criteria(self) -> Dict[str, Any]:
        """Get criteria for automatic chart selection"""
        return {
            'numeric_columns': ['distribution', 'box', 'correlation', 'scatter'],
            'categorical_columns': ['categorical'],
            'datetime_columns': ['timeseries'],
            'mixed_columns': ['scatter with color encoding']
        }