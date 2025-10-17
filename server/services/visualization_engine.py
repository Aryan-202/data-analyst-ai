import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, Any, List, Optional
import json


class VisualizationEngine:
    def __init__(self):
        self.chart_types = {
            'histogram': self._create_histogram,
            'scatter': self._create_scatter_plot,
            'line': self._create_line_plot,
            'bar': self._create_bar_chart,
            'box': self._create_box_plot,
            'heatmap': self._create_heatmap,
            'pie': self._create_pie_chart
        }

    async def create_chart(self, file_path: str, chart_type: str,
                           x_axis: Optional[str] = None, y_axis: Optional[str] = None,
                           color_by: Optional[str] = None, filters: Optional[Dict] = None) -> Dict[str, Any]:
        """Create specific chart type"""
        df = pd.read_csv(file_path)

        # Apply filters if provided
        if filters:
            df = self._apply_filters(df, filters)

        if chart_type in self.chart_types:
            chart_data = await self.chart_types[chart_type](df, x_axis, y_axis, color_by)
            return chart_data
        else:
            raise ValueError(f"Unsupported chart type: {chart_type}")

    async def auto_generate_charts(self, file_path: str, max_charts: int = 6) -> List[Dict[str, Any]]:
        """Automatically generate meaningful visualizations"""
        df = pd.read_csv(file_path)
        charts = []

        # Get dataset characteristics
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        datetime_cols = self._identify_datetime_columns(df)

        # 1. Distribution of numeric variables (histograms)
        for col in numeric_cols[:2]:  # First 2 numeric columns
            if len(charts) >= max_charts:
                break
            chart = await self._create_histogram(df, col)
            charts.append({
                'type': 'histogram',
                'title': f'Distribution of {col}',
                'data': chart
            })

        # 2. Correlation heatmap if multiple numeric columns
        if len(numeric_cols) > 1 and len(charts) < max_charts:
            chart = await self._create_heatmap(df)
            charts.append({
                'type': 'heatmap',
                'title': 'Correlation Matrix',
                'data': chart
            })

        # 3. Scatter plot for strongest correlation
        if len(numeric_cols) >= 2 and len(charts) < max_charts:
            corr_matrix = df[numeric_cols].corr()
            strong_corr = self._find_strongest_correlation(corr_matrix)
            if strong_corr:
                chart = await self._create_scatter_plot(df, strong_corr[0], strong_corr[1])
                charts.append({
                    'type': 'scatter',
                    'title': f'{strong_corr[0]} vs {strong_corr[1]}',
                    'data': chart
                })

        # 4. Bar charts for categorical variables
        for col in categorical_cols[:2]:
            if len(charts) >= max_charts:
                break
            if df[col].nunique() <= 20:  # Avoid high cardinality
                chart = await self._create_bar_chart(df, col)
                charts.append({
                    'type': 'bar',
                    'title': f'Distribution of {col}',
                    'data': chart
                })

        # 5. Box plots for numeric by categorical
        if categorical_cols and numeric_cols and len(charts) < max_charts:
            cat_col = categorical_cols[0]
            num_col = numeric_cols[0]
            if df[cat_col].nunique() <= 10:
                chart = await self._create_box_plot(df, cat_col, num_col)
                charts.append({
                    'type': 'box',
                    'title': f'{num_col} by {cat_col}',
                    'data': chart
                })

        return charts

    async def get_chart_suggestions(self, file_path: str) -> List[Dict[str, Any]]:
        """Get suggested chart types for the dataset"""
        df = pd.read_csv(file_path)
        suggestions = []

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

        # Histogram suggestions for numeric columns
        for col in numeric_cols:
            suggestions.append({
                'chart_type': 'histogram',
                'x_axis': col,
                'reason': f'Show distribution of {col}',
                'suitable': True
            })

        # Scatter plot suggestions for numeric pairs
        if len(numeric_cols) >= 2:
            for i in range(min(3, len(numeric_cols))):
                for j in range(i + 1, min(4, len(numeric_cols))):
                    suggestions.append({
                        'chart_type': 'scatter',
                        'x_axis': numeric_cols[i],
                        'y_axis': numeric_cols[j],
                        'reason': f'Explore relationship between {numeric_cols[i]} and {numeric_cols[j]}',
                        'suitable': True
                    })

        # Bar chart suggestions for categorical columns
        for col in categorical_cols:
            if df[col].nunique() <= 20:
                suggestions.append({
                    'chart_type': 'bar',
                    'x_axis': col,
                    'reason': f'Show frequency distribution of {col}',
                    'suitable': True
                })

        return suggestions[:10]  # Return top 10 suggestions

    async def _create_histogram(self, df: pd.DataFrame, x_axis: str,
                                y_axis: Optional[str] = None, color_by: Optional[str] = None) -> Dict[str, Any]:
        """Create histogram chart"""
        fig = px.histogram(df, x=x_axis, color=color_by, marginal="box")
        return json.loads(fig.to_json())

    async def _create_scatter_plot(self, df: pd.DataFrame, x_axis: str, y_axis: str,
                                   color_by: Optional[str] = None) -> Dict[str, Any]:
        """Create scatter plot"""
        fig = px.scatter(df, x=x_axis, y=y_axis, color=color_by, trendline="ols")
        return json.loads(fig.to_json())

    async def _create_line_plot(self, df: pd.DataFrame, x_axis: str, y_axis: str,
                                color_by: Optional[str] = None) -> Dict[str, Any]:
        """Create line plot"""
        # Sort by x_axis for line plots
        df_sorted = df.sort_values(x_axis)
        fig = px.line(df_sorted, x=x_axis, y=y_axis, color=color_by)
        return json.loads(fig.to_json())

    async def _create_bar_chart(self, df: pd.DataFrame, x_axis: str,
                                y_axis: Optional[str] = None, color_by: Optional[str] = None) -> Dict[str, Any]:
        """Create bar chart"""
        if y_axis:
            # Grouped bar chart
            fig = px.bar(df, x=x_axis, y=y_axis, color=color_by)
        else:
            # Count plot
            value_counts = df[x_axis].value_counts().reset_index()
            value_counts.columns = [x_axis, 'count']
            fig = px.bar(value_counts, x=x_axis, y='count', color=x_axis)

        return json.loads(fig.to_json())

    async def _create_box_plot(self, df: pd.DataFrame, x_axis: str, y_axis: str,
                               color_by: Optional[str] = None) -> Dict[str, Any]:
        """Create box plot"""
        fig = px.box(df, x=x_axis, y=y_axis, color=color_by)
        return json.loads(fig.to_json())

    async def _create_heatmap(self, df: pd.DataFrame, x_axis: Optional[str] = None,
                              y_axis: Optional[str] = None, color_by: Optional[str] = None) -> Dict[str, Any]:
        """Create correlation heatmap"""
        numeric_df = df.select_dtypes(include=[np.number])
        corr_matrix = numeric_df.corr()

        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0
        ))

        fig.update_layout(title='Correlation Heatmap')
        return json.loads(fig.to_json())

    async def _create_pie_chart(self, df: pd.DataFrame, x_axis: str,
                                y_axis: Optional[str] = None, color_by: Optional[str] = None) -> Dict[str, Any]:
        """Create pie chart"""
        if y_axis:
            fig = px.pie(df, values=y_axis, names=x_axis)
        else:
            value_counts = df[x_axis].value_counts().reset_index()
            value_counts.columns = ['category', 'count']
            fig = px.pie(value_counts, values='count', names='category')

        return json.loads(fig.to_json())

    def _apply_filters(self, df: pd.DataFrame, filters: Dict) -> pd.DataFrame:
        """Apply filters to dataframe"""
        filtered_df = df.copy()

        for column, filter_info in filters.items():
            if column in filtered_df.columns:
                filter_type = filter_info.get('type')
                filter_value = filter_info.get('value')

                if filter_type == 'range' and pd.api.types.is_numeric_dtype(filtered_df[column]):
                    min_val, max_val = filter_value
                    filtered_df = filtered_df[(filtered_df[column] >= min_val) & (filtered_df[column] <= max_val)]
                elif filter_type == 'categorical':
                    filtered_df = filtered_df[filtered_df[column].isin(filter_value)]
                elif filter_type == 'text_contains':
                    filtered_df = filtered_df[filtered_df[column].str.contains(filter_value, na=False)]

        return filtered_df

    def _identify_datetime_columns(self, df: pd.DataFrame) -> List[str]:
        """Identify potential datetime columns"""
        datetime_cols = []
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                datetime_cols.append(col)
            else:
                try:
                    pd.to_datetime(df[col])
                    datetime_cols.append(col)
                except:
                    continue
        return datetime_cols

    def _find_strongest_correlation(self, corr_matrix: pd.DataFrame) -> Optional[tuple]:
        """Find the pair with strongest correlation"""
        strong_corr = None
        max_corr = 0

        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr_val = abs(corr_matrix.iloc[i, j])
                if corr_val > max_corr and corr_val < 1.0:
                    max_corr = corr_val
                    strong_corr = (corr_matrix.columns[i], corr_matrix.columns[j])

        return strong_corr