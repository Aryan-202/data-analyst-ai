import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import os
import uuid
from datetime import datetime
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
import matplotlib.pyplot as plt
import io
import base64


class ReportGenerator:
    def __init__(self, settings):
        self.settings = settings
        self.reports_storage = settings.reports_folder

    async def generate_report(self, file_path: str, report_type: str,
                              include_charts: bool = True, include_insights: bool = True,
                              include_models: bool = False, sections: Optional[List[str]] = None) -> Dict[str, Any]:
        """Generate comprehensive report"""
        df = pd.read_csv(file_path)
        report_id = str(uuid.uuid4())

        if report_type == 'pdf':
            report_path = await self._generate_pdf_report(df, report_id, include_charts, include_insights,
                                                          include_models, sections)
        elif report_type == 'excel':
            report_path = await self._generate_excel_report(df, report_id, include_charts, include_insights,
                                                            include_models, sections)
        else:
            raise ValueError(f"Unsupported report type: {report_type}")

        return {
            'report_id': report_id,
            'report_path': report_path,
            'report_type': report_type,
            'generated_at': datetime.now().isoformat()
        }

    async def _generate_pdf_report(self, df: pd.DataFrame, report_id: str,
                                   include_charts: bool, include_insights: bool,
                                   include_models: bool, sections: Optional[List[str]]) -> str:
        """Generate PDF report"""
        report_path = os.path.join(self.reports_storage, f"{report_id}.pdf")

        doc = SimpleDocTemplate(report_path, pagesize=letter)
        story = []
        styles = getSampleStyleSheet()

        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=18,
            spaceAfter=30,
            textColor=colors.darkblue
        )

        story.append(Paragraph("Data Analysis Report", title_style))
        story.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles['Normal']))
        story.append(Spacer(1, 20))

        # Dataset Overview
        story.append(Paragraph("Dataset Overview", styles['Heading2']))
        overview_data = [
            ['Total Rows', str(len(df))],
            ['Total Columns', str(len(df.columns))],
            ['Memory Usage', f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB"],
            ['Missing Values', str(df.isnull().sum().sum())]
        ]

        overview_table = Table(overview_data, colWidths=[2 * inch, 2 * inch])
        overview_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))

        story.append(overview_table)
        story.append(Spacer(1, 20))

        # Column Summary
        story.append(Paragraph("Column Summary", styles['Heading2']))
        column_data = [['Column Name', 'Data Type', 'Missing Values', 'Unique Values']]

        for col in df.columns:
            col_data = df[col]
            column_data.append([
                col,
                str(col_data.dtype),
                str(col_data.isnull().sum()),
                str(col_data.nunique())
            ])

        column_table = Table(column_data, colWidths=[1.5 * inch, 1 * inch, 1 * inch, 1 * inch])
        column_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTSIZE', (0, 1), (-1, -1), 8)
        ]))

        story.append(column_table)
        story.append(Spacer(1, 20))

        # Basic Statistics
        if include_insights:
            story.append(Paragraph("Basic Statistics", styles['Heading2']))

            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols[:3]:  # First 3 numeric columns
                story.append(Paragraph(f"Statistics for {col}", styles['Heading3']))
                stats = df[col].describe()

                stats_data = [
                    ['Metric', 'Value'],
                    ['Count', f"{stats['count']:.0f}"],
                    ['Mean', f"{stats['mean']:.2f}"],
                    ['Std Dev', f"{stats['std']:.2f}"],
                    ['Min', f"{stats['min']:.2f}"],
                    ['25%', f"{stats['25%']:.2f}"],
                    ['50%', f"{stats['50%']:.2f}"],
                    ['75%', f"{stats['75%']:.2f}"],
                    ['Max', f"{stats['max']:.2f}"]
                ]

                stats_table = Table(stats_data, colWidths=[1 * inch, 1 * inch])
                stats_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))

                story.append(stats_table)
                story.append(Spacer(1, 10))

        # Key Insights
        if include_insights:
            story.append(Paragraph("Key Insights", styles['Heading2']))

            insights = await self._generate_basic_insights(df)
            for insight in insights:
                story.append(Paragraph(f"• {insight}", styles['Normal']))
                story.append(Spacer(1, 5))

        # Build PDF
        doc.build(story)
        return report_path

    async def _generate_excel_report(self, df: pd.DataFrame, report_id: str,
                                     include_charts: bool, include_insights: bool,
                                     include_models: bool, sections: Optional[List[str]]) -> str:
        """Generate Excel report"""
        report_path = os.path.join(self.reports_storage, f"{report_id}.xlsx")

        with pd.ExcelWriter(report_path, engine='openpyxl') as writer:
            # Dataset sheet
            df.to_excel(writer, sheet_name='Dataset', index=False)

            # Summary sheet
            summary_data = []
            for col in df.columns:
                col_data = df[col]
                summary_data.append({
                    'Column': col,
                    'Data Type': str(col_data.dtype),
                    'Missing Values': col_data.isnull().sum(),
                    'Unique Values': col_data.nunique(),
                    'Mean': col_data.mean() if pd.api.types.is_numeric_dtype(col_data) else 'N/A',
                    'Std Dev': col_data.std() if pd.api.types.is_numeric_dtype(col_data) else 'N/A'
                })

            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)

            # Statistics sheet for numeric columns
            numeric_df = df.select_dtypes(include=[np.number])
            if not numeric_df.empty:
                stats_df = numeric_df.describe().T
                stats_df.to_excel(writer, sheet_name='Statistics')

            # Correlation matrix
            if len(numeric_df.columns) > 1:
                corr_matrix = numeric_df.corr()
                corr_matrix.to_excel(writer, sheet_name='Correlations')

        return report_path

    async def _generate_basic_insights(self, df: pd.DataFrame) -> List[str]:
        """Generate basic insights for report"""
        insights = []

        # Basic dataset insights
        insights.append(f"Dataset contains {len(df)} rows and {len(df.columns)} columns")

        # Missing values insight
        missing_total = df.isnull().sum().sum()
        if missing_total > 0:
            insights.append(f"Dataset has {missing_total} missing values that may need handling")
        else:
            insights.append("No missing values detected in the dataset")

        # Numeric columns insights
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            insights.append(f"Found {len(numeric_cols)} numeric columns for analysis")

            # Strong correlation insight
            if len(numeric_cols) >= 2:
                corr_matrix = df[numeric_cols].corr()
                strong_corrs = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i + 1, len(corr_matrix.columns)):
                        corr = corr_matrix.iloc[i, j]
                        if abs(corr) > 0.7:
                            strong_corrs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr))

                if strong_corrs:
                    strongest = max(strong_corrs, key=lambda x: abs(x[2]))
                    insights.append(
                        f"Strong correlation between {strongest[0]} and {strongest[1]} (r={strongest[2]:.2f})")

        # Categorical columns insights
        categorical_cols = df.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            insights.append(f"Found {len(categorical_cols)} categorical columns")

            for col in categorical_cols[:2]:  # First 2 categorical columns
                if df[col].nunique() <= 10:  # Low cardinality
                    top_category = df[col].value_counts().index[0]
                    top_count = df[col].value_counts().iloc[0]
                    insights.append(
                        f"'{top_category}' is the most frequent category in {col} ({top_count} occurrences)")

        return insights

    def _create_simple_chart(self, data: pd.Series, title: str) -> str:
        """Create simple chart and return base64 encoded image"""
        plt.figure(figsize=(8, 4))

        if pd.api.types.is_numeric_dtype(data):
            data.hist(bins=20)
            plt.title(f'Distribution of {title}')
        else:
            data.value_counts().head(10).plot(kind='bar')
            plt.title(f'Top Categories in {title}')

        plt.tight_layout()

        # Convert to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()

        return image_base64