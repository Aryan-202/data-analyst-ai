import pandas as pd
import os
import uuid
from datetime import datetime
from typing import Dict, Any, List
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
import matplotlib.pyplot as plt
import io
import base64

from server.config import settings
from server.utils.logger import setup_logger

logger = setup_logger()


class ReportGenerator:
    """Generates reports in various formats (PDF, Excel, PPT)"""

    def __init__(self):
        self.report_templates = {
            'pdf': self._generate_pdf_report,
            'excel': self._generate_excel_report,
            'ppt': self._generate_ppt_report
        }

    async def generate_report(self, df: pd.DataFrame, report_type: str, dataset_id: str,
                              additional_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive report"""

        if report_type not in self.report_templates:
            raise ValueError(f"Unsupported report type: {report_type}")

        try:
            report_id = str(uuid.uuid4())
            filename = f"{report_id}_{dataset_id}.{report_type}"
            file_path = os.path.join(settings.REPORTS_DIR, filename)

            # Generate report
            report_data = await self.report_templates[report_type](
                df, dataset_id, additional_data, file_path
            )

            logger.info(f"Report generated: {file_path}")

            return {
                'report_id': report_id,
                'filename': filename,
                'file_path': file_path,
                'report_type': report_type,
                'generated_at': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Report generation failed: {str(e)}")
            raise

    async def _generate_pdf_report(self, df: pd.DataFrame, dataset_id: str,
                                   additional_data: Dict[str, Any], file_path: str) -> Dict[str, Any]:
        """Generate PDF report"""
        try:
            doc = SimpleDocTemplate(file_path, pagesize=letter)
            styles = getSampleStyleSheet()
            story = []

            # Title
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=18,
                spaceAfter=30,
                textColor=colors.darkblue
            )
            story.append(Paragraph("Data Analysis Report", title_style))
            story.append(Spacer(1, 12))

            # Dataset Information
            story.append(Paragraph("Dataset Overview", styles['Heading2']))
            overview_data = [
                ["Dataset ID", dataset_id],
                ["Total Rows", str(len(df))],
                ["Total Columns", str(len(df.columns))],
                ["Generated On", datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
            ]
            overview_table = Table(overview_data, colWidths=[200, 200])
            overview_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
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
            column_data = [["Column Name", "Data Type", "Non-Null Count", "Null Count"]]