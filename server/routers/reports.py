from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import os

from services.report_generator import ReportGenerator
from services.file_manager import FileManager
from config import get_settings, Settings

router = APIRouter()


class ReportRequest(BaseModel):
    file_id: str
    report_type: str  # pdf, excel, powerpoint
    include_charts: bool = True
    include_insights: bool = True
    include_models: bool = False
    sections: Optional[List[str]] = None


class ReportResponse(BaseModel):
    success: bool
    message: str
    report_id: str
    download_url: str


@router.post("/generate-report", response_model=ReportResponse)
async def generate_report(
        request: ReportRequest,
        settings: Settings = Depends(get_settings)
):
    """
    Generate comprehensive report
    """
    try:
        file_manager = FileManager(settings)
        report_generator = ReportGenerator(settings)

        file_path = file_manager.get_file_path(request.file_id)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")

        # Generate report
        report_info = await report_generator.generate_report(
            file_path,
            request.report_type,
            request.include_charts,
            request.include_insights,
            request.include_models,
            request.sections
        )

        download_url = f"/reports/{report_info['report_id']}.{request.report_type}"

        return ReportResponse(
            success=True,
            message=f"Report generated successfully",
            report_id=report_info['report_id'],
            download_url=download_url
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")


@router.get("/download-report/{report_id}")
async def download_report(
        report_id: str,
        report_type: str,
        settings: Settings = Depends(get_settings)
):
    """
    Download generated report
    """
    try:
        file_path = os.path.join(settings.reports_folder, f"{report_id}.{report_type}")

        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Report not found")

        media_type = {
            'pdf': 'application/pdf',
            'xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            'pptx': 'application/vnd.openxmlformats-officedocument.presentationml.presentation'
        }.get(report_type, 'application/octet-stream')

        return FileResponse(
            path=file_path,
            media_type=media_type,
            filename=f"data_report_{report_id}.{report_type}"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/report-templates")
async def get_report_templates():
    """
    Get available report templates
    """
    return {
        "templates": [
            {
                "id": "executive_summary",
                "name": "Executive Summary",
                "description": "High-level insights for management"
            },
            {
                "id": "technical_detailed",
                "name": "Technical Detailed",
                "description": "Comprehensive analysis with technical details"
            },
            {
                "id": "dashboard",
                "name": "Interactive Dashboard",
                "description": "Interactive charts and filters"
            }
        ]
    }