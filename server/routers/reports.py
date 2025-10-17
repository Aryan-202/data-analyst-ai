from fastapi import APIRouter, HTTPException, BackgroundTasks
from datetime import datetime
import os
from server.models.dataset_schema import ReportRequest, ReportResponse
from server.services.report_generator import ReportGenerator
from server.services.data_loader import DataLoader
from server.utils.logger import setup_logger

router = APIRouter()
logger = setup_logger()
report_generator = ReportGenerator()
data_loader = DataLoader()


@router.post("/generate", response_model=ReportResponse)
async def generate_report(
        background_tasks: BackgroundTasks,
        request: ReportRequest
):
    """
    Generate comprehensive reports (PDF, Excel, PPT)
    """
    try:
        # Load dataset
        df = data_loader.load_dataset_by_id(request.dataset_id)
        if df is None:
            raise HTTPException(status_code=404, detail="Dataset not found")

        # Load additional data (insights, models, etc.)
        additional_data = {
            "include_charts": request.include_charts,
            "include_insights": request.include_insights,
            "include_models": request.include_models,
        }

        # Generate report
        report_data = await report_generator.generate_report(
            df,
            request.report_type,
            request.dataset_id,
            additional_data
        )

        # Generate download URL
        download_url = f"/reports/{report_data['filename']}"

        logger.info(f"Report generated for dataset: {request.dataset_id}")

        return ReportResponse(
            dataset_id=request.dataset_id,
            report_id=report_data['report_id'],
            report_type=request.report_type,
            download_url=download_url,
            generated_at=datetime.now()
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating report: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")


@router.get("/templates")
async def get_report_templates():
    """Get available report templates"""
    return {
        "available_templates": report_generator.get_available_templates(),
        "customization_options": report_generator.get_customization_options()
    }