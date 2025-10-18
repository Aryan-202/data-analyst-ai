import datetime

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Dict, Any, List
import traceback
import os

from services.eda_engine import EDAEngine
from services.file_manager import FileManager
from config import get_settings, Settings
from utils import logger
from services import data_loader
from services import eda_engine

router = APIRouter()


class EDARequest(BaseModel):
    file_id: str
    analysis_types: List[str] = ["summary", "correlation", "outliers"]


class EDAResponse(BaseModel):
    success: bool
    message: str
    analysis_results: Dict[str, Any]


@router.post("/analyze")
async def perform_eda(request: EDARequest):
    """
    Perform Exploratory Data Analysis on a dataset
    """
    try:
        logger.info(f"Starting EDA for file_id: {request.file_id}")
        logger.info(f"Analysis types requested: {request.analysis_types}")

        # Load dataset
        df = await data_loader.load_dataset_by_id(request.file_id)
        if df is None or df.empty:
            raise HTTPException(status_code=404, detail="Dataset not found or empty")

        # Validate dataset has data
        if len(df) == 0:
            raise HTTPException(status_code=400, detail="Dataset is empty")

        # Perform EDA with error handling
        analysis_results = await eda_engine.analyze_dataset(
            df, request.analysis_types
        )

        # Check if any critical analyses failed
        failed_analyses = {
            analysis: result for analysis, result in analysis_results.items()
            if 'error' in result
        }

        if failed_analyses and len(failed_analyses) == len(request.analysis_types):
            raise HTTPException(
                status_code=500,
                detail=f"All analyses failed: {failed_analyses}"
            )

        logger.info(f"EDA completed for file_id: {request.file_id}")

        return {
            "file_id": request.file_id,
            "analysis_results": analysis_results,
            "warnings": list(failed_analyses.keys()) if failed_analyses else [],
            "timestamp": datetime.now().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"EDA failed: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"EDA processing failed: {str(e)}"
        )

@router.get("/summary/{file_id}")
async def get_dataset_summary(
        file_id: str,
        settings: Settings = Depends(get_settings)
):
    """
    Get basic dataset summary
    """
    try:
        print(f"Getting summary for file_id: {file_id}")

        file_manager = FileManager(settings)
        eda_engine = EDAEngine()

        file_path = file_manager.get_file_path(file_id)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")

        print(f"Found file at: {file_path}")

        summary = await eda_engine.get_basic_summary(file_path)

        print("Summary generated successfully")

        return {
            "file_id": file_id,
            "summary": summary
        }

    except Exception as e:
        print(f"Summary error: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))