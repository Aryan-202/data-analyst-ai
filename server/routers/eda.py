from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Dict, Any, List
import traceback
import os
import pandas as pd
from datetime import datetime

from services.eda_engine import EDAEngine
from services.file_manager import FileManager
from config import get_settings, Settings
from utils.logger import setup_logger
from utils.json_encoder import safe_json_serialize_dict

router = APIRouter()
logger = setup_logger(__name__)


class EDARequest(BaseModel):
    file_id: str
    analysis_types: List[str] = ["summary", "correlation", "outliers"]


class EDAResponse(BaseModel):
    success: bool
    message: str
    analysis_results: Dict[str, Any]


@router.post("/analyze")
async def perform_eda(
    request: EDARequest,
    settings: Settings = Depends(get_settings)
):
    """
    Perform Exploratory Data Analysis on a dataset
    """
    try:
        logger.info(f"Starting EDA for file_id: {request.file_id}")
        logger.info(f"Analysis types requested: {request.analysis_types}")

        file_manager = FileManager(settings)
        eda_engine = EDAEngine()

        # Get file path
        file_path = file_manager.get_file_path(request.file_id)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")

        logger.info(f"Found file at: {file_path}")

        # Load dataset
        file_ext = file_path.split('.')[-1].lower()
        if file_ext == 'csv':
            df = pd.read_csv(file_path)
        elif file_ext in ['xlsx', 'xls']:
            df = pd.read_excel(file_path)
        elif file_ext == 'json':
            df = pd.read_json(file_path)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported file format: {file_ext}")

        # Validate dataset
        if df is None or df.empty:
            raise HTTPException(status_code=400, detail="Dataset is empty or could not be loaded")

        if len(df) == 0:
            raise HTTPException(status_code=400, detail="Dataset has no rows")

        logger.info(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

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
        analysis_results = safe_json_serialize_dict(analysis_results)

        return {
            "success": True,
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
        logger.info(f"Getting summary for file_id: {file_id}")

        file_manager = FileManager(settings)
        eda_engine = EDAEngine()

        file_path = file_manager.get_file_path(file_id)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")

        logger.info(f"Found file at: {file_path}")

        summary = await eda_engine.get_basic_summary(file_path)

        logger.info("Summary generated successfully")

        return {
            "success": True,
            "file_id": file_id,
            "summary": summary
        }

    except Exception as e:
        logger.error(f"Summary error: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/correlations/{file_id}")
async def get_correlations(
    file_id: str,
    settings: Settings = Depends(get_settings)
):
    """
    Get correlation analysis for dataset
    """
    try:
        logger.info(f"Getting correlations for file_id: {file_id}")

        file_manager = FileManager(settings)
        eda_engine = EDAEngine()

        file_path = file_manager.get_file_path(file_id)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")

        # Load data
        file_ext = file_path.split('.')[-1].lower()
        if file_ext == 'csv':
            df = pd.read_csv(file_path)
        elif file_ext in ['xlsx', 'xls']:
            df = pd.read_excel(file_path)
        elif file_ext == 'json':
            df = pd.read_json(file_path)

        # Get correlations
        correlation_results = await eda_engine._analyze_correlations(df)

        return {
            "success": True,
            "file_id": file_id,
            "correlations": correlation_results
        }

    except Exception as e:
        logger.error(f"Correlation analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/outliers/{file_id}")
async def get_outliers(
    file_id: str,
    settings: Settings = Depends(get_settings)
):
    """
    Get outlier detection for dataset
    """
    try:
        logger.info(f"Getting outliers for file_id: {file_id}")

        file_manager = FileManager(settings)
        eda_engine = EDAEngine()

        file_path = file_manager.get_file_path(file_id)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")

        # Load data
        file_ext = file_path.split('.')[-1].lower()
        if file_ext == 'csv':
            df = pd.read_csv(file_path)
        elif file_ext in ['xlsx', 'xls']:
            df = pd.read_excel(file_path)
        elif file_ext == 'json':
            df = pd.read_json(file_path)

        # Get outliers
        outlier_results = await eda_engine._detect_outliers(df)

        return {
            "success": True,
            "file_id": file_id,
            "outliers": outlier_results
        }

    except Exception as e:
        logger.error(f"Outlier detection error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/distributions/{file_id}")
async def get_distributions(
    file_id: str,
    settings: Settings = Depends(get_settings)
):
    """
    Get distribution analysis for dataset
    """
    try:
        logger.info(f"Getting distributions for file_id: {file_id}")

        file_manager = FileManager(settings)
        eda_engine = EDAEngine()

        file_path = file_manager.get_file_path(file_id)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")

        # Load data
        file_ext = file_path.split('.')[-1].lower()
        if file_ext == 'csv':
            df = pd.read_csv(file_path)
        elif file_ext in ['xlsx', 'xls']:
            df = pd.read_excel(file_path)
        elif file_ext == 'json':
            df = pd.read_json(file_path)

        # Get distributions
        distribution_results = await eda_engine._analyze_distributions(df)

        return {
            "success": True,
            "file_id": file_id,
            "distributions": distribution_results
        }

    except Exception as e:
        logger.error(f"Distribution analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    """
    Health check for EDA service
    """
    return {
        "status": "healthy",
        "service": "EDA Engine",
        "timestamp": datetime.now().isoformat()
    }