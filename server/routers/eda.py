from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Dict, Any, List

from services.eda_engine import EDAEngine
from services.file_manager import FileManager
from config import get_settings, Settings

router = APIRouter()


class EDARequest(BaseModel):
    file_id: str
    analysis_types: List[str] = ["summary", "correlation", "outliers"]


class EDAResponse(BaseModel):
    success: bool
    message: str
    analysis_results: Dict[str, Any]


@router.post("/analyze", response_model=EDAResponse)
async def perform_eda(
        request: EDARequest,
        settings: Settings = Depends(get_settings)
):
    """
    Perform Exploratory Data Analysis
    """
    try:
        file_manager = FileManager(settings)
        eda_engine = EDAEngine()

        file_path = file_manager.get_file_path(request.file_id)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")

        # Perform EDA
        analysis_results = await eda_engine.analyze_dataset(
            file_path,
            request.analysis_types
        )

        return EDAResponse(
            success=True,
            message="EDA completed successfully",
            analysis_results=analysis_results
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"EDA failed: {str(e)}")


@router.get("/summary/{file_id}")
async def get_dataset_summary(
        file_id: str,
        settings: Settings = Depends(get_settings)
):
    """
    Get basic dataset summary
    """
    try:
        file_manager = FileManager(settings)
        eda_engine = EDAEngine()

        file_path = file_manager.get_file_path(file_id)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")

        summary = await eda_engine.get_basic_summary(file_path)

        return {
            "file_id": file_id,
            "summary": summary
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))