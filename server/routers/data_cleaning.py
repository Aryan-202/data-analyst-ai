from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Dict, Any, Optional
import os

from services.data_cleaner import DataCleaner
from services.file_manager import FileManager
from config import get_settings, Settings

router = APIRouter()


class CleaningOptions(BaseModel):
    file_id: str
    remove_duplicates: bool = True
    handle_missing_values: str = "auto"  # auto, drop, fill_mean, fill_median, fill_mode
    fix_data_types: bool = True
    remove_outliers: bool = False
    outlier_method: str = "iqr"  # iqr, zscore


class CleaningResponse(BaseModel):
    success: bool
    message: str
    cleaned_file_id: str
    cleaning_report: Dict[str, Any]


@router.post("/clean", response_model=CleaningResponse)
async def clean_data(
        options: CleaningOptions,
        settings: Settings = Depends(get_settings)
):
    """
    Clean and preprocess dataset
    """
    try:
        file_manager = FileManager(settings)
        data_cleaner = DataCleaner()

        # Load original data
        file_path = file_manager.get_file_path(options.file_id)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")

        # Perform cleaning
        cleaning_result = await data_cleaner.clean_dataset(
            file_path,
            options.dict()
        )

        # Save cleaned data
        cleaned_file_id = file_manager.generate_file_id()
        cleaned_file_path = file_manager.get_processed_path(cleaned_file_id)

        cleaning_result['cleaned_data'].to_csv(cleaned_file_path, index=False)

        return CleaningResponse(
            success=True,
            message="Data cleaning completed successfully",
            cleaned_file_id=cleaned_file_id,
            cleaning_report=cleaning_result['report']
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Data cleaning failed: {str(e)}")


@router.get("/cleaning-report/{file_id}")
async def get_cleaning_report(
        file_id: str,
        settings: Settings = Depends(get_settings)
):
    """
    Get data quality report before cleaning
    """
    try:
        file_manager = FileManager(settings)
        data_cleaner = DataCleaner()

        file_path = file_manager.get_file_path(file_id)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")

        quality_report = await data_cleaner.get_data_quality_report(file_path)

        return {
            "file_id": file_id,
            "quality_report": quality_report
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))