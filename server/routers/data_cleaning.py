from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Dict, Any, Optional
import traceback
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
        print(f"Starting data cleaning for file_id: {options.file_id}")

        file_manager = FileManager(settings)
        data_cleaner = DataCleaner()

        # Load original data
        file_path = file_manager.get_file_path(options.file_id)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")

        print(f"Found file at: {file_path}")

        # Perform cleaning
        cleaning_result = await data_cleaner.clean_dataset(
            file_path,
            options.dict()
        )

        print(
            f"Data cleaning completed. Original shape: {cleaning_result['report']['original_shape']}, Final shape: {cleaning_result['report']['final_shape']}")

        # Save cleaned data
        cleaned_file_id = file_manager.generate_file_id()
        cleaned_file_path = file_manager.get_processed_path(cleaned_file_id)

        cleaning_result['cleaned_data'].to_csv(cleaned_file_path, index=False)

        print(f"Saved cleaned data to: {cleaned_file_path}")

        return CleaningResponse(
            success=True,
            message="Data cleaning completed successfully",
            cleaned_file_id=cleaned_file_id,
            cleaning_report=cleaning_result['report']
        )

    except Exception as e:
        print(f"Data cleaning failed: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
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
        print(f"Getting cleaning report for file_id: {file_id}")

        file_manager = FileManager(settings)
        data_cleaner = DataCleaner()

        file_path = file_manager.get_file_path(file_id)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")

        quality_report = await data_cleaner.get_data_quality_report(file_path)

        print("Quality report generated successfully")

        return {
            "file_id": file_id,
            "quality_report": quality_report
        }

    except Exception as e:
        print(f"Cleaning report error: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))