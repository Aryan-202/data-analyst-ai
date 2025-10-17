from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
import pandas as pd
import os
import uuid
from typing import List

from config import settings
from models.dataset_schema import DataUploadResponse, FileType
from server.services.file_manager import FileManager
from server.services.data_loader import DataLoader
from server.utils.logger import setup_logger

router = APIRouter()
logger = setup_logger()
file_manager = FileManager()
data_loader = DataLoader()


@router.post("/upload", response_model=DataUploadResponse)
async def upload_dataset(
        background_tasks: BackgroundTasks,
        file: UploadFile = File(...)
):
    """
    Upload a dataset file (CSV, Excel, JSON)
    """
    try:
        # Validate file type
        file_extension = os.path.splitext(file.filename)[1].lower()
        if file_extension not in settings.ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"File type {file_extension} not supported. Allowed: {settings.ALLOWED_EXTENSIONS}"
            )

        # Generate unique dataset ID
        dataset_id = str(uuid.uuid4())

        # Save file
        file_path = await file_manager.save_uploaded_file(file, dataset_id)

        # Load and validate data
        df, file_type = await data_loader.load_dataset(file_path, file_extension)

        # Get basic info
        rows, columns = df.shape
        columns_info = data_loader.get_columns_info(df)

        # Store dataset metadata (in background)
        background_tasks.add_task(
            data_loader.store_dataset_metadata,
            dataset_id, file.filename, file_type, df
        )

        logger.info(f"Dataset uploaded: {file.filename}, ID: {dataset_id}")

        return DataUploadResponse(
            dataset_id=dataset_id,
            filename=file.filename,
            file_type=file_type,
            rows=rows,
            columns=columns,
            columns_info=columns_info,
            message="Dataset uploaded successfully"
        )

    except Exception as e:
        logger.error(f"Error uploading file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@router.get("/datasets")
async def list_datasets():
    """List all uploaded datasets"""
    try:
        datasets = data_loader.list_available_datasets()
        return {"datasets": datasets}
    except Exception as e:
        logger.error(f"Error listing datasets: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to list datasets")


@router.get("/datasets/{dataset_id}")
async def get_dataset_info(dataset_id: str):
    """Get information about a specific dataset"""
    try:
        dataset_info = data_loader.get_dataset_info(dataset_id)
        if not dataset_info:
            raise HTTPException(status_code=404, detail="Dataset not found")
        return dataset_info
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting dataset info: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get dataset info")