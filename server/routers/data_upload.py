from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from fastapi.responses import JSONResponse
import pandas as pd
import os
from typing import Optional
import uuid

from services.data_loader import DataLoader
from services.file_manager import FileManager
from models.dataset_schema import DatasetInfo, UploadResponse
from config import get_settings, Settings

router = APIRouter()


@router.post("/upload", response_model=UploadResponse)
async def upload_file(
        file: UploadFile = File(...),
        settings: Settings = Depends(get_settings)
):
    """
    Upload CSV, Excel, or JSON files for analysis
    """
    try:
        file_manager = FileManager(settings)
        data_loader = DataLoader()

        # Validate file type
        if not file_manager.is_valid_file_type(file.filename):
            raise HTTPException(
                status_code=400,
                detail=f"File type not supported. Allowed: {settings.allowed_file_types}"
            )

        # Generate unique file ID
        file_id = str(uuid.uuid4())

        # Save uploaded file
        file_path = await file_manager.save_uploaded_file(file, file_id)

        # Load data
        df = await data_loader.load_data(file_path)

        # Get dataset info
        dataset_info = DatasetInfo(
            file_id=file_id,
            filename=file.filename,
            file_path=file_path,
            file_size=os.path.getsize(file_path),
            rows=len(df),
            columns=len(df.columns),
            column_names=list(df.columns),
            data_types=df.dtypes.astype(str).to_dict()
        )

        return UploadResponse(
            success=True,
            message="File uploaded successfully",
            dataset_info=dataset_info
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@router.get("/datasets/{file_id}/preview")
async def preview_data(
        file_id: str,
        rows: int = 10,
        settings: Settings = Depends(get_settings)
):
    """
    Preview uploaded dataset
    """
    try:
        file_manager = FileManager(settings)
        data_loader = DataLoader()

        file_path = file_manager.get_file_path(file_id)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")

        df = await data_loader.load_data(file_path)

        return {
            "file_id": file_id,
            "preview": df.head(rows).to_dict(orient="records"),
            "columns": list(df.columns),
            "shape": {"rows": len(df), "columns": len(df.columns)}
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))