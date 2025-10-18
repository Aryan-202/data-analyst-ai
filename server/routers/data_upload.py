from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from fastapi.responses import JSONResponse
import pandas as pd
import os
from typing import Optional
import uuid
import traceback
import numpy as np

from config import get_settings, Settings
from utils.json_utils import dataframe_to_json_safe

router = APIRouter()


@router.post("/upload")
async def upload_file(
        file: UploadFile = File(...),
        settings: Settings = Depends(get_settings)
):
    """
    Upload CSV, Excel, or JSON files for analysis
    """
    try:
        print(f"Starting upload for file: {file.filename}")

        # Validate file type - FIXED: Use the property from settings
        file_ext = file.filename.split('.')[-1].lower() if '.' in file.filename else ""

        print(f"File extension: {file_ext}")
        print(f"Allowed file types: {settings.allowed_file_types}")

        if file_ext not in settings.allowed_file_types:
            raise HTTPException(
                status_code=400,
                detail=f"File type '{file_ext}' not supported. Allowed: {settings.allowed_file_types}"
            )

        # Generate unique file ID
        file_id = str(uuid.uuid4())
        filename = f"{file_id}.{file_ext}"
        file_path = os.path.join(settings.upload_folder, filename)

        # Ensure directory exists
        os.makedirs(settings.upload_folder, exist_ok=True)

        print(f"Saving file to: {file_path}")

        # Save uploaded file
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)

        print(f"File saved successfully. Size: {len(content)} bytes")

        # Load data with pandas
        print("Loading data with pandas...")
        try:
            if file_ext == 'csv':
                df = pd.read_csv(file_path)
            elif file_ext in ['xlsx', 'xls']:
                df = pd.read_excel(file_path)
            elif file_ext == 'json':
                df = pd.read_json(file_path)

            print(f"Data loaded successfully: {df.shape}")

        except Exception as pandas_error:
            print(f"Pandas error: {str(pandas_error)}")
            # Clean up the invalid file
            if os.path.exists(file_path):
                os.remove(file_path)
            raise HTTPException(
                status_code=400,
                detail=f"Error reading file: {str(pandas_error)}"
            )

        # Get dataset info
        dataset_info = {
            "file_id": file_id,
            "filename": file.filename,
            "file_path": file_path,
            "file_size": len(content),
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": list(df.columns),
            "data_types": df.dtypes.astype(str).to_dict()
        }

        print("Upload completed successfully")

        return {
            "success": True,
            "message": "File uploaded successfully",
            "dataset_info": dataset_info
        }

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        print(f"Unexpected error in upload: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"Upload failed: {str(e)}"
        )


@router.get("/datasets/{file_id}/preview")
async def preview_data(
        file_id: str,
        rows: int = 10,
        settings: Settings = Depends(get_settings)
):
    """
    Preview uploaded dataset with proper NaN handling
    """
    try:
        print(f"Preview requested for file_id: {file_id}, rows: {rows}")

        # Look for the file in raw directory
        file_path = None
        for ext in settings.allowed_file_types:
            potential_path = os.path.join(settings.upload_folder, f"{file_id}.{ext}")
            if os.path.exists(potential_path):
                file_path = potential_path
                break

        if not file_path:
            raise HTTPException(status_code=404, detail="File not found")

        print(f"Found file at: {file_path}")

        # Load data based on file extension
        file_ext = file_path.split('.')[-1].lower()

        if file_ext == 'csv':
            df = pd.read_csv(file_path)
        elif file_ext in ['xlsx', 'xls']:
            df = pd.read_excel(file_path)
        elif file_ext == 'json':
            df = pd.read_json(file_path)

        print(f"Loaded data with shape: {df.shape}")

        # Manual cleaning function for the preview data
        def clean_preview_data(df_preview):
            """Clean DataFrame preview data for JSON serialization"""
            cleaned_records = []
            for record in df_preview.to_dict('records'):
                cleaned_record = {}
                for key, value in record.items():
                    # Handle NaN and None values
                    if pd.isna(value) or value is None:
                        cleaned_record[key] = None
                    # Handle numpy types
                    elif isinstance(value, (np.integer, np.int32, np.int64)):
                        cleaned_record[key] = int(value)
                    elif isinstance(value, (np.floating, np.float32, np.float64)):
                        if np.isnan(value) or np.isinf(value):
                            cleaned_record[key] = None
                        else:
                            cleaned_record[key] = float(value)
                    elif isinstance(value, np.bool_):
                        cleaned_record[key] = bool(value)
                    # Handle pandas Timestamp
                    elif isinstance(value, pd.Timestamp):
                        cleaned_record[key] = value.isoformat()
                    # Handle other types
                    else:
                        try:
                            # Try to convert to native Python type
                            cleaned_record[key] = value.item() if hasattr(value, 'item') else value
                        except:
                            # Fallback to string representation
                            cleaned_record[key] = str(value)
                cleaned_records.append(cleaned_record)
            return cleaned_records

        # Get and clean preview data
        preview_df = df.head(rows)
        cleaned_preview = clean_preview_data(preview_df)

        # Clean column names
        cleaned_columns = [str(col) for col in df.columns.tolist()]

        print(f"Preview data cleaned successfully. Rows: {len(cleaned_preview)}")

        return {
            "file_id": file_id,
            "preview": cleaned_preview,
            "columns": cleaned_columns,
            "shape": {"rows": len(df), "columns": len(df.columns)}
        }

    except Exception as e:
        print(f"Preview error: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Preview failed: {str(e)}")