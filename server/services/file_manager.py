import os
import shutil
import uuid
from fastapi import UploadFile, HTTPException
from typing import Optional
import pandas as pd
from server.config import settings
from server.utils.logger import setup_logger

logger = setup_logger()


class FileManager:
    """Manages file operations for uploaded datasets"""

    def __init__(self):
        self.upload_dir = settings.UPLOAD_DIR
        self.processed_dir = settings.PROCESSED_DIR
        self.reports_dir = settings.REPORTS_DIR
        self._ensure_directories()

    def _ensure_directories(self):
        """Create necessary directories if they don't exist"""
        os.makedirs(self.upload_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        os.makedirs(self.reports_dir, exist_ok=True)

    async def save_uploaded_file(self, file: UploadFile, dataset_id: str) -> str:
        """Save uploaded file to disk"""
        try:
            # Generate unique filename
            file_extension = os.path.splitext(file.filename)[1].lower()
            filename = f"{dataset_id}{file_extension}"
            file_path = os.path.join(self.upload_dir, filename)

            # Save file
            with open(file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)

            logger.info(f"File saved: {file_path}")
            return file_path

        except Exception as e:
            logger.error(f"Error saving file: {str(e)}")
            raise HTTPException(status_code=500, detail=f"File save failed: {str(e)}")

    def get_file_path(self, dataset_id: str, file_type: str = "raw") -> Optional[str]:
        """Get file path for a dataset"""
        if file_type == "raw":
            directory = self.upload_dir
        elif file_type == "processed":
            directory = self.processed_dir
        else:
            directory = self.reports_dir

        # Look for any file with the dataset_id
        for file in os.listdir(directory):
            if file.startswith(dataset_id):
                return os.path.join(directory, file)
        return None

    def save_processed_data(self, df: pd.DataFrame, dataset_id: str, suffix: str = "processed") -> str:
        """Save processed DataFrame to CSV"""
        try:
            filename = f"{dataset_id}_{suffix}.csv"
            file_path = os.path.join(self.processed_dir, filename)
            df.to_csv(file_path, index=False)
            return file_path
        except Exception as e:
            logger.error(f"Error saving processed data: {str(e)}")
            raise

    def delete_file(self, file_path: str):
        """Delete a file"""
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"File deleted: {file_path}")
        except Exception as e:
            logger.error(f"Error deleting file: {str(e)}")

    def cleanup_old_files(self, max_age_hours: int = 24):
        """Clean up files older than specified hours"""
        # Implementation for cleaning old files
        pass