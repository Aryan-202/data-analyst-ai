import os
import uuid
import aiofiles
from fastapi import UploadFile
from typing import Optional
from config import Settings


class FileManager:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.ensure_directories()

    def ensure_directories(self):
        """Ensure all required directories exist"""
        os.makedirs(self.settings.upload_folder, exist_ok=True)
        os.makedirs(self.settings.processed_folder, exist_ok=True)
        os.makedirs(self.settings.reports_folder, exist_ok=True)

    def is_valid_file_type(self, filename: str) -> bool:
        """Check if file type is supported"""
        file_ext = filename.split('.')[-1].lower()
        return file_ext in self.settings.allowed_file_types

    async def save_uploaded_file(self, file: UploadFile, file_id: str) -> str:
        """Save uploaded file and return file path"""
        file_ext = file.filename.split('.')[-1]
        filename = f"{file_id}.{file_ext}"
        file_path = os.path.join(self.settings.upload_folder, filename)

        async with aiofiles.open(file_path, 'wb') as f:
            content = await file.read()
            await f.write(content)

        return file_path

    def get_file_path(self, file_id: str) -> str:
        """Get file path by file ID"""
        for ext in self.settings.allowed_file_types:
            potential_path = os.path.join(self.settings.upload_folder, f"{file_id}.{ext}")
            if os.path.exists(potential_path):
                return potential_path

        # Also check processed folder
        for ext in self.settings.allowed_file_types:
            potential_path = os.path.join(self.settings.processed_folder, f"{file_id}.{ext}")
            if os.path.exists(potential_path):
                return potential_path

        raise FileNotFoundError(f"File with ID {file_id} not found")

    def get_processed_path(self, file_id: str, extension: str = "csv") -> str:
        """Get path for processed file"""
        return os.path.join(self.settings.processed_folder, f"{file_id}.{extension}")

    def generate_file_id(self) -> str:
        """Generate unique file ID"""
        return str(uuid.uuid4())

    def delete_file(self, file_id: str) -> bool:
        """Delete file by file ID"""
        try:
            file_path = self.get_file_path(file_id)
            os.remove(file_path)
            return True
        except:
            return False

    def get_file_info(self, file_id: str) -> Optional[dict]:
        """Get file information"""
        try:
            file_path = self.get_file_path(file_id)
            stat = os.stat(file_path)

            return {
                'file_id': file_id,
                'file_path': file_path,
                'file_size': stat.st_size,
                'created_time': stat.st_ctime,
                'modified_time': stat.st_mtime
            }
        except:
            return None

    def list_uploaded_files(self) -> list:
        """List all uploaded files"""
        files = []
        for filename in os.listdir(self.settings.upload_folder):
            if any(filename.endswith(ext) for ext in self.settings.allowed_file_types):
                file_path = os.path.join(self.settings.upload_folder, filename)
                stat = os.stat(file_path)
                file_id = filename.split('.')[0]

                files.append({
                    'file_id': file_id,
                    'filename': filename,
                    'file_size': stat.st_size,
                    'uploaded_time': stat.st_ctime
                })

        return files