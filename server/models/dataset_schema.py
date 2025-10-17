from pydantic import BaseModel
from typing import Dict, List, Optional, Any
from datetime import datetime

class DatasetInfo(BaseModel):
    file_id: str
    filename: str
    file_path: str
    file_size: int
    rows: int
    columns: int
    column_names: List[str]
    data_types: Dict[str, str]
    uploaded_at: Optional[datetime] = None

class UploadResponse(BaseModel):
    success: bool
    message: str
    dataset_info: DatasetInfo

class DataQualityReport(BaseModel):
    completeness: Dict[str, float]
    uniqueness: Dict[str, Any]
    validity: Dict[str, Any]
    consistency: Dict[str, Any]

class EDAResult(BaseModel):
    summary: Dict[str, Any]
    correlations: Optional[Dict[str, Any]] = None
    distributions: Optional[Dict[str, Any]] = None
    outliers: Optional[Dict[str, Any]] = None

class InsightResult(BaseModel):
    summary: str
    key_findings: List[str]
    recommendations: List[str]
    alerts: List[str]

class ModelInfo(BaseModel):
    model_id: str
    algorithm: str
    problem_type: str
    target_column: str
    performance: Dict[str, float]
    feature_importance: Optional[Dict[str, Any]] = None

class ReportInfo(BaseModel):
    report_id: str
    report_type: str
    generated_at: datetime
    download_url: str