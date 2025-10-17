from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum
from datetime import datetime

class FileType(str, Enum):
    CSV = "csv"
    EXCEL = "excel"
    JSON = "json"

class ColumnType(str, Enum):
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    DATETIME = "datetime"
    TEXT = "text"
    BOOLEAN = "boolean"

class DataUploadResponse(BaseModel):
    dataset_id: str
    filename: str
    file_type: FileType
    rows: int
    columns: int
    columns_info: List[Dict[str, Any]]
    message: str

class DataCleaningRequest(BaseModel):
    dataset_id: str
    operations: List[str] = Field(..., description="List of cleaning operations to perform")
    options: Optional[Dict[str, Any]] = None

class DataCleaningResponse(BaseModel):
    dataset_id: str
    cleaned_dataset_id: str
    operations_performed: List[str]
    rows_before: int
    rows_after: int
    cleaning_report: Dict[str, Any]

class EDARequest(BaseModel):
    dataset_id: str
    analyze_correlations: bool = True
    detect_outliers: bool = True
    generate_summary: bool = True

class EDAResponse(BaseModel):
    dataset_id: str
    summary_stats: Dict[str, Any]
    correlations: Optional[Dict[str, Any]] = None
    outliers_report: Optional[Dict[str, Any]] = None
    data_quality: Dict[str, Any]
    suggestions: List[str]

class VisualizationRequest(BaseModel):
    dataset_id: str
    chart_type: Optional[str] = None  # If None, auto-generate
    x_axis: Optional[str] = None
    y_axis: Optional[str] = None
    color_by: Optional[str] = None

class VisualizationResponse(BaseModel):
    dataset_id: str
    charts: List[Dict[str, Any]]
    insights: List[str]

class InsightRequest(BaseModel):
    dataset_id: str
    analysis_type: str = "comprehensive"
    focus_areas: Optional[List[str]] = None

class InsightResponse(BaseModel):
    dataset_id: str
    insights: List[str]
    key_findings: List[str]
    recommendations: List[str]
    generated_at: datetime

class ModelingRequest(BaseModel):
    dataset_id: str
    target_column: str
    problem_type: str  # classification, regression, clustering
    model_type: Optional[str] = "auto"
    test_size: float = 0.2

class ModelingResponse(BaseModel):
    dataset_id: str
    model_id: str
    problem_type: str
    target_column: str
    model_performance: Dict[str, Any]
    feature_importance: Optional[Dict[str, float]] = None
    predictions_sample: List[Any]

class ReportRequest(BaseModel):
    dataset_id: str
    report_type: str  # pdf, excel, ppt
    include_charts: bool = True
    include_insights: bool = True
    include_models: bool = False

class ReportResponse(BaseModel):
    dataset_id: str
    report_id: str
    report_type: str
    download_url: str
    generated_at: datetime