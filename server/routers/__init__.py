# server/routers/__init__.py
from .data_upload import router as data_upload_router
from .data_cleaning import router as data_cleaning_router
from .eda import router as eda_router
from .visualization import router as visualization_router
from .insights import router as insights_router
from .modeling import router as modeling_router
from .reports import router as reports_router

__all__ = [
    "data_upload_router",
    "data_cleaning_router",
    "eda_router", 
    "visualization_router",
    "insights_router",
    "modeling_router",
    "reports_router"
]