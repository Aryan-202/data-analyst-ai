from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import Response
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import json
import os

from services.visualization_engine import VisualizationEngine
from services.file_manager import FileManager
from config import get_settings, Settings

router = APIRouter()


class VisualizationRequest(BaseModel):
    file_id: str
    chart_type: str
    x_axis: Optional[str] = None
    y_axis: Optional[str] = None
    color_by: Optional[str] = None
    filters: Optional[Dict[str, Any]] = None


class AutoVisualizationRequest(BaseModel):
    file_id: str
    max_charts: int = 6


@router.post("/visualize")
async def create_visualization(
        request: VisualizationRequest,
        settings: Settings = Depends(get_settings)
):
    """
    Create specific visualization
    """
    try:
        file_manager = FileManager(settings)
        viz_engine = VisualizationEngine()

        file_path = file_manager.get_file_path(request.file_id)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")

        # Create visualization
        chart_data = await viz_engine.create_chart(
            file_path,
            request.chart_type,
            request.x_axis,
            request.y_axis,
            request.color_by,
            request.filters
        )

        return {
            "success": True,
            "chart_type": request.chart_type,
            "chart_data": chart_data
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Visualization failed: {str(e)}")


@router.post("/auto-visualize")
async def auto_generate_visualizations(
        request: AutoVisualizationRequest,
        settings: Settings = Depends(get_settings)
):
    """
    Automatically generate meaningful visualizations
    """
    try:
        file_manager = FileManager(settings)
        viz_engine = VisualizationEngine()

        file_path = file_manager.get_file_path(request.file_id)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")

        # Auto-generate visualizations
        charts = await viz_engine.auto_generate_charts(
            file_path,
            request.max_charts
        )

        return {
            "success": True,
            "message": f"Generated {len(charts)} visualizations",
            "charts": charts
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Auto-visualization failed: {str(e)}")


@router.get("/chart-types/{file_id}")
async def get_suggested_chart_types(
        file_id: str,
        settings: Settings = Depends(get_settings)
):
    """
    Get suggested chart types for the dataset
    """
    try:
        file_manager = FileManager(settings)
        viz_engine = VisualizationEngine()

        file_path = file_manager.get_file_path(file_id)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")

        suggestions = await viz_engine.get_chart_suggestions(file_path)

        return {
            "file_id": file_id,
            "suggested_charts": suggestions
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))