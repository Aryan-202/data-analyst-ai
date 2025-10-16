from fastapi import APIRouter, HTTPException
from server.models.dataset_schema import VisualizationRequest, VisualizationResponse
from server.services.visualization_engine import VisualizationEngine
from server.services.data_loader import DataLoader
from server.utils.logger import setup_logger

router = APIRouter()
logger = setup_logger()
viz_engine = VisualizationEngine()
data_loader = DataLoader()

@router.post("/generate", response_model=VisualizationResponse)
async def generate_visualizations(request: VisualizationRequest):
    """
    Generate visualizations for a dataset
    """
    try:
        # Load dataset
        df = data_loader.load_dataset_by_id(request.dataset_id)
        if df is None:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        # Generate visualizations
        if request.chart_type:
            # Specific chart request
            charts = await viz_engine.generate_specific_chart(
                df, 
                request.chart_type, 
                request.x_axis, 
                request.y_axis, 
                request.color_by
            )
        else:
            # Auto-generate charts
            charts = await viz_engine.auto_generate_charts(df)
        
        # Extract insights from charts
        insights = viz_engine.extract_insights_from_charts(charts, df)
        
        logger.info(f"Generated {len(charts)} visualizations for dataset: {request.dataset_id}")
        
        return VisualizationResponse(
            dataset_id=request.dataset_id,
            charts=charts,
            insights=insights
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating visualizations: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Visualization generation failed: {str(e)}")

@router.get("/chart-types")
async def get_available_chart_types():
    """Get available chart types and their requirements"""
    return {
        "available_chart_types": viz_engine.get_available_chart_types(),
        "auto_chart_selection_criteria": viz_engine.get_chart_selection_criteria()
    }