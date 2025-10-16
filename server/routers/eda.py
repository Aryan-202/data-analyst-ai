from fastapi import APIRouter, HTTPException, BackgroundTasks
from server.models.dataset_schema import EDARequest, EDAResponse
from server.services.eda_engine import EDAEngine
from server.services.data_loader import DataLoader
from server.utils.logger import setup_logger

router = APIRouter()
logger = setup_logger()
eda_engine = EDAEngine()
data_loader = DataLoader()

@router.post("/analyze", response_model=EDAResponse)
async def perform_eda(
    background_tasks: BackgroundTasks,
    request: EDARequest
):
    """
    Perform Exploratory Data Analysis on a dataset
    """
    try:
        # Load dataset
        df = data_loader.load_dataset_by_id(request.dataset_id)
        if df is None:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        # Perform EDA
        eda_results = await eda_engine.analyze_dataset(
            df, 
            request.analyze_correlations, 
            request.detect_outliers,
            request.generate_summary
        )
        
        # Store EDA results (in background)
        background_tasks.add_task(
            eda_engine.store_eda_results,
            request.dataset_id, eda_results
        )
        
        logger.info(f"EDA completed for dataset: {request.dataset_id}")
        
        return EDAResponse(
            dataset_id=request.dataset_id,
            summary_stats=eda_results.get("summary_stats", {}),
            correlations=eda_results.get("correlations", {}),
            outliers_report=eda_results.get("outliers_report", {}),
            data_quality=eda_results.get("data_quality", {}),
            suggestions=eda_results.get("suggestions", [])
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error performing EDA: {str(e)}")
        raise HTTPException(status_code=500, detail=f"EDA failed: {str(e)}")

@router.get("/datasets/{dataset_id}/eda")
async def get_eda_results(dataset_id: str):
    """Get stored EDA results for a dataset"""
    try:
        eda_results = eda_engine.get_stored_eda_results(dataset_id)
        if not eda_results:
            raise HTTPException(status_code=404, detail="EDA results not found")
        return eda_results
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving EDA results: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve EDA results")