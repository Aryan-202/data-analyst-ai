from fastapi import APIRouter, HTTPException, BackgroundTasks
from datetime import datetime
from server.models.dataset_schema import InsightRequest, InsightResponse
from server.services.insight_generator import InsightGenerator
from server.services.data_loader import DataLoader
from server.utils.logger import setup_logger

router = APIRouter()
logger = setup_logger()
insight_generator = InsightGenerator()
data_loader = DataLoader()


@router.post("/generate", response_model=InsightResponse)
async def generate_insights(
        background_tasks: BackgroundTasks,
        request: InsightRequest
):
    """
    Generate AI-powered insights from dataset
    """
    try:
        # Load dataset
        df = data_loader.load_dataset_by_id(request.dataset_id)
        if df is None:
            raise HTTPException(status_code=404, detail="Dataset not found")

        # Load EDA results if available
        eda_results = None
        # You might want to check if EDA was already performed

        # Generate insights
        insights_data = await insight_generator.generate_comprehensive_insights(
            df,
            request.analysis_type,
            request.focus_areas or [],
            eda_results
        )

        # Store insights (in background)
        background_tasks.add_task(
            insight_generator.store_insights,
            request.dataset_id, insights_data
        )

        logger.info(f"Insights generated for dataset: {request.dataset_id}")

        return InsightResponse(
            dataset_id=request.dataset_id,
            insights=insights_data.get("insights", []),
            key_findings=insights_data.get("key_findings", []),
            recommendations=insights_data.get("recommendations", []),
            generated_at=datetime.now()
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating insights: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Insight generation failed: {str(e)}")


@router.post("/chat")
async def chat_with_data(dataset_id: str, question: str):
    """
    Chat interface for asking questions about the data
    """
    try:
        # Load dataset
        df = data_loader.load_dataset_by_id(dataset_id)
        if df is None:
            raise HTTPException(status_code=404, detail="Dataset not found")

        # Generate answer using AI
        answer = await insight_generator.answer_question(df, question)

        return {
            "dataset_id": dataset_id,
            "question": question,
            "answer": answer,
            "timestamp": datetime.now()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in chat with data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to answer question: {str(e)}")