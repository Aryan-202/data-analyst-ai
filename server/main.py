from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import os
from dotenv import load_dotenv

from config import get_settings, Settings
from routers import (
    data_upload,
    data_cleaning,
    eda,
    visualization,
    insights,
    modeling,
    reports
)
from utils.logger import setup_logger

# Add this after the imports in main.py
print("🚀 Starting server with AI detection...")

# Test the insight generator directly
try:
    from services.insight_generator import InsightGenerator
    from config import get_settings

    settings = get_settings()
    insight_gen = InsightGenerator(settings)
    print(f"🤖 AI Provider Status: {insight_gen.ai_provider}")
    print(f"🤖 AI Enabled: {insight_gen.ai_enabled}")
except Exception as e:
    print(f"❌ Error creating InsightGenerator: {e}")

# Load environment variables
load_dotenv()

# Setup logger
logger = setup_logger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting Data Analyst AI Server")
    yield
    # Shutdown
    logger.info("Shutting down Data Analyst AI Server")

app = FastAPI(
    title="Data Analyst AI API",
    description="Automated data analysis and insights platform",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(data_upload.router, prefix="/api/v1", tags=["Data Upload"])
app.include_router(data_cleaning.router, prefix="/api/v1", tags=["Data Cleaning"])
app.include_router(eda.router, prefix="/api/v1", tags=["EDA"])
app.include_router(visualization.router, prefix="/api/v1", tags=["Visualization"])
app.include_router(insights.router, prefix="/api/v1", tags=["Insights"])
app.include_router(modeling.router, prefix="/api/v1", tags=["Modeling"])
app.include_router(reports.router, prefix="/api/v1", tags=["Reports"])

# Mount static files for reports
os.makedirs("data/reports", exist_ok=True)
app.mount("/reports", StaticFiles(directory="data/reports"), name="reports")

@app.get("/")
async def root():
    return {"message": "Data Analyst AI API", "status": "running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "Data Analyst AI API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)