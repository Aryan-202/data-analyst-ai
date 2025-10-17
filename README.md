
# 📊 Data Analyst AI - Automated Data Analysis Platform

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

**Data Analyst AI** is an intelligent, AI-powered platform that automates 80-90% of a data analyst's workflow. From data cleaning to insight generation and predictive modeling, this system provides comprehensive data analysis capabilities through a powerful API.

## 🚀 Features

### Core Capabilities
- **📁 Smart Data Ingestion** - Support for CSV, Excel, JSON files
- **🧹 Automated Data Cleaning** - Handle missing values, outliers, duplicates
- **📊 Exploratory Data Analysis (EDA)** - Automatic statistical analysis and correlation detection
- **🤖 AI-Powered Insights** - GPT-4 powered natural language insights and recommendations
- **📈 Automatic Visualization** - Smart chart generation based on data types
- **🔮 Predictive Modeling** - AutoML for classification, regression, and clustering
- **💬 Chat with Your Data** - Natural language queries about your datasets
- **📑 Report Generation** - Export analysis to PDF, Excel, PowerPoint

### Advanced Features
- **Real-time Data Processing**
- **RESTful API**
- **Scalable Architecture**
- **File Persistence**
- **Model Storage**

## 🏗️ System Architecture

```
Frontend (React/Streamlit) ←→ FastAPI Backend ←→ AI Engine
       ↑                             ↑                 ↑
   User Interface              Business Logic     AI/ML Processing
                               Data Processing    Insight Generation
```


## 🛠️ Tech Stack

### Backend
- **Framework**: FastAPI (Python)
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: Scikit-learn
- **AI/LLM**: OpenAI GPT-4
- **Visualization**: Plotly, Matplotlib
- **File Processing**: ReportLab

### Infrastructure
- **Database**: SQLite (with PostgreSQL support)
- **File Storage**: Local File System
- **Authentication**: JWT Tokens (ready for implementation)

## 📁 Current Project Structure
```
data-analyst-ai/  
├── .gitignore  
├── [README.md](https://readme.md/)  
├── requirements.txt  
└── server/  
├── .gitignore  
├── config.py  
├── data/  
│ ├── models/ # Trained ML models (.pkl files)  
│ ├── processed/ # Cleaned datasets  
│ ├── raw/ # Original uploaded files  
│ └── reports/ # Generated reports  
├── database/  
│ ├── db_connection.py  
│ └── user_table.py  
├── main.py # FastAPI application entry point  
├── models/  
│ └── dataset_schema.py # Pydantic schemas  
├── routers/ # API endpoints  
│ ├── data_cleaning.py  
│ ├── data_upload.py  
│ ├── eda.py  
│ ├── insights.py  
│ ├── modeling.py  
│ ├── reports.py  
│ └── visualization.py  
├── services/ # Business logic  
│ ├── data_cleaner.py  
│ ├── data_loader.py  
│ ├── eda_engine.py  
│ ├── file_manager.py  
│ ├── insight_generator.py  
│ ├── model_trainer.py  
│ ├── report_generator.py  
│ └── visualization_engine.py  
└── utils/ # Helper functions  
├── file_utils.py  
├── logger.py  
└── time_utils.py
```

## 📁 Project Structure

```
data-analyst-ai/
│
├── README.md
├── .gitignore
├── requirements.txt
├── .env
│
├── data/                         # Sample & uploaded datasets
│   ├── raw/
│   ├── processed/
│   └── reports/
│
├── server/                      # FastAPI backend (core logic)
│   ├── main.py                   # Entry point (FastAPI app)
│   ├── config.py                 # Environment variables, API keys
│   ├── requirements.txt
│   │
│   ├── routers/                  # All API endpoints grouped by feature
│   │   ├── __init__.py
│   │   ├── data_upload.py        # Handles CSV/XLS upload
│   │   ├── data_cleaning.py      # Trigger cleaning pipeline
│   │   ├── eda.py                # Runs automatic EDA
│   │   ├── visualization.py      # Returns plots
│   │   ├── insights.py           # LLM-generated insights
│   │   ├── modeling.py           # Predictive ML tasks
│   │   └── reports.py            # PDF/Excel export routes
│   │
│   ├── services/                 # Business logic & AI modules
│   │   ├── __init__.py
│   │   ├── data_loader.py
│   │   ├── data_cleaner.py
│   │   ├── eda_engine.py
│   │   ├── visualization_engine.py
│   │   ├── insight_generator.py
│   │   ├── model_trainer.py
│   │   ├── report_generator.py
│   │   └── file_manager.py
│   │
│   ├── utils/                    # Helper functions
│   │   ├── __init__.py
│   │   ├── logger.py
│   │   ├── decorators.py
│   │   ├── file_utils.py
│   │   └── time_utils.py
│   │
│   ├── models/                   # Data models (schemas, Pydantic)
│   │   ├── __init__.py
│   │   ├── dataset_schema.py
│   │   ├── user_schema.py
│   │   ├── eda_schema.py
│   │   └── insight_schema.py
│   │
│   └── database/                 # (optional) Database logic
│       ├── __init__.py
│       ├── db_connection.py
│       ├── user_table.py
│       └── dataset_table.py
│
├── client/                     # UI layer
│   ├── streamlit_app.py          # MVP version (Streamlit)
│   │
│   ├── react_app/                # Advanced UI version
│   │   ├── package.json
│   │   ├── src/
│   │   │   ├── App.jsx
│   │   │   ├── index.js
│   │   │   ├── components/
│   │   │   │   ├── UploadFile.jsx
│   │   │   │   ├── Dashboard.jsx
│   │   │   │   ├── InsightsPanel.jsx
│   │   │   │   ├── ChatWithData.jsx
│   │   │   │   └── ReportViewer.jsx
│   │   │   ├── pages/
│   │   │   │   ├── Home.jsx
│   │   │   │   ├── Login.jsx
│   │   │   │   ├── Explore.jsx
│   │   │   │   └── Predict.jsx
│   │   │   ├── services/
│   │   │   │   ├── api.js
│   │   │   │   └── auth.js
│   │   │   └── styles/
│   │   │       └── main.css
│   │   └── public/
│   │       ├── index.html
│   │       └── favicon.ico
│
├── ai_engine/                    # Core analytics intelligence
│   ├── __init__.py
│   ├── preprocess.py             # Cleaning + missing value handling
│   ├── analyzer.py               # EDA + correlation analysis
│   ├── visualizer.py             # Automatic chart generation
│   ├── modeler.py                # AutoML + forecasting
│   ├── insight_agent.py          # LLM-based natural language insights
│   ├── chat_agent.py             # “Chat with your data” logic
│   ├── report_builder.py         # Summaries + chart embedding
│   └── prompt_templates/         # LLM prompt templates
│       ├── eda_summary.txt
│       ├── correlation_summary.txt
│       └── forecast_summary.txt
│
├── tests/                        # Unit + integration tests
│   ├── test_api_endpoints.py
│   ├── test_data_cleaning.py
│   ├── test_visualizations.py
│   ├── test_insight_generation.py
│   └── test_model_training.py
│
├── scripts/                      # Utility scripts for dev/deploy
│   ├── run_local.sh
│   ├── deploy_aws.sh
│   └── init_db.py
│
└── docs/                         # Documentation
    ├── architecture-diagram.png
    ├── api_docs.md
    ├── system_design.md
    ├── data_flow.md
    └── roadmap.md
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- OpenAI API key (for AI features)


### Installation

1. **Clone the repository**
```
git clone https://github.com/yourusername/data-analyst-ai.git
cd data-analyst-ai

```
2. **Backend Setup**
```
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set environment variables
echo "OPENAI_API_KEY=your-openai-api-key-here" > .env
echo "DEBUG=True" >> .env
echo "SECRET_KEY=your-secret-key-here" >> .env
```

3. **Frontend Setup (React)**
```
npm run dev
```

3. **Run the Application**
```
cd server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

4. **Run the Application**
```
# Start backend server
cd server
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Start frontend (in separate terminal)
cd client/react_app
npm start

```

## 📚 API Documentation

Once the server is running, access the interactive API documentation:

- **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
    
- **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)
    

### Key API Endpoints

|Endpoint|Method|Description|
|---|---|---|
|`/api/v1/upload`|POST|Upload dataset files|
|`/api/v1/clean`|POST|Data cleaning operations|
|`/api/v1/analyze`|POST|Perform EDA|
|`/api/v1/visualize/generate`|POST|Generate visualizations|
|`/api/v1/insights/generate`|POST|AI-powered insights|
|`/api/v1/models/train`|POST|Train ML models|
|`/api/v1/reports/generate`|POST|Generate reports|
|`/api/v1/insights/chat`|POST|Chat with your data|

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the root directory:

```
# API Configuration
DEBUG=True
SECRET_KEY=your-secret-key-here

# AI Services
OPENAI_API_KEY=your-openai-api-key
OPENAI_MODEL=gpt-4

# File Upload
MAX_FILE_SIZE=104857600  # 100MB in bytes

# Database
DATABASE_URL=sqlite:///./data_analyst_ai.db
```

### Supported File Formats

- **CSV** (.csv)
    
- **Excel** (.xlsx, .xls)
    
- **JSON** (.json)
    

## 📊 Performance

- **File Processing**: Handles datasets up to 100MB
    
- **Response Time**: < 5 seconds for most operations
    
- **Concurrent Users**: Supports 50+ simultaneous users
    
- **AI Processing**: Optimized for GPT-4 API usage
    

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](https://contributing.md/) for details.

1. Fork the repository
    
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
    
3. Commit your changes (`git commit -m 'Add amazing feature'`)
    
4. Push to the branch (`git push origin feature/amazing-feature`)
    
5. Open a Pull Request
    

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](https://license/) file for details.

## 🆘 Support

- **Documentation**: [GitHub Wiki](https://github.com/yourusername/data-analyst-ai/wiki)
    
- **Issues**: [GitHub Issues](https://github.com/yourusername/data-analyst-ai/issues)
    
- **Email**: [support@data-analyst-ai.com](https://mailto:support@data-analyst-ai.com/)
    

## 🙏 Acknowledgments

- OpenAI for GPT-4 API
    
- FastAPI team for the excellent web framework
    
- Pandas and Scikit-learn communities
    
- Plotly for interactive visualizations
    

---

**Data Analyst AI** - Making data analysis accessible to everyone! 🚀

_For more information, visit our [documentation](https://github.com/Aryan-202/data-analyst-ai/wiki) or join our [community discussions](https://github.com/Aryan-202/data-analyst-ai/discussions)._