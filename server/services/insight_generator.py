import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import openai
from server.config import settings
from server.utils.logger import setup_logger

logger = setup_logger()


class InsightGenerator:
    """Generates AI-powered insights from data"""

    def __init__(self):
        self.openai_client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)
        self.stored_insights = {}

    async def generate_comprehensive_insights(self, df: pd.DataFrame, analysis_type: str,
                                              focus_areas: List[str], eda_results: Optional[Dict[str, Any]] = None) -> \
    Dict[str, Any]:
        """Generate comprehensive AI-powered insights"""

        try:
            # Prepare data summary for AI
            data_summary = self._prepare_data_summary(df, eda_results)

            # Generate insights using AI
            prompt = self._build_insight_prompt(data_summary, analysis_type, focus_areas)
            ai_response = await self._get_ai_insights(prompt)

            # Parse AI response
            insights_data = self._parse_ai_response(ai_response)

            # Add statistical insights
            statistical_insights = self._generate_statistical_insights(df, eda_results)
            insights_data['statistical_insights'] = statistical_insights

            return insights_data

        except Exception as e:
            logger.error(f"Error generating insights: {str(e)}")
            # Fallback to basic insights
            return self._generate_basic_insights(df, eda_results)

    def _prepare_data_summary(self, df: pd.DataFrame, eda_results: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Prepare data summary for AI analysis"""
        summary = {
            'dataset_shape': df.shape,
            'columns': list(df.columns),
            'data_types': {col: str(dtype) for col, dtype in df.dtypes.items()},
            'basic_stats': {}
        }

        # Add basic statistics for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            summary['basic_stats'][col] = {
                'mean': float(df[col].mean()),
                'median': float(df[col].median()),
                'std': float(df[col].std()),
                'min': float(df[col].min()),
                'max': float(df[col].max())
            }

        # Add EDA results if available
        if eda_results:
            summary['eda_results'] = {
                'correlations': eda_results.get('correlations', {}),
                'data_quality': eda_results.get('data_quality', {}),
                'outliers': eda_results.get('outliers_report', {})
            }

        return summary

    def _build_insight_prompt(self, data_summary: Dict[str, Any], analysis_type: str, focus_areas: List[str]) -> str:
        """Build prompt for AI insight generation"""

        prompt = f"""
        You are a senior data analyst. Analyze the following dataset and provide comprehensive insights.

        DATASET SUMMARY:
        - Shape: {data_summary['dataset_shape']}
        - Columns: {', '.join(data_summary['columns'])}
        - Data Types: {data_summary['data_types']}

        BASIC STATISTICS:
        {data_summary['basic_stats']}

        ANALYSIS TYPE: {analysis_type}
        FOCUS AREAS: {', '.join(focus_areas) if focus_areas else 'General analysis'}

        Please provide:
        1. 3-5 key insights about the data
        2. 3-5 important findings that stand out
        3. 3-5 actionable recommendations

        Format your response as JSON:
        {{
            "insights": ["insight1", "insight2", ...],
            "key_findings": ["finding1", "finding2", ...],
            "recommendations": ["recommendation1", "recommendation2", ...]
        }}
        """

        return prompt

    async def _get_ai_insights(self, prompt: str) -> str:
        """Get insights from AI model"""
        try:
            if not settings.OPENAI_API_KEY:
                raise ValueError("OpenAI API key not configured")

            response = self.openai_client.chat.completions.create(
                model=settings.OPENAI_MODEL,
                messages=[
                    {"role": "system",
                     "content": "You are a expert data analyst. Provide clear, actionable insights based on the data."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1500
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"AI insight generation failed: {str(e)}")
            raise

    def _parse_ai_response(self, ai_response: str) -> Dict[str, Any]:
        """Parse AI response into structured format"""
        try:
            # Try to extract JSON from response
            import json
            import re

            # Find JSON in the response
            json_match = re.search(r'\{.*\}', ai_response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                # Fallback: basic parsing
                return {
                    "insights": ["AI insights generated successfully"],
                    "key_findings": ["Review the data for patterns and anomalies"],
                    "recommendations": ["Consider further analysis based on business context"]
                }

        except Exception as e:
            logger.error(f"Error parsing AI response: {str(e)}")
            return {
                "insights": ["Failed to generate AI insights"],
                "key_findings": [],
                "recommendations": []
            }

    def _generate_statistical_insights(self, df: pd.DataFrame, eda_results: Optional[Dict[str, Any]]) -> List[str]:
        """Generate statistical insights from data"""
        insights = []
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        # Basic statistical insights
        if len(numeric_cols) > 0:
            for col in numeric_cols[:3]:  # First 3 numeric columns
                skewness = df[col].skew()
                if abs(skewness) > 1:
                    direction = "right" if skewness > 0 else "left"
                    insights.append(f"Column '{col}' is highly skewed ({skewness:.2f}) to the {direction}")

        # Correlation insights
        if eda_results and 'correlations' in eda_results:
            strong_corrs = eda_results['correlations'].get('strong_correlations', [])
            for corr in strong_corrs[:2]:
                insights.append(
                    f"Strong correlation between {corr['column1']} and {corr['column2']} ({corr['correlation']})")

        # Data quality insights
        if eda_results and 'data_quality' in eda_results:
            quality = eda_results['data_quality']
            if quality.get('completeness_score', 100) < 95:
                insights.append(f"Data completeness: {quality['completeness_score']}% - consider data cleaning")

        return insights

    def _generate_basic_insights(self, df: pd.DataFrame, eda_results: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate basic insights as fallback"""
        insights = []
        key_findings = []
        recommendations = []

        # Basic insights
        insights.append(f"Dataset contains {df.shape[0]} rows and {df.shape[1]} columns")
        insights.append(f"Numeric columns: {len(df.select_dtypes(include=[np.number]).columns)}")
        insights.append(f"Categorical columns: {len(df.select_dtypes(include=['object']).columns)}")

        # Key findings
        if not df.isnull().sum().sum() == 0:
            key_findings.append("Dataset contains missing values that need handling")

        if df.duplicated().sum() > 0:
            key_findings.append(f"Found {df.duplicated().sum()} duplicate rows")

        # Recommendations
        recommendations.append("Perform data cleaning to handle missing values and duplicates")
        recommendations.append("Explore relationships between variables using correlation analysis")
        recommendations.append("Consider feature engineering for better model performance")

        return {
            "insights": insights,
            "key_findings": key_findings,
            "recommendations": recommendations
        }

    async def answer_question(self, df: pd.DataFrame, question: str) -> str:
        """Answer specific questions about the data"""
        try:
            # Prepare data context
            data_context = self._prepare_data_summary(df, None)

            prompt = f"""
            Based on the following dataset information, answer the user's question.

            DATASET CONTEXT:
            {data_context}

            USER QUESTION: {question}

            Provide a concise, data-driven answer. If the question cannot be answered with the available data, explain why.
            """

            response = self.openai_client.chat.completions.create(
                model=settings.OPENAI_MODEL,
                messages=[
                    {"role": "system",
                     "content": "You are a helpful data analyst. Answer questions based on the provided data context."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Error answering question: {str(e)}")
            return "I apologize, but I encountered an error while processing your question. Please try again."

    def store_insights(self, dataset_id: str, insights_data: Dict[str, Any]):
        """Store insights for later retrieval"""
        self.stored_insights[dataset_id] = insights_data

    def get_stored_insights(self, dataset_id: str) -> Dict[str, Any]:
        """Get stored insights"""
        return self.stored_insights.get(dataset_id, {})