import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from openai import OpenAI  # Updated import for v1.0+
from config import Settings
import asyncio
import json


class InsightGenerator:
    def __init__(self, settings: Settings):
        self.settings = settings
        if settings.openai_api_key:
            self.client = OpenAI(api_key=settings.openai_api_key)

    async def generate_comprehensive_insights(self, file_path: str,
                                              analysis_types: List[str],
                                              focus_areas: Optional[List[str]] = None) -> Dict[str, Any]:
        """Generate comprehensive AI-powered insights"""
        df = pd.read_csv(file_path)

        insights = {
            'summary': await self._generate_dataset_summary(df),
            'key_findings': [],
            'recommendations': [],
            'alerts': []
        }

        if 'trends' in analysis_types:
            trends = await self._analyze_trends(df)
            insights['trends'] = trends
            insights['key_findings'].extend(trends.get('key_trends', []))

        if 'correlations' in analysis_types:
            correlations = await self._analyze_correlations_insights(df)
            insights['correlations'] = correlations
            insights['key_findings'].extend(correlations.get('significant_correlations', []))

        if 'anomalies' in analysis_types:
            anomalies = await self._detect_anomalies_insights(df)
            insights['anomalies'] = anomalies
            insights['alerts'].extend(anomalies.get('critical_anomalies', []))

        # Generate AI summary
        if self.settings.openai_api_key:
            try:
                ai_summary = await self._generate_ai_summary(insights, df, focus_areas)
                insights['ai_summary'] = ai_summary
                insights['recommendations'] = ai_summary.get('recommendations', [])
            except Exception as e:
                insights['ai_summary'] = {
                    'summary': f"AI insights temporarily unavailable: {str(e)}",
                    'recommendations': []
                }

        return insights

    async def answer_question(self, file_path: str, question: str,
                              conversation_history: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """Answer natural language questions about the data"""
        if not self.settings.openai_api_key:
            return {
                'answer': 'OpenAI API key not configured. Cannot answer questions.',
                'supporting_data': None,
                'suggested_followups': []
            }

        df = pd.read_csv(file_path)

        # Analyze data to provide context
        data_context = await self._prepare_data_context(df, question)

        # Prepare conversation context
        messages = [
            {
                "role": "system",
                "content": f"""You are a data analyst AI. Answer questions about the dataset based on the following context:

Dataset Overview:
- Shape: {df.shape}
- Columns: {list(df.columns)}
- Sample data: {df.head(3).to_dict(orient='records')}

Data Analysis Context:
{data_context}

Guidelines:
- Be precise and data-driven
- Reference specific numbers and trends
- Suggest actionable insights
- If you can't answer based on data, say so
- Suggest follow-up questions"""
            }
        ]

        # Add conversation history
        if conversation_history:
            for msg in conversation_history[-6:]:  # Last 6 messages for context
                messages.append(msg)

        # Add current question
        messages.append({"role": "user", "content": question})

        try:
            # Updated OpenAI API call for v1.0+
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.chat.completions.create(
                    model=self.settings.default_llm_model,
                    messages=messages,
                    max_tokens=500,
                    temperature=0.3
                )
            )

            answer = response.choices[0].message.content

            # Extract supporting data if mentioned
            supporting_data = await self._extract_supporting_data(df, question, answer)

            # Generate follow-up questions
            suggested_followups = await self._generate_followup_questions(question, df)

            return {
                'answer': answer,
                'supporting_data': supporting_data,
                'suggested_followups': suggested_followups
            }

        except Exception as e:
            return {
                'answer': f"Error generating response: {str(e)}",
                'supporting_data': None,
                'suggested_followups': []
            }

    async def _generate_ai_summary(self, insights: Dict[str, Any],
                                   df: pd.DataFrame, focus_areas: Optional[List[str]] = None) -> Dict[str, Any]:
        """Generate AI-powered summary using OpenAI"""
        if not self.settings.openai_api_key:
            return {'summary': 'OpenAI API key not configured', 'recommendations': []}

        try:
            prompt = self._build_insight_prompt(insights, df, focus_areas)

            # Updated OpenAI API call for v1.0+
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.chat.completions.create(
                    model=self.settings.default_llm_model,
                    messages=[
                        {"role": "system",
                         "content": "You are a senior data analyst. Provide concise, actionable insights and recommendations."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=self.settings.max_insight_tokens,
                    temperature=0.7
                )
            )

            content = response.choices[0].message.content

            # Parse the response (assuming format: Summary: ... Recommendations: ...)
            summary_parts = content.split('Recommendations:')
            summary = summary_parts[0].replace('Summary:', '').strip()
            recommendations = []

            if len(summary_parts) > 1:
                recommendations = [rec.strip() for rec in summary_parts[1].split('\n') if
                                   rec.strip() and rec.strip().startswith('-')]
                recommendations = [rec[1:].strip() for rec in recommendations]

            return {
                'summary': summary,
                'recommendations': recommendations
            }

        except Exception as e:
            return {
                'summary': f"AI summary generation failed: {str(e)}",
                'recommendations': []
            }

    # ... (keep all the other methods the same as before)
    async def _generate_dataset_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate basic dataset summary"""
        return {
            'dataset_size': f"{df.shape[0]} rows × {df.shape[1]} columns",
            'memory_usage': f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB",
            'missing_values': int(df.isnull().sum().sum()),
            'missing_percentage': f"{(df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100:.1f}%",
            'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': len(df.select_dtypes(include=['object']).columns)
        }

    async def _analyze_trends(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze trends in the data"""
        trends = {
            'key_trends': [],
            'seasonal_patterns': [],
            'growth_metrics': []
        }

        numeric_cols = df.select_dtypes(include=[np.number]).columns

        # Simple trend analysis for numeric columns
        for col in numeric_cols[:5]:  # Analyze first 5 numeric columns
            data = df[col].dropna()
            if len(data) > 1:
                # Calculate simple trend
                x = np.arange(len(data))
                slope = np.polyfit(x, data.values, 1)[0]

                trend_direction = "increasing" if slope > 0 else "decreasing"
                trend_strength = "strong" if abs(slope) > data.std() else "moderate"

                trends['key_trends'].append({
                    'metric': col,
                    'direction': trend_direction,
                    'strength': trend_strength,
                    'rate_of_change': float(slope)
                })

        return trends

    async def _analyze_correlations_insights(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate correlation insights"""
        numeric_df = df.select_dtypes(include=[np.number])

        if len(numeric_df.columns) < 2:
            return {'significant_correlations': []}

        corr_matrix = numeric_df.corr()
        significant_correlations = []

        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr = corr_matrix.iloc[i, j]
                if abs(corr) > 0.7:  # Strong correlation
                    significance = "very strong" if abs(corr) > 0.9 else "strong"
                    relationship = "positive" if corr > 0 else "negative"

                    significant_correlations.append({
                        'variables': [corr_matrix.columns[i], corr_matrix.columns[j]],
                        'correlation': float(corr),
                        'significance': significance,
                        'relationship': relationship,
                        'insight': f"{significance} {relationship} relationship between {corr_matrix.columns[i]} and {corr_matrix.columns[j]}"
                    })

        return {
            'significant_correlations': significant_correlations,
            'strongest_correlation': significant_correlations[0] if significant_correlations else None
        }

    async def _detect_anomalies_insights(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect anomalies and generate insights"""
        numeric_df = df.select_dtypes(include=[np.number])
        critical_anomalies = []

        for col in numeric_df.columns:
            data = df[col].dropna()
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outliers = data[(data < lower_bound) | (data > upper_bound)]

            if len(outliers) > 0:
                outlier_percentage = (len(outliers) / len(data)) * 100
                if outlier_percentage > 5:  # More than 5% outliers
                    critical_anomalies.append({
                        'column': col,
                        'outlier_count': len(outliers),
                        'outlier_percentage': f"{outlier_percentage:.1f}%",
                        'severity': 'high' if outlier_percentage > 10 else 'medium',
                        'description': f"High number of outliers detected in {col}"
                    })

        return {
            'critical_anomalies': critical_anomalies,
            'total_anomalies_detected': len(critical_anomalies)
        }

    def _build_insight_prompt(self, insights: Dict[str, Any], df: pd.DataFrame,
                              focus_areas: Optional[List[str]]) -> str:
        """Build prompt for AI insight generation"""
        prompt = f"""
        Dataset Overview:
        - Shape: {df.shape}
        - Columns: {list(df.columns)}
        - Numeric columns: {len(df.select_dtypes(include=[np.number]).columns)}
        - Categorical columns: {len(df.select_dtypes(include=['object']).columns)}

        Key Findings:
        {json.dumps(insights.get('key_findings', []), indent=2)}

        Trends:
        {json.dumps(insights.get('trends', {}), indent=2)}

        Correlations:
        {json.dumps(insights.get('correlations', {}), indent=2)}

        Anomalies:
        {json.dumps(insights.get('anomalies', {}), indent=2)}
        """

        if focus_areas:
            prompt += f"\nFocus Areas: {', '.join(focus_areas)}"

        prompt += """
        Please provide:
        1. A concise executive summary of the most important insights
        2. 3-5 actionable recommendations based on the data

        Format your response as:
        Summary: [your summary here]

        Recommendations:
        - [recommendation 1]
        - [recommendation 2]
        - [recommendation 3]
        """

        return prompt

    async def _prepare_data_context(self, df: pd.DataFrame, question: str) -> str:
        """Prepare data context for question answering"""
        context_parts = []

        # Basic stats for numeric columns mentioned in question
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col.lower() in question.lower():
                stats = df[col].describe()
                context_parts.append(f"{col}: mean={stats['mean']:.2f}, min={stats['min']:.2f}, max={stats['max']:.2f}")

        # Value counts for categorical columns mentioned
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col.lower() in question.lower():
                top_values = df[col].value_counts().head(3)
                context_parts.append(f"{col}: top values: {dict(top_values)}")

        return "\n".join(context_parts) if context_parts else "No specific columns mentioned in question."

    async def _extract_supporting_data(self, df: pd.DataFrame, question: str, answer: str) -> Optional[Dict]:
        """Extract supporting data for the answer"""
        # Simple extraction - in real implementation, this would be more sophisticated
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            if col in answer:
                return {
                    'column': col,
                    'statistics': df[col].describe().to_dict(),
                    'sample_values': df[col].head(5).tolist()
                }

        return None

    async def _generate_followup_questions(self, question: str, df: pd.DataFrame) -> List[str]:
        """Generate relevant follow-up questions"""
        followups = []

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

        # Generic follow-ups
        followups.extend([
            "What are the main trends over time?",
            "Which factors are most correlated?",
            "Are there any outliers or anomalies?",
            "What are the key segments in the data?"
        ])

        # Context-specific follow-ups based on original question
        if "sales" in question.lower() or "revenue" in question.lower():
            followups.extend([
                "What is the sales forecast for next quarter?",
                "Which products are performing best?",
                "What is the customer acquisition cost?"
            ])

        if "customer" in question.lower():
            followups.extend([
                "What is the customer retention rate?",
                "What are the main customer segments?",
                "What factors influence customer satisfaction?"
            ])

        return followups[:5]  # Return top 5 follow-ups