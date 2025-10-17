import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import asyncio
import json
import requests
import os
from config import Settings


class InsightGenerator:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.ai_enabled = False
        self.ai_provider = "none"
        self.model_name = "gemma:2b"  # Faster model

        print("🔍 Testing AI providers...")

        # Test available AI providers
        if self._test_ollama():
            self.ai_enabled = True
            self.ai_provider = "ollama"
            print(f"✅ Ollama AI enabled - using {self.model_name}")
        else:
            print("ℹ️  No AI service available - using basic analytics only")

    def _test_ollama(self) -> bool:
        """Test if Ollama is running locally"""
        try:
            print("🧪 Testing Ollama connection...")

            # Test with a simple request
            test_response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": "Say 'Hello' in one word",
                    "stream": False
                },
                timeout=10
            )

            print(f"Ollama response status: {test_response.status_code}")

            if test_response.status_code == 200:
                print("✅ Ollama connection successful!")
                return True
            else:
                print(f"❌ Ollama returned status: {test_response.status_code}")
                return False

        except Exception as e:
            print(f"❌ Ollama test failed: {e}")
            return False

    async def generate_comprehensive_insights(self, file_path: str,
                                              analysis_types: List[str],
                                              focus_areas: Optional[List[str]] = None) -> Dict[str, Any]:
        """Generate comprehensive AI-powered insights"""
        df = pd.read_csv(file_path)

        insights = {
            'summary': await self._generate_dataset_summary(df),
            'key_findings': [],
            'recommendations': [],
            'alerts': [],
            'ai_provider': self.ai_provider,
            'model_used': self.model_name
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

        # Generate AI summary if available
        if self.ai_enabled:
            try:
                print("🤖 Generating AI-powered insights...")
                ai_summary = await self._generate_ai_summary(insights, df, focus_areas)
                insights['ai_summary'] = ai_summary
                insights['recommendations'] = ai_summary.get('recommendations', [])
                print("✅ AI insights generated successfully")
            except Exception as e:
                print(f"❌ AI insight generation failed: {e}")
                insights['ai_summary'] = {
                    'summary': f'AI insights temporarily unavailable: {str(e)}',
                    'recommendations': self._generate_basic_recommendations(insights)
                }
        else:
            insights['ai_summary'] = {
                'summary': 'AI service not available. Using advanced analytics only.',
                'recommendations': self._generate_basic_recommendations(insights)
            }

        return insights

    async def answer_question(self, file_path: str, question: str,
                              conversation_history: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """Answer natural language questions about the data"""
        if not self.ai_enabled:
            return {
                'answer': 'AI service not available. Please install Ollama for free local AI.',
                'supporting_data': None,
                'suggested_followups': [],
                'ai_provider': 'none'
            }

        df = pd.read_csv(file_path)

        # Analyze data to provide context
        data_context = await self._prepare_data_context(df, question)

        # Prepare conversation context
        messages = [
            {
                "role": "system",
                "content": f"""You are a data analyst AI. Answer questions about this dataset:

Dataset: {df.shape[0]} rows, {df.shape[1]} columns
Columns: {', '.join(df.columns)}
Sample: {df.head(2).to_dict(orient='records')}

Context: {data_context}

Be concise and data-driven. Reference specific numbers."""
            }
        ]

        # Add conversation history
        if conversation_history:
            for msg in conversation_history[-3:]:  # Only last 3 messages
                messages.append(msg)

        # Add current question
        messages.append({"role": "user", "content": question})

        try:
            print(f"🤖 Processing question: {question}")
            answer = await self._call_ollama(messages)

            # Extract supporting data if mentioned
            supporting_data = await self._extract_supporting_data(df, question, answer)

            # Generate follow-up questions
            suggested_followups = await self._generate_followup_questions(question, df)

            print("✅ Question answered successfully")

            return {
                'answer': answer,
                'supporting_data': supporting_data,
                'suggested_followups': suggested_followups,
                'ai_provider': self.ai_provider,
                'model_used': self.model_name
            }

        except Exception as e:
            print(f"❌ Error answering question: {e}")
            return {
                'answer': f"Error generating response: {str(e)}",
                'supporting_data': None,
                'suggested_followups': [],
                'ai_provider': self.ai_provider
            }

    async def _call_ollama(self, messages: List[Dict]) -> str:
        """Call Ollama local API with optimized settings for Gemma 2B"""
        try:
            # Convert messages to prompt format for Ollama
            prompt = ""
            for msg in messages:
                if msg["role"] == "system":
                    prompt += f"System: {msg['content']}\n\n"
                elif msg["role"] == "user":
                    prompt += f"User: {msg['content']}\n\n"
                elif msg["role"] == "assistant":
                    prompt += f"Assistant: {msg['content']}\n\n"

            prompt += "Assistant: "

            print("📡 Calling Ollama API...")
            print(f"Prompt length: {len(prompt)} characters")

            # Truncate very long prompts for faster processing
            if len(prompt) > 2000:
                print("⚠️ Prompt too long, truncating...")
                prompt = prompt[:2000] + "... [truncated]"

            # Retry logic
            max_retries = 2
            for attempt in range(max_retries):
                try:
                    response = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: requests.post(
                            "http://localhost:11434/api/generate",
                            json={
                                "model": self.model_name,
                                "prompt": prompt,
                                "stream": False,
                                "options": {
                                    "temperature": 0.3,
                                    "top_p": 0.9,
                                    "num_predict": 400,  # Shorter responses for speed
                                    "top_k": 40
                                }
                            },
                            timeout=45  # 45 second timeout
                        )
                    )

                    if response.status_code == 200:
                        result = response.json()
                        response_text = result.get('response', 'No response from AI')
                        print(f"✅ Ollama response received: {len(response_text)} characters")
                        return response_text
                    else:
                        error_msg = f"Ollama API error: {response.status_code} - {response.text}"
                        print(f"❌ {error_msg}")
                        if attempt < max_retries - 1:
                            print(f"🔄 Retrying... ({attempt + 1}/{max_retries})")
                            continue
                        raise Exception(error_msg)

                except requests.exceptions.Timeout:
                    if attempt < max_retries - 1:
                        print(f"⏰ Timeout, retrying... ({attempt + 1}/{max_retries})")
                        continue
                    else:
                        print("❌ Ollama request timed out after retries")
                        return "AI response timed out. Please try a simpler question or smaller dataset."

        except Exception as e:
            print(f"❌ Ollama call failed: {e}")
            return f"AI service temporarily unavailable: {str(e)}"

    async def _generate_ai_summary(self, insights: Dict[str, Any],
                                   df: pd.DataFrame, focus_areas: Optional[List[str]] = None) -> Dict[str, Any]:
        """Generate AI-powered summary using Ollama"""
        prompt = self._build_insight_prompt(insights, df, focus_areas)

        try:
            print("🤖 Generating AI summary...")
            content = await self._call_ollama([
                {"role": "system",
                 "content": "You are a data analyst. Provide concise insights and 2-3 recommendations based on the data."},
                {"role": "user", "content": prompt}
            ])

            # Parse the response
            summary_parts = content.split('Recommendations:')
            summary = summary_parts[0].replace('Summary:', '').strip()
            recommendations = []

            if len(summary_parts) > 1:
                recommendations = [rec.strip() for rec in summary_parts[1].split('\n') if
                                   rec.strip() and rec.strip().startswith('-')]
                recommendations = [rec[1:].strip() for rec in recommendations][:3]  # Max 3 recommendations
            else:
                recommendations = self._generate_basic_recommendations(insights)

            return {
                'summary': summary,
                'recommendations': recommendations
            }

        except Exception as e:
            print(f"❌ AI summary generation failed: {e}")
            return {
                'summary': f'AI summary failed: {str(e)}',
                'recommendations': self._generate_basic_recommendations(insights)
            }

    def _build_insight_prompt(self, insights: Dict[str, Any], df: pd.DataFrame,
                              focus_areas: Optional[List[str]]) -> str:
        """Build optimized prompt for AI insight generation"""

        # Create a simplified summary of key findings
        key_findings_summary = []
        for finding in insights.get('key_findings', [])[:3]:  # Limit to 3 key findings
            if 'metric' in finding:
                key_findings_summary.append(f"{finding['metric']}: {finding.get('direction', 'unknown')} trend")
            elif 'variables' in finding:
                key_findings_summary.append(
                    f"Correlation: {finding['variables'][0]} & {finding['variables'][1]} ({finding['correlation']:.2f})")

        # Get top correlations
        top_correlations = insights.get('correlations', {}).get('significant_correlations', [])[:2]

        prompt = f"""
Analyze this sales data:

Dataset: {df.shape[0]} rows, {len(df.columns)} columns
Key metrics: {', '.join(df.select_dtypes(include=[np.number]).columns.tolist())}

Patterns found:
{chr(10).join(f"- {finding}" for finding in key_findings_summary)}

Provide:
1. Brief summary (2 sentences)
2. 2-3 actionable recommendations

Be concise and data-focused.
"""

        if focus_areas:
            prompt += f"\nFocus on: {', '.join(focus_areas)}"

        return prompt

    def _generate_basic_recommendations(self, insights: Dict[str, Any]) -> List[str]:
        """Generate basic recommendations when AI is not available"""
        recommendations = []

        # Extract insights from the data
        key_findings = insights.get('key_findings', [])
        correlations = insights.get('correlations', {}).get('significant_correlations', [])

        # Generate recommendations based on data patterns
        for finding in key_findings:
            if 'sales' in finding.get('metric', ''):
                if finding.get('direction') == 'increasing':
                    recommendations.append("Continue current sales strategies - showing positive growth")
                else:
                    recommendations.append("Review and optimize sales strategies")

        for correlation in correlations:
            if correlation.get('relationship') == 'positive' and correlation.get('significance') == 'very strong':
                vars = correlation.get('variables', [])
                if 'sales' in vars and 'customers' in vars:
                    recommendations.append("Focus on customer acquisition - strongly correlates with sales")

        # Add general recommendations
        if not recommendations:
            recommendations = [
                "Monitor key metrics regularly",
                "Segment data for deeper insights",
                "Consider optimization testing"
            ]

        return recommendations[:3]  # Max 3 recommendations

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

        for col in numeric_cols[:3]:  # Only first 3 numeric columns
            data = df[col].dropna()
            if len(data) > 1:
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
                if abs(corr) > 0.7:
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
            'significant_correlations': significant_correlations[:5],  # Limit to top 5
            'strongest_correlation': significant_correlations[0] if significant_correlations else None
        }

    async def _detect_anomalies_insights(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect anomalies and generate insights"""
        numeric_df = df.select_dtypes(include=[np.number])
        critical_anomalies = []

        for col in numeric_df.columns[:3]:  # Only first 3 numeric columns
            data = df[col].dropna()
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outliers = data[(data < lower_bound) | (data > upper_bound)]

            if len(outliers) > 0:
                outlier_percentage = (len(outliers) / len(data)) * 100
                if outlier_percentage > 5:
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

    async def _prepare_data_context(self, df: pd.DataFrame, question: str) -> str:
        """Prepare data context for question answering"""
        context_parts = []

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col.lower() in question.lower():
                stats = df[col].describe()
                context_parts.append(f"{col}: mean={stats['mean']:.2f}, range={stats['min']:.2f}-{stats['max']:.2f}")

        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col.lower() in question.lower():
                top_value = df[col].mode().iloc[0] if not df[col].mode().empty else "N/A"
                context_parts.append(f"{col}: most common='{top_value}'")

        return ". ".join(context_parts) if context_parts else "General dataset analysis"

    async def _extract_supporting_data(self, df: pd.DataFrame, question: str, answer: str) -> Optional[Dict]:
        """Extract supporting data for the answer"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            if col in answer:
                return {
                    'column': col,
                    'statistics': {
                        'mean': float(df[col].mean()),
                        'min': float(df[col].min()),
                        'max': float(df[col].max())
                    },
                    'sample_values': df[col].head(3).tolist()
                }

        return None

    async def _generate_followup_questions(self, question: str, df: pd.DataFrame) -> List[str]:
        """Generate relevant follow-up questions"""
        followups = []

        followups.extend([
            "What are the main trends?",
            "Which factors correlate most?",
            "Any outliers or anomalies?",
            "Key segments in the data?"
        ])

        return followups[:3]  # Max 3 follow-up questions