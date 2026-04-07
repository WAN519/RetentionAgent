"""
agents/emotion/emotion_agent.py

Emotion Orchestrator — coordinates Glassdoor sentiment analysis and generates
an AI-powered attrition risk report using the Gemini LLM.

Workflow:
  1. Check MongoDB for an existing analysis for the target company in the current month.
  2. If no cached result exists, load the raw Glassdoor CSV, filter by company,
     and run the GlassdoorEmotionAgent NLP pipeline to produce per-review labels.
  3. Aggregate the labeled reviews into summary statistics via a MongoDB aggregation.
  4. Pass the statistics to Gemini Pro, which returns a structured JSON report with
     attrition drivers, suggested HR actions, and an overall risk level.

MongoDB collections used:
    reviews_analysis  — per-review NLP output written by GlassdoorEmotionAgent
"""

import pandas as pd
from datetime import datetime
from tools.mongoDB import MarketDB
import google.generativeai as genai


class EmotionOrchestrator:
    """
    Coordinates the Glassdoor emotion analysis pipeline and LLM report generation.

    Attributes:
        agent (GlassdoorEmotionAgent): Configured NLP agent for processing reviews.
        db (MarketDB): MongoDB connection for checking existing analysis records.
        llm: Gemini generative model instance for producing the final HR report.
    """

    def __init__(self, agent_instance):
        """
        Args:
            agent_instance (GlassdoorEmotionAgent): An initialized emotion agent.
                The agent's MongoDB collection must match the database this
                orchestrator queries for cached results.
        """
        self.agent = agent_instance
        self.db = MarketDB()

        # Configure Gemini — API key must be set in config.env as GEMINI_API_KEY
        import os
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        self.llm = genai.GenerativeModel('gemini-1.5-pro')

    def check_and_analyze(self, company_name: str, raw_data_path: str) -> str:
        """
        Run the full analysis for a company, using cached results when available.

        Args:
            company_name (str): Company name as it appears in the Glassdoor CSV 'firm' column.
            raw_data_path (str): Path to the Glassdoor reviews CSV file.

        Returns:
            str: Gemini-generated JSON report containing attrition analysis,
                 suggested actions, and risk level. Returns an error message string
                 if no data is available.
        """
        current_month = datetime.now().strftime("%Y-%m")

        # Check if this company's reviews have already been analyzed this month
        existing_count = self.db.collection.count_documents({
            "firm": company_name,
            "analysis_month": current_month
        })

        if existing_count > 0:
            print(f"[Cache Hit] Found {existing_count} records for {company_name} ({current_month})")
        else:
            print(f"[Cache Miss] No records found — starting NLP pipeline for {company_name}")

            df = pd.read_csv(raw_data_path)
            company_df = df[df['firm'] == company_name].copy()

            # Tag each record with the analysis month before inserting into MongoDB
            company_df['analysis_month'] = current_month
            self.agent.run_pipeline(company_df)

        # Aggregate the labeled reviews into summary statistics
        stats = self._get_aggregated_stats(company_name, current_month)

        # Generate natural-language HR report from the aggregated statistics
        return self._generate_gemini_report(company_name, stats)

    def _get_aggregated_stats(self, company_name: str, month: str) -> dict | None:
        """
        Aggregate per-review NLP labels into company-level summary metrics.

        Args:
            company_name (str): Target company name.
            month (str): Analysis month in 'YYYY-MM' format.

        Returns:
            dict | None: Aggregated statistics document, or None if no records found.
                Keys: avg_sentiment, mgmt_issues, salary_issues, workload_issues, total_count.
        """
        pipeline = [
            {"$match": {"firm": company_name, "analysis_month": month}},
            {"$group": {
                "_id": "$firm",
                "avg_sentiment":    {"$avg": "$roberta_score"},
                "mgmt_issues":      {"$sum": "$sem_management"},
                "salary_issues":    {"$sum": "$sem_salary"},
                "workload_issues":  {"$sum": "$sem_workload"},
                "total_count":      {"$count": {}}
            }}
        ]
        result = list(self.db.collection.aggregate(pipeline))
        return result[0] if result else None

    def _generate_gemini_report(self, company_name: str, stats: dict | None) -> str:
        """
        Prompt Gemini Pro to generate a structured HR attrition risk report.

        The prompt includes percentage breakdowns of each topic category and
        asks Gemini to return a JSON object with three fields:
          - attrition_analysis: root cause narrative
          - suggested_actions: list of targeted HR interventions
          - risk_level: High / Medium / Low

        Args:
            company_name (str): Company name included in the prompt for context.
            stats (dict | None): Aggregated statistics from `_get_aggregated_stats`.

        Returns:
            str: Raw Gemini response text (expected to be valid JSON).
        """
        if not stats:
            return "No data available for this company and month."

        total = stats['total_count']
        prompt = f"""
        You are a senior HR consultant. Based on the following AI-analyzed employee review data
        for {company_name}, provide a detailed attrition risk analysis and recommended actions.

        Data Summary:
        - Total reviews analyzed: {total}
        - Average negative sentiment intensity: {stats['avg_sentiment']:.2f}
        - Reviews flagging management issues: {stats['mgmt_issues'] / total:.1%}
        - Reviews flagging compensation issues: {stats['salary_issues'] / total:.1%}
        - Reviews flagging workload/burnout issues: {stats['workload_issues'] / total:.1%}

        Respond in JSON with exactly these fields:
        1. "attrition_analysis": detailed analysis of the primary attrition drivers
        2. "suggested_actions": list of targeted improvement actions
        3. "risk_level": one of "High", "Medium", or "Low"
        """

        response = self.llm.generate_content(prompt)
        return response.text