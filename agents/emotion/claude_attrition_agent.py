"""
agents/emotion/claude_attrition_agent.py

Agentic attrition analyzer powered by Claude Opus 4.6.

Claude autonomously calls MongoDB query tools to gather data, then synthesizes
a structured attrition risk report with root cause analysis and HR interventions.

Tools available to Claude:
  get_emotion_summary    — aggregated Glassdoor sentiment stats per company/month
  get_high_risk_reviews  — sample reviews with strongest negative sentiment
  get_retention_risks    — employee-level risk scores from Retention_Predictions
  get_equity_gaps        — salary gap data from Equity_Predictions

Usage:
  python -m agents.emotion.claude_attrition_agent
  python -m agents.emotion.claude_attrition_agent --company Apple --month 2026-04
"""

import os
import json
import argparse
from datetime import datetime
from dotenv import load_dotenv
from pymongo import MongoClient
import certifi
import anthropic

load_dotenv("config.env")

# ---------------------------------------------------------------------------
# MongoDB connection (reuse existing config)
# ---------------------------------------------------------------------------

def _get_mongo_db():
    """Return the MongoDB database handle using credentials from config.env."""
    uri = os.getenv("MONGODB_URI")
    db_name = os.getenv("MONGODB_NAME", "MarketInformation")
    client = MongoClient(uri, serverSelectionTimeoutMS=5000, tlsCAFile=certifi.where())
    return client[db_name]


# ---------------------------------------------------------------------------
# Tool implementations — these run locally and return JSON-serializable dicts
# ---------------------------------------------------------------------------

def get_emotion_summary(company_name: str, month: str) -> dict:
    """
    Aggregate Glassdoor review labels for a company in a given month.
    Returns topic issue rates and average sentiment score.
    """
    db = _get_mongo_db()
    pipeline = [
        {"$match": {"firm": company_name, "analysis_month": month}},
        {"$group": {
            "_id":              "$firm",
            "total_reviews":    {"$sum": 1},
            "avg_sentiment":    {"$avg": "$roberta_score"},
            "negative_count":   {"$sum": {"$cond": [{"$eq": ["$roberta_label", "Negative"]}, 1, 0]}},
            "mgmt_issues":      {"$sum": "$sem_management"},
            "salary_issues":    {"$sum": "$sem_salary"},
            "workload_issues":  {"$sum": "$sem_workload"},
            "career_issues":    {"$sum": "$sem_career"},
        }}
    ]
    result = list(db["reviews_analysis"].aggregate(pipeline))
    if not result:
        return {"error": f"No emotion data found for {company_name} in {month}"}

    r = result[0]
    total = r["total_reviews"]
    return {
        "company":          company_name,
        "month":            month,
        "total_reviews":    total,
        "avg_sentiment":    round(r["avg_sentiment"], 3),
        "negative_rate":    round(r["negative_count"] / total, 3),
        "mgmt_issue_rate":  round(r["mgmt_issues"] / total, 3),
        "salary_issue_rate":round(r["salary_issues"] / total, 3),
        "workload_issue_rate": round(r["workload_issues"] / total, 3),
        "career_issue_rate":round(r["career_issues"] / total, 3),
    }


def get_high_risk_reviews(company_name: str, month: str, limit: int = 8) -> dict:
    """
    Fetch the most negatively-scored reviews to give Claude concrete examples.
    Returns a list of review text snippets with their sentiment scores.
    """
    db = _get_mongo_db()
    cursor = (
        db["reviews_analysis"]
        .find(
            {"firm": company_name, "analysis_month": month, "roberta_label": "Negative"},
            {"_id": 0, "cons": 1, "roberta_score": 1, "sem_management": 1,
             "sem_salary": 1, "sem_workload": 1, "sem_career": 1}
        )
        .sort("roberta_score", -1)
        .limit(limit)
    )
    reviews = list(cursor)
    if not reviews:
        return {"error": "No negative reviews found."}

    return {"company": company_name, "month": month, "sample_reviews": reviews}


def get_retention_risks(limit: int = 15) -> dict:
    """
    Fetch the highest-risk employees from the Retention_Predictions collection.
    Provides the distribution of risk buckets and key risk signals.
    """
    db = _get_mongo_db()

    # Risk distribution summary
    pipeline = [
        {"$group": {
            "_id":        "$risk_bucket",
            "count":      {"$sum": 1},
            "avg_gap":    {"$avg": "$pay_gap"},
            "avg_months": {"$avg": "$months_since_promotion"},
        }}
    ]
    distribution = {r["_id"]: r for r in db["Retention_Predictions"].aggregate(pipeline)}

    # Top high-risk employee records
    high_risk = list(
        db["Retention_Predictions"]
        .find(
            {"risk_bucket": "High"},
            {"_id": 0, "employee_id": 1, "risk_score": 1, "risk_pct": 1,
             "pay_gap": 1, "months_since_promotion": 1, "rule_flag": 1, "risk_reasons": 1}
        )
        .sort("risk_score", -1)
        .limit(limit)
    )

    return {
        "risk_distribution": distribution,
        "high_risk_employees": high_risk,
        "total_high_risk": distribution.get("High", {}).get("count", 0),
    }


def get_equity_gaps(limit: int = 10) -> dict:
    """
    Fetch employees with the worst salary gaps from Equity_Predictions.
    External gap < -15% means significantly underpaid vs. market.
    """
    db = _get_mongo_db()
    cursor = (
        db["Equity_Predictions"]
        .find(
            {"equity_gaps.external_gap_pct": {"$lt": -10}},
            {"_id": 0, "employee_id": 1, "role_name": 1,
             "actual_salary": 1, "benchmarks": 1, "equity_gaps": 1}
        )
        .sort("equity_gaps.external_gap_pct", 1)
        .limit(limit)
    )
    underpaid = list(cursor)
    return {
        "underpaid_employees": underpaid,
        "count": len(underpaid),
        "note": "external_gap_pct < -10 means employee earns more than 10% below market"
    }


# ---------------------------------------------------------------------------
# Tool dispatcher — maps Claude's tool_use block to the right function
# ---------------------------------------------------------------------------

TOOL_MAP = {
    "get_emotion_summary":   get_emotion_summary,
    "get_high_risk_reviews": get_high_risk_reviews,
    "get_retention_risks":   get_retention_risks,
    "get_equity_gaps":       get_equity_gaps,
}

def execute_tool(name: str, inputs: dict) -> str:
    """Run the requested tool and return its result as a JSON string."""
    fn = TOOL_MAP.get(name)
    if fn is None:
        return json.dumps({"error": f"Unknown tool: {name}"})
    try:
        result = fn(**inputs)
        return json.dumps(result, default=str, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)})


# ---------------------------------------------------------------------------
# Tool schemas passed to Claude
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "name": "get_emotion_summary",
        "description": (
            "Retrieve aggregated Glassdoor employee review statistics for a company "
            "in a specific month. Returns total reviews, average sentiment score, "
            "negative review rate, and issue rates for management, salary, workload, "
            "and career growth topics."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "company_name": {"type": "string", "description": "Company name as stored in Glassdoor data (e.g. 'Apple')."},
                "month":        {"type": "string", "description": "Month in YYYY-MM format (e.g. '2026-04')."},
            },
            "required": ["company_name", "month"],
        },
    },
    {
        "name": "get_high_risk_reviews",
        "description": (
            "Fetch a sample of the most negatively-scored employee reviews for a company "
            "in a given month. Useful for understanding specific grievances behind the numbers."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "company_name": {"type": "string"},
                "month":        {"type": "string", "description": "Month in YYYY-MM format."},
                "limit":        {"type": "integer", "description": "Max reviews to return (default 8).", "default": 8},
            },
            "required": ["company_name", "month"],
        },
    },
    {
        "name": "get_retention_risks",
        "description": (
            "Retrieve employee-level attrition risk scores from the Cox survival model. "
            "Returns the risk bucket distribution (High/Mid/Low) and the top high-risk employees "
            "with their contributing risk factors."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "description": "Max high-risk employees to return (default 15).", "default": 15},
            },
            "required": [],
        },
    },
    {
        "name": "get_equity_gaps",
        "description": (
            "Fetch employees who are significantly underpaid relative to the 2026 market benchmark. "
            "A negative external_gap_pct indicates underpayment. "
            "Use this to correlate pay inequity with attrition risk."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "description": "Max records to return (default 10).", "default": 10},
            },
            "required": [],
        },
    },
]


# ---------------------------------------------------------------------------
# Main agent — agentic loop with streaming
# ---------------------------------------------------------------------------

def run_attrition_agent(company_name: str, month: str | None = None) -> str:
    """
    Run the Claude-powered attrition analysis agent.

    Claude autonomously calls the MongoDB tools above to gather data, then
    produces a final report covering:
      - Attrition risk level (High / Medium / Low)
      - Root cause breakdown (compensation, management, workload, career)
      - Specific employee risk signals from the Cox model
      - Prioritized HR action plan

    Args:
        company_name: Company to analyze (must match Glassdoor 'firm' field).
        month: Analysis month in YYYY-MM format. Defaults to current month.

    Returns:
        The final report as a string.
    """
    if month is None:
        month = datetime.now().strftime("%Y-%m")

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY is not set in config.env or environment.")

    client = anthropic.Anthropic(api_key=api_key)

    system_prompt = """You are a senior HR analytics consultant specializing in employee retention.
You have access to tools that query a MongoDB database containing:
  1. Glassdoor employee review sentiment analysis (NLP-labeled by topic)
  2. Cox survival model attrition risk scores for individual employees
  3. Salary equity gap analysis comparing employees against market benchmarks

Your task: conduct a thorough attrition risk analysis by calling the available tools
to gather data, then produce a comprehensive report.

The report must include:
  ## 1. Overall Attrition Risk Assessment
     - Risk level: High / Medium / Low with justification

  ## 2. Root Cause Analysis
     - Which drivers (compensation, management, workload, career) are most critical
     - Evidence from both Glassdoor sentiment data and employee risk model data

  ## 3. High-Risk Employee Signals
     - How many employees are in the High risk bucket
     - Common patterns among high-risk employees (pay gap, overtime, promotion stagnation)

  ## 4. Recommended HR Actions
     - Prioritized list of interventions (be specific, not generic)
     - Which teams or roles to focus on first

Be data-driven. Cite specific percentages and counts from the data you retrieve."""

    user_message = (
        f"Please conduct a full attrition risk analysis for **{company_name}** "
        f"for the period **{month}**. "
        f"Start by gathering the emotion summary, then fetch high-risk reviews and "
        f"retention risk data. Cross-reference salary equity gaps with attrition signals. "
        f"Synthesize everything into an actionable HR report."
    )

    messages = [{"role": "user", "content": user_message}]
    final_report = ""

    print(f"\n{'='*60}")
    print(f"  RetentionAgent — Attrition Analysis")
    print(f"  Company: {company_name} | Period: {month}")
    print(f"{'='*60}\n")

    # Agentic loop: continue until Claude stops calling tools
    turn = 0
    while True:
        turn += 1
        print(f"[Turn {turn}] Calling Claude...\n")

        # Stream the response so output appears in real time
        with client.messages.stream(
            model="claude-opus-4-6",
            max_tokens=8000,
            thinking={"type": "adaptive"},
            system=system_prompt,
            tools=TOOLS,
            messages=messages,
        ) as stream:
            for text in stream.text_stream:
                print(text, end="", flush=True)

            response = stream.get_final_message()

        print()  # newline after streamed output

        # Collect tool use blocks from this turn
        tool_use_blocks = [b for b in response.content if b.type == "tool_use"]

        # If no tool calls, Claude is done
        if response.stop_reason == "end_turn" or not tool_use_blocks:
            # Extract final text as the report
            final_report = " ".join(
                b.text for b in response.content if b.type == "text"
            )
            break

        # Append Claude's full response (preserves tool_use blocks in history)
        messages.append({"role": "assistant", "content": response.content})

        # Execute every tool Claude requested and collect results
        tool_results = []
        for block in tool_use_blocks:
            print(f"\n[Tool] {block.name}({json.dumps(block.input)})")
            result_str = execute_tool(block.name, block.input)
            print(f"[Result] {result_str[:200]}{'...' if len(result_str) > 200 else ''}\n")

            tool_results.append({
                "type":        "tool_result",
                "tool_use_id": block.id,
                "content":     result_str,
            })

        # Feed results back to Claude as a user message
        messages.append({"role": "user", "content": tool_results})

    print(f"\n{'='*60}")
    print("  Analysis complete.")
    print(f"{'='*60}\n")

    return final_report


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Claude-powered attrition risk analyzer")
    parser.add_argument("--company", default="Apple",   help="Company name to analyze")
    parser.add_argument("--month",   default=None,      help="Month in YYYY-MM format (default: current month)")
    args = parser.parse_args()

    report = run_attrition_agent(company_name=args.company, month=args.month)