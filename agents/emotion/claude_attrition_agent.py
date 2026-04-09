"""
agents/emotion/claude_attrition_agent.py

Claude Opus-powered Glassdoor sentiment analysis agent.

Workflow:
  1. Claude calls run_emotion_analysis → NLP pipeline labels every Glassdoor
     review by topic (management / salary / workload / career) and sentiment.
  2. Claude calls get_emotion_summary → aggregated topic-level statistics.
  3. Claude calls get_high_risk_reviews → concrete review excerpts with the
     highest negative scores for qualitative grounding.
  4. Claude synthesizes the sentiment data into an attrition risk report:
     root causes derived from employee voice, and targeted HR interventions.

All tool implementations live in tools/attrition_tools.py.

Usage:
  python -m agents.emotion.claude_attrition_agent \\
      --company Apple \\
      --csv glassdoor_reviews.csv \\
      --month 2026-04
"""

import copy
import os
import argparse
from datetime import datetime, timezone

import anthropic
import certifi
from dotenv import load_dotenv
from pathlib import Path
from pymongo import MongoClient

from tools.attrition_tools import TOOLS, execute_tool

# Resolve config.env relative to the project root regardless of the working directory
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(_PROJECT_ROOT / "config.env")

# Only expose the three emotion-related tools to this agent
EMOTION_TOOLS = [t for t in TOOLS if t["name"] in {
    "run_emotion_analysis",
    "get_emotion_summary",
    "get_high_risk_reviews",
}]

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a senior HR analyst specializing in employee sentiment and attrition prevention.

You have access to three tools that let you analyze Glassdoor employee reviews:

  1. run_emotion_analysis  — run the NLP pipeline on the raw Glassdoor CSV for a company
  2. get_emotion_summary   — retrieve aggregated sentiment and topic-issue rates from MongoDB
  3. get_high_risk_reviews — fetch the most negatively-scored review samples

Follow this sequence strictly:
  Step 1: Call run_emotion_analysis to label and store the reviews.
  Step 2: Call get_emotion_summary to get the quantitative breakdown.
  Step 3: Call get_high_risk_reviews to read actual employee language.
  Step 4: Write the final report.

Your final report must follow this structure:

## Overall Sentiment Assessment
[State the overall negative sentiment rate and what it signals about attrition risk.
 Cite the exact numbers returned by the tools.]

## Top Attrition Drivers (from Employee Voice)
[For each issue category that exceeds 20% of reviews, explain:
 - What percentage of reviews flag it
 - What the actual review excerpts reveal about employee frustration
 - Why this driver leads to voluntary turnover]

## Predicted Attrition Risk by Driver
[Rank the drivers from highest to lowest attrition impact. Justify the ranking
 using both the sentiment score and the review language.]

## Recommended HR Interventions
[Numbered list. For each recommendation:
 - State which attrition driver it addresses
 - Describe the specific action (not generic advice)
 - Identify which employee group is the priority target]

Be concrete. Quote employee language from the reviews. Every claim must be backed
by a number or a direct quote from the data you retrieved."""


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

def run_attrition_agent(
    company_name: str,
    csv_path: str,
    month: str | None = None,
) -> str:
    """
    Run the sentiment-driven attrition analysis agent.

    Args:
        company_name: Company to analyze. Must match the 'firm' field in the CSV.
        csv_path:     Path to the Glassdoor reviews CSV file.
        month:        Analysis month in YYYY-MM format. Defaults to current month.

    Returns:
        The final structured HR report as a string.
    """
    if month is None:
        month = datetime.now().strftime("%Y-%m")

    api_key = os.getenv("CLAUDE_API_KEY")
    if not api_key:
        raise EnvironmentError("CLAUDE_API_KEY is not set in config.env")

    client = anthropic.Anthropic(api_key=api_key)

    # Build a modified tool schema: strip csv_path from run_emotion_analysis.
    # Claude only needs to pass company_name + month; csv_path is injected by Python.
    emotion_tools = copy.deepcopy(EMOTION_TOOLS)
    for t in emotion_tools:
        if t["name"] == "run_emotion_analysis":
            t["input_schema"]["properties"].pop("csv_path", None)
            t["input_schema"]["required"] = [
                r for r in t["input_schema"].get("required", []) if r != "csv_path"
            ]

    # Wrapper that injects csv_path before dispatching to execute_tool
    def _execute(name: str, inputs: dict) -> str:
        if name == "run_emotion_analysis":
            inputs = {**inputs, "csv_path": csv_path}
        return execute_tool(name, inputs)

    user_message = (
        f"Please analyze employee sentiment and predict attrition risk for "
        f"**{company_name}**, period **{month}**.\n\n"
        f"Run the NLP pipeline first, then retrieve the aggregated sentiment "
        f"statistics, then sample the most negative reviews. "
        f"Identify the primary attrition drivers from the employee voice data "
        f"and provide specific HR interventions for each driver."
    )

    messages = [{"role": "user", "content": user_message}]
    final_report = ""

    print(f"\n{'='*62}")
    print(f"  RetentionAgent  |  Sentiment Attrition Analysis")
    print(f"  Company : {company_name}")
    print(f"  Period  : {month}")
    print(f"  CSV     : {csv_path}")
    print(f"{'='*62}\n")

    turn = 0
    while True:
        turn += 1
        print(f"[Turn {turn}] Sending request to Claude...\n")

        with client.messages.stream(
            model="claude-opus-4-6",
            max_tokens=8000,
            thinking={"type": "adaptive"},
            system=SYSTEM_PROMPT,
            tools=emotion_tools,
            messages=messages,
        ) as stream:
            for text in stream.text_stream:
                print(text, end="", flush=True)
            response = stream.get_final_message()

        print()

        tool_use_blocks = [b for b in response.content if b.type == "tool_use"]

        if response.stop_reason == "end_turn" or not tool_use_blocks:
            final_report = "".join(
                b.text for b in response.content if b.type == "text"
            )
            break

        messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        for block in tool_use_blocks:
            result = _execute(block.name, block.input)
            print(f"[Tool] {block.name}: {result[:300]}\n")
            tool_results.append({"type": "tool_result", "tool_use_id": block.id, "content": result})

        messages.append({"role": "user", "content": tool_results})

    print(f"\n{'='*62}")
    print("  Analysis complete.")
    print(f"{'='*62}\n")

    _save_report(final_report, company_name, month)

    return final_report


def _save_report(report: str, company_name: str, month: str) -> None:
    """
    Save one JSON document per employee to the Emotion collection.
    Each document contains the employee's individual risk data from
    Retention_Predictions plus the company-level sentiment report as context.
    """
    uri             = os.getenv("MONGODB_URI")
    db_name         = os.getenv("MONGODB_NAME", "MarketInformation")
    collection_name = os.getenv("COLLECTION_NAME_EMOTION", "Emotion")

    client = MongoClient(uri, serverSelectionTimeoutMS=5000, tlsCAFile=certifi.where())
    db = client[db_name]

    if collection_name not in db.list_collection_names():
        db.create_collection(collection_name)
        print(f"[MongoDB] Created collection '{collection_name}'")

    # Fetch all employees from Retention_Predictions (emotion-related fields only)
    employees = list(db["Retention_Predictions"].find(
        {},
        {"_id": 0, "employee_id": 1, "risk_score": 1, "risk_pct": 1,
         "risk_bucket": 1, "months_since_promotion": 1, "risk_reasons": 1}
    ))

    if not employees:
        print("[MongoDB] No employees found in Retention_Predictions — saving company-level report only.")
        db[collection_name].insert_one({
            "company":    company_name,
            "month":      month,
            "report":     report,
            "created_at": datetime.now(timezone.utc).isoformat(),
        })
        client.close()
        return

    now = datetime.now(timezone.utc).isoformat()
    docs = [
        {
            "employee_id":              emp["employee_id"],
            "company":                  company_name,
            "month":                    month,
            "risk_score":               emp.get("risk_score"),
            "risk_pct":                 emp.get("risk_pct"),
            "risk_bucket":              emp.get("risk_bucket"),
            "months_since_promotion":   emp.get("months_since_promotion"),
            "risk_reasons":             emp.get("risk_reasons"),
            "company_sentiment_report": report,
            "created_at":               now,
        }
        for emp in employees
    ]

    result = db[collection_name].insert_many(docs)
    print(f"[MongoDB] Saved {len(result.inserted_ids)} employee reports → collection='{collection_name}'")
    client.close()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Claude-powered Glassdoor sentiment attrition analyzer"
    )
    parser.add_argument("--company", default="Apple",
                        help="Company name to analyze (must match 'firm' column in CSV)")
    parser.add_argument("--csv",     default="glassdoor_reviews.csv",
                        help="Path to the Glassdoor reviews CSV file")
    parser.add_argument("--month",   default=None,
                        help="Analysis month in YYYY-MM format (default: current month)")
    args = parser.parse_args()

    report = run_attrition_agent(
        company_name=args.company,
        csv_path=args.csv,
        month=args.month,
    )