"""
agents/retention/ai_salary_risk_agent.py

AI Salary Risk Agent — orchestrates the full pipeline:

  Tool  →  SalaryRiskAgent.run()       scores each employee (Cox + salary gap)
  API   →  Claude claude-opus-4-6      interprets scores, produces HR analysis JSON
  Store →  SalaryRiskAgent.save_results()  inserts enriched JSON to MongoDB: Risk

Pipeline:
  1. Call SalaryRiskAgent (retention_agent_v1.py) as a scoring tool.
     Returns list[dict] — one scored document per employee.
  2. Serialise scored docs and send to Claude API (streaming, adaptive thinking).
     Claude returns structured JSON with a batch summary and per-employee analysis.
  3. Merge Claude's analysis fields into each scored doc.
  4. Insert the enriched JSON documents to MongoDB collection: Risk (all runs preserved).

Enriched MongoDB document schema (Risk collection):
    {
        # ── Scoring layer (from SalaryRiskAgent) ─────────────────
        employee_id, scoring_date, role_name,
        current_salary, market_median_2026,
        salary_gap_abs, salary_gap_pct, salary_risk_tier,
        internal_salary_rank, performance_consistency,
        pay_gap_from_equity,
        cox_risk_score, cox_risk_pct, cox_risk_bucket,
        combined_risk_bucket, risk_factors,
        model_version, analysis_type,

        # ── AI analysis layer (from Claude) ──────────────────────
        risk_level,              # "High" | "Mid" | "Low"  ← one per employee
        claude_analysis: {
            risk_summary,        # 1-2 sentence narrative
            root_causes,         # key contributing factors
            recommendations,     # prioritised HR action items
            urgency,             # "Immediate" | "Near-term" | "Monitor"
            priority_score,      # 1–10 across the cohort (10 = most urgent)
        },
        analysis_date,           # UTC timestamp of Claude analysis
    }
"""

import os
import json
import uuid
import datetime
import anthropic
from pathlib import Path
from dotenv import load_dotenv
from agents.retention.retention_agent_v1 import SalaryRiskAgent

# ---------------------------------------------------------------------------
# Structured output schema for Claude's HR analysis response
# ---------------------------------------------------------------------------
_ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "employee_analyses": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "employee_id":    {"type": "string"},
                    "risk_level":     {"type": "string", "enum": ["High", "Mid", "Low"]},
                    "risk_summary":   {"type": "string"},
                    "root_causes":    {"type": "array", "items": {"type": "string"}},
                    "recommendations": {"type": "array", "items": {"type": "string"}},
                    "urgency":        {"type": "string", "enum": ["Immediate", "Near-term", "Monitor"]},
                    "priority_score": {"type": "integer"},
                },
                "required": [
                    "employee_id", "risk_level", "risk_summary", "root_causes",
                    "recommendations", "urgency", "priority_score",
                ],
                "additionalProperties": False,
            },
        },
    },
    "required": ["employee_analyses"],
    "additionalProperties": False,
}

_SYSTEM_PROMPT = """You are a senior HR analyst specialising in employee retention and \
compensation equity. You receive quantitative risk scores produced by a Cox proportional \
hazard survival model combined with market salary gap analysis.

For each employee, provide:
- risk_level: overall retention risk classification — "High", "Mid", or "Low".
  Use combined_risk_bucket as the primary signal; override upward if salary gap or Cox \
score clearly warrants it.
- risk_summary: 1–2 sentences explaining the main risk drivers.
- root_causes: key factors behind the risk (be specific, not generic).
- recommendations: prioritised HR actions to address the risk.
- urgency: "Immediate" (act within 2 weeks), "Near-term" (1–3 months), "Monitor" (3+ months).
- priority_score: 1–10 across the whole cohort (10 = most urgent).

Be concise and actionable. Avoid generic advice."""
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(_PROJECT_ROOT / "config.env")
class AISalaryRiskAgent:
    """
    Orchestrates Cox-based salary risk scoring (tool) → Claude API analysis → MongoDB.

    Attributes:
        scorer (SalaryRiskAgent): Scoring tool — runs Cox model + salary gap analysis.
        client (anthropic.Anthropic): Anthropic API client.
        model (str): Claude model ID.
    """

    MODEL = "claude-opus-4-6"

    def __init__(self, model_path: str, feature_json_path: str):
        """
        Initialise the scoring tool and the Anthropic client.

        Args:
            model_path       : Path to cox_retention_v1.pkl.
            feature_json_path: Path to cox_retention_v1_features.json.

        Raises:
            FileNotFoundError: If model or feature file is missing.
            RuntimeError     : If MongoDB connection fails.
        """
        self.scorer = SalaryRiskAgent(model_path, feature_json_path)
        api_key = os.getenv("CLAUDE_API_KEY")
        if not api_key:
            raise EnvironmentError("CLAUDE_API_KEY is not set in config.env")

        self.client = anthropic.Anthropic(api_key=api_key)
    # ------------------------------------------------------------------
    # Step 1 — scoring tool
    # ------------------------------------------------------------------

    def _run_scoring_tool(self, csv_path: str) -> list:
        """
        Call SalaryRiskAgent as a black-box scoring tool.

        Returns:
            list[dict]: Scored documents (JSON-serialisable, no datetime objects).
        """
        print("[Tool] SalaryRiskAgent — running Cox + salary gap scoring...")
        raw_docs = self.scorer.run(csv_path)
        print(f"[Tool] Scored {len(raw_docs)} employees.")

        # Strip non-JSON-serialisable datetime before sending to Claude
        clean = []
        for doc in raw_docs:
            clean.append({k: v for k, v in doc.items() if not isinstance(v, datetime.datetime)})
        return clean, raw_docs  # clean for Claude, raw for later merge

    # ------------------------------------------------------------------
    # Step 2 — Claude API analysis
    # ------------------------------------------------------------------

    def _analyze_with_claude(self, scored_docs: list) -> dict:
        """
        Send scored employee records to Claude for structured HR analysis.

        Uses:
          - claude-opus-4-6 with adaptive thinking (model reasons before responding)
          - Streaming to handle long outputs without HTTP timeout
          - Structured JSON output constrained by _ANALYSIS_SCHEMA

        Args:
            scored_docs: JSON-serialisable list of scored employee dicts.

        Returns:
            dict: Parsed JSON matching _ANALYSIS_SCHEMA.
        """
        user_content = (
            "Analyse the following employee salary risk records and produce "
            "structured HR insights.\n\n"
            f"Scored Records:\n{json.dumps(scored_docs, indent=2, default=str)}"
        )

        print(f"[Claude API] Sending {len(scored_docs)} records to {self.MODEL}...")

        with self.client.messages.stream(
            model=self.MODEL,
            max_tokens=8000,
            thinking={"type": "adaptive"},
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_content}],
            output_config={"format": {"type": "json_schema", "schema": _ANALYSIS_SCHEMA}},
        ) as stream:
            final = stream.get_final_message()

        # Extract thinking token count for transparency
        thinking_tokens = sum(
            len(getattr(b, "thinking", "") or "")
            for b in final.content
            if b.type == "thinking"
        )
        print(f"[Claude API] Done. Thinking chars: {thinking_tokens} | "
              f"Input tokens: {final.usage.input_tokens} | "
              f"Output tokens: {final.usage.output_tokens}")

        text_block = next(b for b in final.content if b.type == "text")
        return json.loads(text_block.text)

    # ------------------------------------------------------------------
    # Step 3 — merge + save
    # ------------------------------------------------------------------

    def _merge_and_save(self, raw_docs: list, analysis: dict) -> list:
        """
        Insert each run's results as new documents into the Risk collection.

        Every call generates a unique run_id (UUID4) shared across all documents
        in the batch, so all records from the same pipeline run can be queried
        together. No existing documents are modified — all historical runs are
        preserved.

        Args:
            raw_docs : Original scored dicts from SalaryRiskAgent.run().
            analysis : Parsed Claude JSON matching _ANALYSIS_SCHEMA.

        Returns:
            list[dict]: Enriched documents inserted into MongoDB.
        """
        employee_map = {
            item["employee_id"]: item
            for item in analysis.get("employee_analyses", [])
        }
        analysis_date = datetime.datetime.now(datetime.timezone.utc)
        run_id = str(uuid.uuid4())

        # Use the still-open DB connection from the scorer
        risk_coll = self.scorer.db.db["Risk"]
        db_name   = self.scorer.db.db.name
        print(f"[MongoDB] Inserting {len(raw_docs)} docs into {db_name}.Risk (run_id={run_id}) ...")

        enriched_docs = []
        inserted = 0

        for doc in raw_docs:
            emp_id = doc.get("employee_id", "")
            ai     = employee_map.get(emp_id, {})

            doc_to_insert = {
                "run_id":       run_id,
                "employee_id":  emp_id,
                **{k: v for k, v in doc.items() if k != "employee_id"},
                "risk_level": ai.get("risk_level", doc.get("combined_risk_bucket", "Low")),
                "claude_analysis": {
                    "risk_summary":    ai.get("risk_summary", ""),
                    "root_causes":     ai.get("root_causes", []),
                    "recommendations": ai.get("recommendations", []),
                    "urgency":         ai.get("urgency", "Monitor"),
                    "priority_score":  ai.get("priority_score", 0),
                },
                "analysis_date": analysis_date,
            }

            risk_coll.insert_one(doc_to_insert)
            inserted += 1
            print(f"  [INSERT] {emp_id}")

            enriched_docs.append(doc_to_insert)

        # Read-back one document to verify the write reached Atlas
        if enriched_docs:
            sample_id = enriched_docs[0].get("employee_id")
            verify = risk_coll.find_one({"run_id": run_id, "employee_id": sample_id}, {"_id": 0})
            if verify:
                print(f"[Verify] {sample_id} — stored fields: {list(verify.keys())}")
            else:
                print(f"[Verify] WARNING: {sample_id} not found after write!")

        self.scorer.db.close()
        print(f"[MongoDB] Done — inserted: {inserted} docs")
        return enriched_docs

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self, csv_path: str) -> list:
        """
        Execute the full AI salary risk pipeline.

          1. Score  — SalaryRiskAgent (Cox model + salary gap analysis)
          2. Analyse — Claude API (structured HR insights)
          3. Save   — MongoDB Risk collection (enriched JSON)

        Args:
            csv_path: Path to ibm_enhanced_test.csv.

        Returns:
            list[dict]: Enriched documents written to MongoDB.
        """
        # Step 1: scoring tool
        clean_docs, raw_docs = self._run_scoring_tool(csv_path)

        # Step 2: Claude API analysis
        analysis = self._analyze_with_claude(clean_docs)

        # Step 3: merge + save
        enriched_docs = self._merge_and_save(raw_docs, analysis)

        # Console summary (not stored in MongoDB)
        levels = {"High": 0, "Mid": 0, "Low": 0}
        for d in enriched_docs:
            levels[d.get("risk_level", "Low")] = levels.get(d.get("risk_level", "Low"), 0) + 1
        print(f"\nAI Salary Risk Agent complete — {len(enriched_docs)} records → MongoDB: Risk")
        print(f"  High: {levels['High']}  Mid: {levels['Mid']}  Low: {levels['Low']}")

        return enriched_docs


# ---------------------------------------------------------------------------
# Module-level entry point
# ---------------------------------------------------------------------------

def run_ai_salary_risk_agent():
    """Resolve project-root paths and run the AI salary risk pipeline."""
    base_dir          = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    model_path        = os.path.join(base_dir, "models", "cox_retention_v1.pkl")
    feature_json_path = os.path.join(base_dir, "models", "cox_retention_v1_features.json")
    csv_path          = os.path.join(base_dir, "data",   "ibm_enhanced_test.csv")

    agent = AISalaryRiskAgent(model_path, feature_json_path)
    agent.run(csv_path)


if __name__ == "__main__":
    run_ai_salary_risk_agent()