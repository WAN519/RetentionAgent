"""
agents/pipeline.py

LangGraph orchestrator — runs all five agents in the correct dependency order.

Execution graph:

  START ──► equity ──► retention ──┐
        └──► emotion              ──┴──► recommendation ──► audit ──► END

  • equity and emotion both start from START in PARALLEL (no dependency).
  • retention runs after equity (needs its MongoDB Equity_Predictions output).
  • recommendation fans in from retention + emotion; waits for BOTH.
  • audit runs last; gates distribution to HR / managers.

Agent roles:
  equity          — LightGBM salary fairness scoring → MongoDB Equity_Predictions
  emotion         — Glassdoor sentiment analysis (claude_attrition_agent)
  retention       — Cox survival + Claude AI risk analysis → MongoDB Risk
  recommendation  — Per-employee retention recommendations with audit gate
  audit           — Adversarial critic; final quality gate before HR distribution

Usage:
    python -m agents.pipeline

    or with custom inputs:
        from agents.pipeline import run_pipeline
        result = run_pipeline(
            employee_csv="data/mock_dataset.csv",
            company_name="Apple",
            month="2026-04",
        )
"""

import operator
import traceback
from pathlib import Path
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph

# ──────────────────────────────────────────────────────────────────────────────
# Path constants
# ──────────────────────────────────────────────────────────────────────────────
_BASE = Path(__file__).resolve().parent.parent

_DEFAULT_EMPLOYEE_CSV   = str(_BASE / "data" / "ibm_enhanced_test.csv")
_DEFAULT_GLASSDOOR_CSV  = str(_BASE / "data" / "glassdoor_reviews.csv")
_EQUITY_MODEL_PATH      = str(_BASE / "models" / "agent_salary_regressor.pkl")
_COX_MODEL_PATH         = str(_BASE / "models" / "cox_retention_v1.pkl")
_COX_FEATURES_PATH      = str(_BASE / "models" / "cox_retention_v1_features.json")


# ──────────────────────────────────────────────────────────────────────────────
# Shared pipeline state
# ──────────────────────────────────────────────────────────────────────────────
class PipelineState(TypedDict):
    # ── inputs ────────────────────────────────────────────────────────────────
    employee_csv_path:  str          # CSV with IBM-schema employee records
    glassdoor_csv_path: str          # Glassdoor reviews CSV for emotion agent
    company_name:       str          # company name for emotion analysis
    month:              str | None   # YYYY-MM, defaults to current month

    # ── outputs ───────────────────────────────────────────────────────────────
    emotion_report:     str | None   # narrative HR report from emotion agent
    equity_done:        bool         # True once equity agent writes to MongoDB
    retention_run_id:   str | None   # run_id of the latest Risk batch
    retention_docs:     list | None  # enriched employee risk docs
    recommendation_done: bool         # True once recommendation agent completes
    audit_result:       dict | None  # full audit report from audit agent

    # ── error log (reducer allows parallel nodes to append safely) ────────────
    errors: Annotated[list[str], operator.add]


# ──────────────────────────────────────────────────────────────────────────────
# Node 1 — Emotion agent  (parallel branch A)
# ──────────────────────────────────────────────────────────────────────────────
def emotion_node(state: PipelineState) -> dict:
    """
    Run Glassdoor sentiment analysis via Claude.
    Produces a free-text HR report; stores employee-level reports to MongoDB.
    """
    print("\n[Pipeline] ▶ emotion_node started")
    try:
        from agents.emotion.claude_attrition_agent import run_attrition_agent

        report = run_attrition_agent(
            company_name=state["company_name"],
            csv_path=state["glassdoor_csv_path"],
            month=state.get("month"),
        )
        print("[Pipeline] ✓ emotion_node complete")
        return {"emotion_report": report, "errors": []}

    except Exception as exc:
        msg = f"[emotion_node] {type(exc).__name__}: {exc}"
        print(f"[Pipeline] ✗ {msg}")
        traceback.print_exc()
        return {"emotion_report": None, "errors": [msg]}


# ──────────────────────────────────────────────────────────────────────────────
# Node 2 — Equity agent  (parallel branch B)
# ──────────────────────────────────────────────────────────────────────────────
def equity_node(state: PipelineState) -> dict:
    """
    Run LightGBM salary fairness scoring.
    Writes results to MongoDB Equity_Predictions for downstream retention agent.
    """
    print("\n[Pipeline] ▶ equity_node started")
    try:
        from agents.equity.equity_agent import EquityAgent

        agent = EquityAgent(model_path=_EQUITY_MODEL_PATH)
        agent.run_analysis_pipeline(state["employee_csv_path"])

        print("[Pipeline] ✓ equity_node complete")
        return {"equity_done": True, "errors": []}

    except Exception as exc:
        msg = f"[equity_node] {type(exc).__name__}: {exc}"
        print(f"[Pipeline] ✗ {msg}")
        traceback.print_exc()
        # Non-fatal: retention can still run (pay_gap_from_equity will be NaN)
        return {"equity_done": False, "errors": [msg]}


# ──────────────────────────────────────────────────────────────────────────────
# Node 3 — Retention agent  (fan-in; waits for both emotion + equity)
# ──────────────────────────────────────────────────────────────────────────────
def retention_node(state: PipelineState) -> dict:
    """
    Run Cox survival model + Claude AI risk analysis.
    Runs in parallel with emotion_node after equity_node completes.
    Reads equity signals from MongoDB Equity_Predictions (written by equity_node).
    Writes enriched results to MongoDB Risk collection.
    """
    print("\n[Pipeline] ▶ retention_node started")
    if not state.get("equity_done"):
        print("[Pipeline]   ⚠ equity_node did not complete — retention will run without equity signals")

    try:
        from agents.retention.ai_salary_risk_agent import AISalaryRiskAgent

        agent = AISalaryRiskAgent(
            model_path=_COX_MODEL_PATH,
            feature_json_path=_COX_FEATURES_PATH,
        )
        docs = agent.run(state["employee_csv_path"])

        # Extract run_id from the first returned document
        run_id = docs[0].get("run_id") if docs else None

        print(f"[Pipeline] ✓ retention_node complete — run_id={run_id}, docs={len(docs)}")
        return {
            "retention_run_id": run_id,
            "retention_docs":   docs,
            "errors":           [],
        }

    except Exception as exc:
        msg = f"[retention_node] {type(exc).__name__}: {exc}"
        print(f"[Pipeline] ✗ {msg}")
        traceback.print_exc()
        return {"retention_run_id": None, "retention_docs": None, "errors": [msg]}


# ──────────────────────────────────────────────────────────────────────────────
# Node 4 — Recommendation agent  (fan-in from retention + emotion; before audit)
# ──────────────────────────────────────────────────────────────────────────────
def recommendation_node(state: PipelineState) -> dict:
    """
    Generate per-employee retention recommendations with built-in adversarial
    audit gate. Runs after both retention and emotion have completed.
    Reads risk scores + employee comments from MongoDB.
    """
    print("\n[Pipeline] ▶ recommendation_node started")
    try:
        from agents.recommendation.claude_recommendation_agent import run_recommendation_agent
        from datetime import datetime, timezone

        month = state.get("month") or datetime.now(timezone.utc).strftime("%Y-%m")
        run_recommendation_agent(month=month)

        print("[Pipeline] ✓ recommendation_node complete")
        return {"recommendation_done": True, "errors": []}

    except Exception as exc:
        msg = f"[recommendation_node] {type(exc).__name__}: {exc}"
        print(f"[Pipeline] ✗ {msg}")
        traceback.print_exc()
        return {"recommendation_done": False, "errors": [msg]}


# ──────────────────────────────────────────────────────────────────────────────
# Node 5 — Audit agent  (sequential; runs after recommendation)
# ──────────────────────────────────────────────────────────────────────────────
def audit_node(state: PipelineState) -> dict:
    """
    Adversarial critic: reviews the retention risk report and issues a
    verdict (APPROVED / NEEDS_REVIEW / REJECTED) before HR distribution.
    """
    print("\n[Pipeline] ▶ audit_node started")
    run_id = state.get("retention_run_id")

    if run_id is None:
        msg = "[audit_node] Skipped — no retention run_id available (retention_node may have failed)"
        print(f"[Pipeline] ✗ {msg}")
        return {"audit_result": None, "errors": [msg]}

    try:
        from agents.audit.audit_agent import AuditAgent

        result = AuditAgent().run(run_id=run_id)

        print(f"[Pipeline] ✓ audit_node complete — verdict={result.get('verdict')}")
        return {"audit_result": result, "errors": []}

    except Exception as exc:
        msg = f"[audit_node] {type(exc).__name__}: {exc}"
        print(f"[Pipeline] ✗ {msg}")
        traceback.print_exc()
        return {"audit_result": None, "errors": [msg]}


# ──────────────────────────────────────────────────────────────────────────────
# Graph construction
# ──────────────────────────────────────────────────────────────────────────────
def build_graph() -> StateGraph:
    """
    Assemble and compile the LangGraph pipeline.

    Topology:
        START ──► equity ──► retention ──┐
                        └──► emotion   ──┴──► recommendation ──► audit ──► END
    """
    builder = StateGraph(PipelineState)

    # Register nodes
    builder.add_node("equity",          equity_node)
    builder.add_node("emotion",         emotion_node)
    builder.add_node("retention",       retention_node)
    builder.add_node("recommendation",  recommendation_node)
    builder.add_node("audit",           audit_node)

    # equity and emotion are independent — both start from START in parallel
    builder.add_edge(START,       "equity")
    builder.add_edge(START,       "emotion")

    # retention runs after equity (needs MongoDB Equity_Predictions output)
    builder.add_edge("equity",    "retention")

    # Fan-in: recommendation waits for BOTH retention and emotion to complete
    builder.add_edge("retention", "recommendation")
    builder.add_edge("emotion",   "recommendation")

    # Sequential: audit runs after recommendation
    builder.add_edge("recommendation", "audit")
    builder.add_edge("audit",          END)

    return builder.compile()


# ──────────────────────────────────────────────────────────────────────────────
# Public entry point
# ──────────────────────────────────────────────────────────────────────────────
def run_pipeline(
    employee_csv:   str = _DEFAULT_EMPLOYEE_CSV,
    glassdoor_csv:  str = _DEFAULT_GLASSDOOR_CSV,
    company_name:   str = "Apple",
    month:          str | None = None,
) -> PipelineState:
    """
    Execute the full multi-agent retention pipeline.

    Args:
        employee_csv : Path to IBM-schema employee CSV (used by equity + retention).
        glassdoor_csv: Path to Glassdoor reviews CSV (used by emotion agent).
        company_name : Company to analyze in the emotion agent.
        month        : Analysis month YYYY-MM. Defaults to current month.

    Returns:
        Final PipelineState with all agent outputs and any errors.
    """
    graph = build_graph()

    initial_state: PipelineState = {
        "employee_csv_path":  employee_csv,
        "glassdoor_csv_path": glassdoor_csv,
        "company_name":       company_name,
        "month":              month,
        "emotion_report":      None,
        "equity_done":         False,
        "retention_run_id":    None,
        "retention_docs":      None,
        "recommendation_done": False,
        "audit_result":        None,
        "errors":              [],
    }

    print("=" * 70)
    print("  RetentionAgent Pipeline — LangGraph Orchestrator")
    print(f"  employee_csv : {employee_csv}")
    print(f"  company      : {company_name}  |  month: {month or 'current'}")
    print("=" * 70)

    final_state = graph.invoke(initial_state)

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  PIPELINE COMPLETE")
    print("=" * 70)
    print(f"  Equity scoring   : {'✓' if final_state.get('equity_done') else '✗ (failed)'}")
    print(f"  Emotion report   : {'✓' if final_state.get('emotion_report') else '✗ (failed)'}")
    print(f"  Retention run_id : {final_state.get('retention_run_id') or '✗ (failed)'}")
    print(f"  Recommendations  : {'✓' if final_state.get('recommendation_done') else '✗ (failed)'}")

    audit = final_state.get("audit_result")
    if audit:
        verdict = audit.get("verdict", "—")
        score   = audit.get("quality_metrics", {}).get("overall_quality_score", "—")
        print(f"  Audit verdict    : {verdict}  (quality score: {score})")
    else:
        print("  Audit verdict    : ✗ (failed)")

    errs = final_state.get("errors") or []
    if errs:
        print(f"\n  Errors ({len(errs)}):")
        for e in errs:
            print(f"    • {e}")

    print("=" * 70 + "\n")
    return final_state


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RetentionAgent LangGraph Pipeline")
    parser.add_argument("--employee-csv",  default=_DEFAULT_EMPLOYEE_CSV)
    parser.add_argument("--glassdoor-csv", default=_DEFAULT_GLASSDOOR_CSV)
    parser.add_argument("--company",       default="Apple")
    parser.add_argument("--month",         default=None)
    args = parser.parse_args()

    run_pipeline(
        employee_csv=args.employee_csv,
        glassdoor_csv=args.glassdoor_csv,
        company_name=args.company,
        month=args.month,
    )
