"""Generate fictional report examples offline using the real scoring and reporting code."""
from __future__ import annotations

import json
from html import escape
from pathlib import Path

from narrative_dislocation.final_score import combine_dislocation_score
from narrative_dislocation.reporting import write_reports
from narrative_dislocation.scoring import score_candidate

ROOT = Path(__file__).resolve().parent / "synthetic_report"
NOTICE = (
    "> **SYNTHETIC DEMO — NOT A REAL COMPANY OR AI RESEARCH.** All financial inputs "
    "and narrative statements are invented. The standard renderer's measured-fact "
    "and AI-inference labels describe output fields only. No market data was fetched, "
    "no AI request was made, and no source evidence was collected.\n\n"
)


def visual_summary(record: dict) -> str:
    """An illustrated report summary, not an application UI screenshot."""
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="720" viewBox="0 0 1200 720" role="img" aria-labelledby="title desc">',
        '<title id="title">StocksScreenerAI synthetic output example</title>',
        '<desc id="desc">Fictional inputs with actual computed scores, missing FCF growth, and a manually simulated narrative. No real company or AI research.</desc>',
        '<rect width="1200" height="720" rx="20" fill="#f1f5f9"/>',
        '<rect x="20" y="20" width="1160" height="680" rx="16" fill="white"/>',
        '<path d="M36 20H1164Q1180 20 1180 36V154H20V36Q20 20 36 20" fill="#132238"/>',
    ]

    def text(x, y, value, size=18, color="#172033", weight=400):
        parts.append(f'<text x="{x}" y="{y}" font-family="Arial, sans-serif" font-size="{size}" fill="{color}" font-weight="{weight}">{escape(str(value))}</text>')

    text(50, 57, "STOCKSSCREENERAI  /  OUTPUT EXAMPLE", 14, "#5eead4", 700)
    text(50, 101, "ExampleCo", 34, "white", 700)
    text(50, 132, "DEMO-ONLY · fictional company and financial inputs", 16, "#cbd5e1")
    text(882, 64, "ILLUSTRATIVE SCORE", 13, "#cbd5e1", 700)
    text(882, 108, f"{record['dislocation_score']:.1f} / 100", 32, "white", 700)
    text(50, 188, "SYNTHETIC DATA + MANUALLY SIMULATED AI · NO LIVE RESEARCH", 14, "#9a3412", 700)
    facts = record["narrative_analysis"]["measured_facts"]
    tiles = [("DECLINE FROM 2Y HIGH", f"{facts['drawdown_from_2y_high']:.0%}"),
             ("REVENUE GROWTH", f"{facts['revenue_growth']:+.0%}"),
             ("FCF GROWTH", "Not supplied")]
    for index, (label, value) in enumerate(tiles):
        x = 50 + index * 374
        parts.append(f'<rect x="{x}" y="210" width="352" height="102" rx="10" fill="#f8fafc" stroke="#e2e8f0"/>')
        text(x+18, 240, label, 13, "#64748b", 700)
        text(x+18, 281, value, 27, "#172033", 700)
    text(50, 358, "QUANT EVIDENCE", 14, "#0f766e", 700)
    labels = [("Price dislocation", "price_dislocation_score"),
              ("Fundamental resilience", "fundamental_resilience_score"),
              ("Valuation compression", "valuation_compression_score"),
              ("Balance-sheet quality", "balance_sheet_quality_score")]
    for index, (label, key) in enumerate(labels):
        y = 397 + index * 47
        text(50, y, label, 16)
        parts.append(f'<rect x="258" y="{y-13}" width="210" height="12" rx="6" fill="#e2e8f0"/>')
        parts.append(f'<rect x="258" y="{y-13}" width="{record[key]*2.1:.2f}" height="12" rx="6" fill="#0f766e"/>')
        text(485, y, f"{record[key]:.1f}", 16, "#172033", 700)
    text(50, 591, f"Weighted data coverage: {record['quant_coverage']:.1f}%", 16, "#64748b")
    text(615, 358, "SIMULATED NARRATIVE", 14, "#0f766e", 700)
    text(615, 396, "Question to investigate", 17, weight=700)
    text(615, 423, "Temporary demand slowdown or lasting damage?", 17)
    text(615, 468, "What would invalidate the thesis?", 17, weight=700)
    text(615, 497, "Two periods of revenue contraction, or", 17)
    text(615, 524, "a free-cash-flow margin below 10%.", 17)
    text(615, 574, "No evidence links: this is an invented example.", 16, "#64748b")
    parts.append('<path d="M50 622H1150" stroke="#e2e8f0"/>')
    text(50, 652, "Real scoring and report functions · fictional inputs · no API key needed", 17, weight=700)
    text(50, 679, "A preview of the report structure, not an investment recommendation or a performance claim.", 15, "#64748b")
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def main():
    fixture = json.loads((ROOT / "input.json").read_text(encoding="utf-8"))
    metrics, narrative = fixture["metrics"], fixture["simulated_ai"]
    quant = score_candidate(metrics)
    final = combine_dislocation_score(
        quant_score=quant["quant_score"], quant_coverage=quant["quant_coverage"],
        narrative_gap_score=narrative["narrative_gap_score"],
        value_trap_probability=narrative["value_trap_probability"],
    )
    record = {
        "ticker": fixture["ticker"], "company_name": fixture["company_name"],
        "synthetic_demo": True, **quant, **final,
        "narrative_analysis": {
            "measured_facts": {**metrics, "drawdown_from_2y_high": -metrics["drawdown_2y"]},
            "ai_inference": narrative, "evidence_sources": [],
            "ai_metadata": {"status": "synthetic_fixture", "api_called": False,
                            "note": "Manually authored example, not model output"},
        },
    }
    paths = write_reports([record], ROOT, run_metadata={
        "run_id": "synthetic-offline-example", "synthetic_demo": True,
        "network_requests": 0, "ai_requests": 0,
        "description": "Scoring and reporting only; no provider fetch, screen or AI research",
    })
    report = paths["markdown_report"]
    report.write_text(NOTICE + report.read_text(encoding="utf-8"), encoding="utf-8")
    (ROOT / "preview.svg").write_text(visual_summary(record), encoding="utf-8")
    print("Generated synthetic Markdown, CSV, JSON and SVG without network or AI calls.")


if __name__ == "__main__":
    main()
