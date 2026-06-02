"""Entry point for the STAI-X Award B automation agent.

Drives the staged agent orchestration (`orchestrator.run_orchestrated_analysis`),
which threads an AnalysisState through profiling -> task inference -> planning ->
leakage gate -> modeling -> evaluation -> reporting -> review, producing
`submission.csv` + `report.pdf`. On any stage failure it falls back to the
deterministic `runner.run_analysis` so the deliverable is never lost.
"""

from pathlib import Path

from src.data_agent.orchestrator import run_orchestrated_analysis


if __name__ == "__main__":
    run_orchestrated_analysis(Path(__file__).resolve().parent)
