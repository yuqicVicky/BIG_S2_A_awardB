"""Entry point for the STAI-X Award B automation agent."""

from pathlib import Path

from src.data_agent.runner import run_analysis


if __name__ == "__main__":
    run_analysis(Path(__file__).resolve().parent)
