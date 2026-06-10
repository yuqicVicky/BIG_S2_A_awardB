#!/usr/bin/env python3
"""
Regenerate report.pdf from outputs/reports/20260610_120000_report.md.
Run: python make_report.py

This script reads the corrected markdown (updated 2026-06-10 to reflect actual
3-round run with sub_registered keep-best, CV RMSLE=0.34835) and converts it
to report.pdf in the repo root.
"""
import os, sys, shutil

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO, "scripts"))

MD   = os.path.join(REPO, "outputs", "reports", "20260610_120000_report.md")
PDF  = os.path.join(REPO, "outputs", "reports", "20260610_120000_report.pdf")
ROOT = os.path.join(REPO, "report.pdf")

os.makedirs(os.path.dirname(PDF), exist_ok=True)

try:
    from md_to_pdf import convert
    convert(MD, PDF)
    shutil.copy(PDF, ROOT)
    size_kb = os.path.getsize(ROOT) // 1024
    print(f"PDF written: {ROOT}  ({size_kb} KB)")
except Exception as e:
    # Fallback: copy markdown as report.pdf
    shutil.copy(MD, ROOT)
    print(f"PDF conversion failed; markdown copied to {ROOT}: {e}")

print("Done.")


if __name__ == "__main__":
    pass
