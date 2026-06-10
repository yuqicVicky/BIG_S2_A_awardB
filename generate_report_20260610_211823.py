#!/usr/bin/env python3
"""Generate report PDF for run 20260610_211823.
Run: python generate_report_20260610_211823.py
"""
import os
import sys
import shutil

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO, "scripts"))

md_path = os.path.join(REPO, "outputs", "reports", "20260610_211823_report.md")
pdf_path = os.path.join(REPO, "outputs", "reports", "20260610_211823_report.pdf")
root_pdf = os.path.join(REPO, "report.pdf")

os.makedirs(os.path.dirname(pdf_path), exist_ok=True)

try:
    from md_to_pdf import convert as md_to_pdf
    md_to_pdf(md_path, pdf_path)
    shutil.copy(pdf_path, root_pdf)
    size_kb = os.path.getsize(root_pdf) // 1024
    print(f"PDF written to {root_pdf}  ({size_kb} KB)")
except Exception as e:
    shutil.copy(md_path, root_pdf)
    print(f"PDF conversion failed; markdown copied to {root_pdf}: {e}")

print("Done.")

if __name__ == "__main__":
    pass
