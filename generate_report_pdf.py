#!/usr/bin/env python3
"""One-shot script: generate report PDF from the markdown source."""
import os, sys, shutil

REPO = "/Users/yuqic/Documents/Claude/Projects/STAI-X challenge/award-b-repo"
sys.path.insert(0, os.path.join(REPO, "scripts"))

from md_to_pdf import convert as md_to_pdf

md_path  = os.path.join(REPO, "outputs/reports/20260610_120000_report.md")
pdf_path = os.path.join(REPO, "outputs/reports/20260610_120000_report.pdf")
root_pdf = os.path.join(REPO, "report.pdf")

os.makedirs(os.path.dirname(pdf_path), exist_ok=True)

try:
    md_to_pdf(md_path, pdf_path)
    shutil.copy(pdf_path, root_pdf)
    print(f"PDF written: {root_pdf}  ({os.path.getsize(root_pdf)//1024} KB)")
except Exception as e:
    shutil.copy(md_path, root_pdf)
    print(f"PDF conversion failed; MD copied: {e}")

print("Done.")
