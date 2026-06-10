#!/usr/bin/env python3
"""Run report PDF generation inline."""
import os, sys, shutil

REPO = "/Users/yuqic/Documents/Claude/Projects/STAI-X challenge/award-b-repo"
sys.path.insert(0, os.path.join(REPO, "scripts"))

md_path  = os.path.join(REPO, "outputs/reports/20260610_120000_report.md")
pdf_path = os.path.join(REPO, "outputs/reports/20260610_120000_report.pdf")
root_pdf = os.path.join(REPO, "report.pdf")

os.makedirs(os.path.dirname(pdf_path), exist_ok=True)

try:
    from md_to_pdf import convert as md_to_pdf
    md_to_pdf(md_path, pdf_path)
    shutil.copy(pdf_path, root_pdf)
    size = os.path.getsize(root_pdf)
    print(f"SUCCESS: PDF written to {root_pdf} ({size} bytes)")
except Exception as e:
    import traceback
    traceback.print_exc()
    shutil.copy(md_path, root_pdf)
    print(f"FALLBACK: Markdown copied to {root_pdf}: {e}")

print("DONE")
