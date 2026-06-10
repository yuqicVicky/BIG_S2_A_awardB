"""Generate report PDF from the pre-written markdown."""
import os
import sys
import shutil

sys.path.insert(0, "/Users/yuqic/Documents/Claude/Projects/STAI-X challenge/award-b-repo/scripts")

md_path = "/Users/yuqic/Documents/Claude/Projects/STAI-X challenge/award-b-repo/outputs/reports/20260610_120000_report.md"
pdf_path = "/Users/yuqic/Documents/Claude/Projects/STAI-X challenge/award-b-repo/outputs/reports/20260610_120000_report.pdf"
root_pdf = "/Users/yuqic/Documents/Claude/Projects/STAI-X challenge/award-b-repo/report.pdf"

try:
    from md_to_pdf import convert as md_to_pdf
    md_to_pdf(md_path, pdf_path)
    shutil.copy(pdf_path, root_pdf)
    print(f"PDF written to {root_pdf}")
    print(f"PDF size: {os.path.getsize(root_pdf)} bytes")
except Exception as e:
    print(f"PDF conversion failed: {e}")
    shutil.copy(md_path, root_pdf)
    print(f"Markdown copied to {root_pdf} as fallback")
