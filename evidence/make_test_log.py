"""Run `pytest -q` and save the real output as evidence/AI_A2_2423862_TEST_LOG.pdf."""
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

root = Path(__file__).resolve().parents[1]
res = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=root, capture_output=True, text=True)
text = f"pytest -q  (AI-A2-3862)   {datetime.now():%Y-%m-%d %H:%M:%S}   exit code {res.returncode}\n\n{res.stdout}{res.stderr}"
out = Path(__file__).with_name("AI_A2_2423862_TEST_LOG.pdf")
c = canvas.Canvas(str(out), pagesize=A4)
y = 800
c.setFont("Courier", 8)
for line in text.splitlines():
    if y < 40:
        c.showPage(); c.setFont("Courier", 8); y = 800
    c.drawString(40, y, line[:120]); y -= 10
c.save()
print(text)
print("wrote", out)
sys.exit(res.returncode)
