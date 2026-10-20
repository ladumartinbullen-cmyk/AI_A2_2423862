"""Create the declaration PDF template. Print it, sign it, and re-save/scan it over this file."""
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

out = Path(__file__).with_name("AI_A2_2423862_DECLARATION.pdf")
c = canvas.Canvas(str(out), pagesize=A4)
y = 800
lines = [
    ("Declaration of Authorship and AI Use", 16),
    ("SWE 3513 Artificial Intelligence - Individual Assignment 2", 11),
    ("", 10),
    ("Name: Bullen Ladu Martin          Registration number: 24/23862", 11),
    ("Verification code: AI-A2-3862", 11),
    ("", 10),
    ("I declare that I submit this project as my own individual work. I understand the code,", 10.5),
    ("the models and the report, and I can explain and modify them in the assessor's verification.", 10.5),
    ("I used an AI assistant (Claude, Anthropic) as disclosed in evidence/AI_USE.md.", 10.5),
    ("The dataset is synthetic. I have not included passwords, tokens or private keys.", 10.5),
]
for text, size in lines:
    c.setFont("Helvetica-Bold" if size > 12 else "Helvetica", size)
    c.drawString(60, y, text)
    y -= size + 8
y -= 30
c.drawString(60, y, "Signature: ______________________________        Date: ____________________")
c.save()
print("wrote", out)
