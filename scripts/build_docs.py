"""Render docs/*.md to self-contained HTML (images embedded) and, if Chrome or Chromium is found, to PDF.

    pip install markdown
    python3 scripts/build_docs.py [REPORT DECISION_RULES DETAILED_REPORT]     # default: all three

Set CHROME=/path/to/chrome to choose the browser used for the PDF.
"""
import base64, os, re, shutil, subprocess, sys

import markdown

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS = os.path.join(ROOT, "docs")
CSS = """body{font-family:-apple-system,Helvetica,Arial,sans-serif;color:#0b0b0b;max-width:900px;margin:28px auto;padding:0 24px;font-size:13px;line-height:1.5}
h1{font-size:22px;margin:30px 0 8px;line-height:1.25} h1:first-child{margin-top:0}
h2{font-size:17px;margin:26px 0 8px;padding-bottom:4px;border-bottom:1px solid #e6e5e0;page-break-after:avoid}
h3{font-size:14px;margin:18px 0 6px;page-break-after:avoid}
p{margin:6px 0} ul,ol{margin:6px 0 6px 20px;padding:0} li{margin:3px 0}
table{border-collapse:collapse;width:100%;margin:8px 0 10px;font-size:11.5px} tr{page-break-inside:avoid}
th,td{border:1px solid #e6e5e0;padding:4px 7px;text-align:left;vertical-align:top} th{background:#f5f5f2}
img{max-width:100%;display:block;margin:10px auto 2px;page-break-inside:avoid} em{color:#52514e;font-size:12px}
blockquote{margin:12px 0;padding:10px 14px;border-left:4px solid #2a78d6;background:#f3f7fd;page-break-inside:avoid}
blockquote p{margin:0}
code{font-family:Menlo,monospace;font-size:11.5px;background:#f5f5f2;padding:1px 4px;border-radius:3px}
pre{background:#f5f5f2;padding:10px 12px;border-radius:4px;overflow-x:auto;font-size:11px;page-break-inside:avoid} pre code{padding:0;background:none}
hr{border:none;border-top:1px solid #e6e5e0;margin:18px 0}
@page{size:A4;margin:14mm 12mm}"""


def chrome():
    for c in [os.environ.get("CHROME"), "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
              shutil.which("google-chrome"), shutil.which("chromium"), shutil.which("chromium-browser")]:
        if c and os.path.exists(c):
            return c
    return None


def build(name):
    md_path = os.path.join(DOCS, f"{name}.md")
    body = markdown.markdown(open(md_path).read(), extensions=["tables", "fenced_code"])

    def embed(m):
        p = os.path.normpath(os.path.join(DOCS, m.group(1)))
        return f'src="data:image/png;base64,{base64.b64encode(open(p, "rb").read()).decode()}"'

    body = re.sub(r'src="([^"]+\.png)"', embed, body)
    html = os.path.join(DOCS, f"{name}.html")
    open(html, "w").write(f"<!doctype html><html><head><meta charset='utf-8'><title>{name}</title><style>{CSS}</style></head><body>{body}</body></html>")
    c = chrome()
    if c:
        pdf = os.path.join(DOCS, f"{name}.pdf")
        subprocess.run([c, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={pdf}", f"file://{html}"],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("wrote", os.path.relpath(pdf, ROOT))
    else:
        print("wrote", os.path.relpath(html, ROOT), "(no Chrome found: open it in a browser and print to PDF)")


for n in sys.argv[1:] or ["REPORT", "DECISION_RULES", "DETAILED_REPORT"]:
    build(n)
