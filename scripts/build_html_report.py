"""Build a self-contained, print-ready HTML report from the markdown source.

Design goals:
  - figures and tables never split across printed pages
  - all images and formulas embedded as base64, so the file is standalone
  - serif body, navy headings, ruled tables with right-aligned numerics
  - print stylesheet tuned so browser "Save as PDF" produces the final artefact
"""
from __future__ import annotations

import base64
import io
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams["mathtext.fontset"] = "stix"


# serif maths that sits with a Charter/Georgia body rather than clashing with it
matplotlib.rcParams["mathtext.fontset"] = "stix"


ROOT = Path(__file__).resolve().parents[1]
REPORT_MD = ROOT / "report" / "Equity_Volatility_Research_Report.md"
FIG_DIR = ROOT / "results" / "figures"
OUT = ROOT / "report" / "Equity_Volatility_Research_Report.html"

NAVY = "#1F3864"


# ---------------------------------------------------------------- math -----
BODY_PT = 11.2          # body font size, must match CSS
DPI = 240               # render resolution (sharpness)


def _tex_to_png(tex: str, fontsize: float = 13.0, display: bool = True) -> str:
    """Render a LaTeX fragment to a transparent base64 PNG at natural size.

    The image is emitted with an explicit em width so that the typeset maths
    sits at the same optical size as the surrounding text rather than being
    stretched to the column width.
    """
    tex = tex.replace(r"\text{", r"\mathrm{").replace(r"\mathbb{1}", r"\mathbf{1}")
    tex = tex.replace(r"\cdot", r"\cdot ").replace("&", "")
    fig = plt.figure(figsize=(0.01, 0.01))
    fig.text(0, 0, f"${tex}$", fontsize=fontsize, color="#24282D")
    buf = io.BytesIO()
    try:
        fig.savefig(buf, format="png", dpi=DPI, transparent=True,
                    bbox_inches="tight", pad_inches=0.02)
    except Exception:
        plt.close(fig)
        return ""
    plt.close(fig)
    data = buf.getvalue()
    from PIL import Image
    w_px, h_px = Image.open(io.BytesIO(data)).size
    px_per_pt = DPI / 72.0
    w_em = (w_px / px_per_pt) / BODY_PT
    h_em = (h_px / px_per_pt) / BODY_PT
    b64 = base64.b64encode(data).decode()
    if display:
        return (f'<img class="mathblock" style="width:{w_em:.2f}em" '
                f'src="data:image/png;base64,{b64}" alt="{tex}">')
    return (f'<img class="mathinline" style="width:{w_em:.2f}em;height:{h_em:.2f}em" '
            f'src="data:image/png;base64,{b64}" alt="{tex}">')


# --------------------------------------------------------------- images ----
def _embed_figure(rel_path: str, caption: str, number: int) -> str:
    name = Path(rel_path).name
    src = FIG_DIR / name
    if not src.exists():
        return ""
    b64 = base64.b64encode(src.read_bytes()).decode()
    return (
        '<figure>'
        f'<img src="data:image/png;base64,{b64}" alt="{caption}">'
        f'<figcaption><span class="fignum">Figure {number}.</span> {caption}</figcaption>'
        '</figure>'
    )


# --------------------------------------------------------------- tables ----
def _table(block: list[str]) -> str:
    # protect escaped pipes so they are not treated as column separators
    rows = [
        [c.replace("\x00", "|") for c in
         r.replace("\\|", "\x00").strip().strip("|").split("|")]
        for r in block
    ]
    align_row = rows[1]
    aligns = []
    for a in align_row:
        a = a.strip()
        aligns.append("right" if a.endswith(":") and not a.startswith(":")
                      else "center" if a.startswith(":") and a.endswith(":")
                      else "left")
    head = rows[0]
    body = rows[2:]
    ncol = max(len(head), len(aligns))
    while len(aligns) < ncol:
        aligns.append("left")
    out = ["<table>", "<thead><tr>"]
    for i, c in enumerate(head):
        al = aligns[i] if i < len(aligns) else "left"
        out.append(f'<th style="text-align:{al}">{_inline(c.strip())}</th>')
    out.append("</tr></thead><tbody>")
    for r in body:
        out.append("<tr>")
        for i, c in enumerate(r):
            al = aligns[i] if i < len(aligns) else "left"
            out.append(f'<td style="text-align:{al}">{_inline(c.strip())}</td>')
        out.append("</tr>")
    out.append("</tbody></table>")
    return "".join(out)



# --------------------------------------------------- simple inline maths ----
_GREEK = {r"\lambda": "\u03bb", r"\sigma": "\u03c3", r"\mu": "\u03bc",
          r"\beta": "\u03b2", r"\rho": "\u03c1", r"\epsilon": "\u03b5",
          r"\alpha": "\u03b1", r"\Delta": "\u0394", r"\theta": "\u03b8",
          r"\gamma": "\u03b3"}
_OPS = {r"\times": "\u00d7", r"\approx": "\u2248", r"\cdot": "\u00b7",
        r"\geq": "\u2265", r"\leq": "\u2264", r"\in": "\u2208",
        r"\pm": "\u00b1", r"\ldots": "\u2026", r"\to": "\u2192",
        r"\,": " ", r"\;": " "}
# presence of any of these means the expression needs proper typesetting
_NEEDS_IMAGE = (r"\frac", r"\sqrt", r"\sum", r"\prod", r"\hat", r"\bar",
                r"\left", r"\right", r"\mathbf", r"\mathbb", r"\arg",
                r"\min", r"\big", "|")


def _simple_math_html(tex: str):
    """Render simple inline maths as real text, or return None to fall back.

    Keeping short expressions as text means they are selectable, searchable,
    and readable by assistive technology. Anything requiring true mathematical
    typesetting is rejected here and rendered as an image instead.
    """
    if any(tok in tex for tok in _NEEDS_IMAGE):
        return None
    s = tex.strip()
    for k, v in {**_GREEK, **_OPS}.items():
        s = s.replace(k, v)
    s = re.sub(r"\\(?:text|mathrm)\{([^{}]*)\}", r"<span class='mup'>\1</span>", s)
    s = re.sub(r"_\{([^{}]*)\}", r"<sub>\1</sub>", s)
    s = re.sub(r"\^\{([^{}]*)\}", r"<sup>\1</sup>", s)
    s = re.sub(r"_([A-Za-z0-9])", r"<sub>\1</sub>", s)
    s = re.sub(r"\^([A-Za-z0-9])", r"<sup>\1</sup>", s)
    if "\\" in s or "{" in s or "}" in s:
        return None
    return f"<span class='mi'>{s}</span>"


# --------------------------------------------------------------- inline ----
def _inline(s: str) -> str:
    s = s.replace("\\|", "&#124;")
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    s = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<em>\1</em>", s)
    s = s.replace("---", "&mdash;").replace("--", "&ndash;")
    # inline math
    def m(match):
        tex = match.group(1)
        html = _simple_math_html(tex)
        return html if html is not None else _tex_to_png(tex, fontsize=BODY_PT, display=False)
    s = re.sub(r"\$([^$]+)\$", m, s)
    return s


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


# ----------------------------------------------------------------- main ----
def build() -> Path:
    md = REPORT_MD.read_text()
    md = re.sub(r"^---\n.*?\n---\n", "", md, flags=re.S)  # strip YAML

    lines = md.split("\n")
    html: list[str] = []
    toc: list[tuple[int, str, str]] = []
    fig_no = 0
    i = 0

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # display math
        if stripped.startswith("$$") and stripped.endswith("$$") and len(stripped) > 4:
            html.append(f'<div class="matheq">{_tex_to_png(stripped[2:-2], fontsize=BODY_PT * 1.18)}</div>')
            i += 1
            continue

        # figure
        fig_m = re.match(r"!\[(.*?)\]\((.*?)\)", stripped)
        if fig_m:
            fig_no += 1
            html.append(_embed_figure(fig_m.group(2), fig_m.group(1), fig_no))
            i += 1
            continue

        # headings
        h = re.match(r"^(#{1,3})\s+(.*)", stripped)
        if h:
            lvl = len(h.group(1))
            text = h.group(2)
            sid = _slug(text)
            toc.append((lvl, text, sid))
            html.append(f'<h{lvl} id="{sid}">{_inline(text)}</h{lvl}>')
            i += 1
            continue

        # table
        if stripped.startswith("|") and i + 1 < len(lines) and re.match(r"^\|[\s:\-|]+\|$", lines[i + 1].strip()):
            block = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                block.append(lines[i])
                i += 1
            html.append(_table(block))
            continue

        # list
        if stripped.startswith(("- ", "* ")):
            items = []
            while i < len(lines) and lines[i].strip().startswith(("- ", "* ")):
                items.append(f"<li>{_inline(lines[i].strip()[2:])}</li>")
                i += 1
            prev = html[-1] if html else ""
            cls = ' class="contentslist"' if 'id="contents"' in prev else ""
            html.append(f"<ul{cls}>" + "".join(items) + "</ul>")
            continue

        if re.match(r"^\d+\.\s", stripped):
            items = []
            while i < len(lines) and re.match(r"^\d+\.\s", lines[i].strip()):
                items.append(f"<li>{_inline(re.sub(r'^\d+\.\s', '', lines[i].strip()))}</li>")
                i += 1
            html.append("<ol>" + "".join(items) + "</ol>")
            continue

        # paragraph
        if stripped:
            para = [stripped]
            i += 1
            while i < len(lines) and lines[i].strip() and not re.match(
                    r"^(#{1,3}\s|\||\$\$|!\[|- |\* |\d+\.\s)", lines[i].strip()):
                para.append(lines[i].strip())
                i += 1
            html.append(f"<p>{_inline(' '.join(para))}</p>")
            continue

        i += 1

    doc = TEMPLATE.format(css=CSS, toc="", body="\n".join(html))
    OUT.write_text(doc, encoding="utf-8")
    return OUT


CSS = """
@page { size: Letter; margin: 22mm 20mm; }

:root { --navy:#1F3864; --accent:#2E75B6; --ink:#1A1A1A; --rule:#C9D1DA; --muted:#5B6570; }

html { -webkit-print-color-adjust: exact; print-color-adjust: exact; }

body {
  font-family: Charter, "Bitstream Charter", Georgia, "Times New Roman", serif;
  font-size: 11.2pt; line-height: 1.62; color: var(--ink);
  max-width: 46em; margin: 0 auto; padding: 3em 1.5em 5em;
  text-rendering: optimizeLegibility;
}

h1.doctitle {
  font-size: 22pt; line-height: 1.25; color: var(--navy); margin: 0 0 .3em;
  letter-spacing: -0.01em;
}
p.byline { color: var(--muted); font-size: 11pt; margin: 0 0 .4em; }
p.dateline { color: var(--muted); font-size: 10pt; margin: 0 0 2.4em; }
hr.titlerule { border: 0; border-top: 2px solid var(--navy); margin: 0 0 2.2em; }

h1 { font-size: 15.5pt; color: var(--navy); margin: 2.1em 0 .5em;
     padding-bottom: .22em; border-bottom: 1px solid var(--rule);
     page-break-after: avoid; break-after: avoid; }
h2 { font-size: 12.6pt; color: var(--accent); margin: 1.7em 0 .4em;
     page-break-after: avoid; break-after: avoid; }
h3 { font-size: 11.6pt; color: var(--ink); margin: 1.3em 0 .35em;
     page-break-after: avoid; break-after: avoid; }

p { margin: 0 0 .95em; orphans: 3; widows: 3; }
strong { color: var(--navy); }
code { font-family: "DejaVu Sans Mono", Menlo, Consolas, monospace;
       font-size: .87em; background: #F2F5F8; padding: .1em .32em;
       border-radius: 3px; color: #24303C; }

ul, ol { margin: 0 0 1em; padding-left: 1.4em; }
li { margin: .22em 0; }

/* --- table of contents --- */
nav.toc { margin: 0 0 2.6em; padding: 1.1em 1.4em; background: #F7F9FB;
          border: 1px solid var(--rule); border-radius: 4px;
          page-break-inside: avoid; break-inside: avoid; }
nav.toc h2 { margin: 0 0 .5em; font-size: 12pt; color: var(--navy); }
nav.toc ul { list-style: none; padding: 0; margin: 0; }
nav.toc li { margin: .16em 0; font-size: 10.4pt; }
nav.toc li.toc1 { font-weight: 600; margin-top: .5em; }
nav.toc li.toc2 { padding-left: 1.4em; }
nav.toc a { color: var(--ink); text-decoration: none; }
nav.toc a:hover { color: var(--accent); }

/* --- tables: never split, ruled, numerics right --- */
table { width: 100%; border-collapse: collapse; margin: 1.2em 0 1.5em;
        font-size: 10.1pt; page-break-inside: avoid; break-inside: avoid; }
thead th { border-bottom: 1.6px solid var(--navy); border-top: 1.2px solid var(--navy);
           padding: .5em .6em; color: var(--navy); font-weight: 600;
           background: #F7F9FB; }
tbody td { border-bottom: .6px solid var(--rule); padding: .42em .6em; }
tbody tr:last-child td { border-bottom: 1.2px solid var(--navy); }
tbody tr:nth-child(even) { background: #FBFCFD; }

/* --- figures: never split --- */
figure { margin: 1.6em 0 1.8em; text-align: center;
         page-break-inside: avoid; break-inside: avoid; }
figure img { max-width: 100%; height: auto; border: 1px solid #E3E8ED; border-radius: 3px; }
figcaption { font-size: 9.4pt; color: var(--muted); margin-top: .55em;
             text-align: left; line-height: 1.45; }
figcaption .fignum { color: var(--navy); font-weight: 600; }

/* --- math --- */
.matheq { text-align: center; margin: 1.15em 0 1.25em;
          page-break-inside: avoid; break-inside: avoid; }
img.mathblock { max-width: 100%; height: auto; vertical-align: middle; }
img.mathinline { vertical-align: -0.28em; margin: 0 .06em; }
.mi { font-style: italic; white-space: nowrap; }
.mi sub, .mi sup { font-style: normal; font-size: .72em; line-height: 0; }
.mi sub { vertical-align: -0.28em; }
.mi sup { vertical-align: 0.42em; }
.mi .mup { font-style: normal; }

h2#contents { page-break-before: always; break-before: page;
              font-size: 15.5pt; color: var(--navy); margin-bottom: .8em; }
ul.contentslist { list-style: none; padding-left: 0; margin: 0; }
ul.contentslist li { margin: .3em 0; font-size: 10.6pt;
                     border-bottom: 1px dotted #DDE3EA; padding-bottom: .28em; }
ul.contentslist li a { color: var(--ink); text-decoration: none; }
ul.contentslist li a:hover { color: var(--accent); }
ul.contentslist { page-break-after: always; break-after: page; }
h1#references, h1#appendix-reproduction {
  page-break-before: always; break-before: page; }

@media print {
  body { padding: 0; max-width: none; font-size: 10.6pt; }
  nav.toc { break-after: auto; }
  a { color: var(--ink); text-decoration: none; }
}
"""

TEMPLATE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>Equity Volatility Research Pipeline</title>
<style>{css}</style></head>
<body>
<h1 class="doctitle">Equity Volatility Research Pipeline</h1>
<p class="byline">Oluwaferanmi A. Omidiran</p>
<p class="dateline">Strategy Backtesting, Market Regime Analysis, and Forecasting with Out-of-Sample Validation in Python</p>
<hr class="titlerule">
{toc}
{body}
</body></html>
"""


if __name__ == "__main__":
    path = build()
    print("wrote", path, f"({path.stat().st_size/1024:.0f} KB)")
