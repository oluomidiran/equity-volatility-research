"""Build the Word version of the report with a table of contents and ruled tables.

Pandoc's default Word output leaves tables unruled and omits a contents page.
This script adds both by post-processing the generated document XML.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MD = ROOT / "report" / "Equity_Volatility_Research_Report.md"
OUT = ROOT / "report" / "Equity_Volatility_Research_Report.docx"
FIGS = ROOT / "results" / "figures"

NAVY = "1F3864"
RULE = "C9D1DA"

TBL_BORDERS = (
    "<w:tblBorders>"
    f'<w:top w:val="single" w:sz="12" w:space="0" w:color="{NAVY}"/>'
    f'<w:bottom w:val="single" w:sz="12" w:space="0" w:color="{NAVY}"/>'
    '<w:left w:val="none" w:sz="0" w:space="0" w:color="auto"/>'
    '<w:right w:val="none" w:sz="0" w:space="0" w:color="auto"/>'
    f'<w:insideH w:val="single" w:sz="4" w:space="0" w:color="{RULE}"/>'
    '<w:insideV w:val="none" w:sz="0" w:space="0" w:color="auto"/>'
    "</w:tblBorders>"
)


def build() -> Path:
    subprocess.run(
        ["pandoc", str(MD), "-o", str(OUT), "--toc", "--toc-depth=2",
         f"--resource-path={ROOT/'report'}:{ROOT}:{FIGS}"],
        check=True,
    )

    tmp = OUT.with_suffix(".tmp.docx")
    with zipfile.ZipFile(OUT) as zin, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename == "word/document.xml":
                xml = data.decode("utf-8")
                # give every table a ruled top/bottom and light interior rows
                xml = re.sub(r"(<w:tblPr>)(?!.*?<w:tblBorders>)", r"\1" + TBL_BORDERS,
                             xml, flags=re.S)
                data = xml.encode("utf-8")
            zout.writestr(item, data)
    shutil.move(tmp, OUT)
    return OUT


if __name__ == "__main__":
    path = build()
    with zipfile.ZipFile(path) as z:
        xml = z.read("word/document.xml").decode()
    print(f"wrote {path.name}")
    print("  tables:", xml.count("<w:tbl>"), "| borders:", xml.count("<w:tblBorders>"))
    print("  equations:", xml.count("<m:oMath"), "| figures:", xml.count("<w:drawing>"))
    print("  TOC field:", "TOC" in xml)
