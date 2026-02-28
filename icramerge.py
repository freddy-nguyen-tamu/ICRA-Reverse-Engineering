from __future__ import annotations

import csv
from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak,
    Image as RLImage, Table, TableStyle, KeepTogether
)

#INPUT_DIR = Path(r"\\wsl.localhost\Ubuntu\home\projects\ICRA\results")
INPUT_DIR = Path("/home/projects/ICRA/result")
OUTPUT_FILE = INPUT_DIR / "merged_report.pdf"

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
CSV_EXTS = {".csv"}

# Limits to keep the PDF readable
MAX_CSV_ROWS = 200
MAX_CSV_COLS = 30


def group_key(name: str) -> str:
    stem = Path(name).stem
    lower = stem.lower()
    if "_case" in lower:
        return stem[: lower.index("_case")]
    if "_" in stem:
        return stem.split("_", 1)[0]
    return stem


def safe_csv_read(path: Path, max_rows: int, max_cols: int) -> list[list[str]]:
    rows: list[list[str]] = []
    with path.open("r", newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if i >= max_rows:
                break
            row = row[:max_cols]
            rows.append([cell.strip() for cell in row])

    # Ensure at least a 1x1 table
    if not rows:
        rows = [["(empty csv)"]]
    return rows


def build_csv_table(data: list[list[str]]) -> Table:
    tbl = Table(data, repeatRows=1)

    style = TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 7),
        ("LEADING", (0, 0), (-1, -1), 9),

        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),

        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ])
    tbl.setStyle(style)
    return tbl


def main() -> None:
    if not INPUT_DIR.is_dir():
        raise SystemExit(f"Directory not found: {INPUT_DIR}")

    files = [p for p in INPUT_DIR.iterdir() if p.is_file()]
    files = [p for p in files if p.suffix.lower() in (IMAGE_EXTS | CSV_EXTS)]

    # Sort: group, then CSV before images, then name
    def sort_key(p: Path):
        g = group_key(p.name).lower()
        ext_rank = 0 if p.suffix.lower() in CSV_EXTS else 1
        return (g, ext_rank, p.name.lower())

    files.sort(key=sort_key)

    # Group files
    groups: dict[str, list[Path]] = {}
    for p in files:
        groups.setdefault(group_key(p.name), []).append(p)

    # PDF setup
    doc = SimpleDocTemplate(
        str(OUTPUT_FILE),
        pagesize=letter,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
        title="Merged ICRA Results",
    )

    styles = getSampleStyleSheet()
    title_style = styles["Title"]
    group_style = ParagraphStyle(
        "GroupHeading",
        parent=styles["Heading1"],
        spaceBefore=12,
        spaceAfter=8,
    )
    item_style = ParagraphStyle(
        "ItemHeading",
        parent=styles["Heading3"],
        spaceBefore=10,
        spaceAfter=6,
    )
    meta_style = ParagraphStyle(
        "Meta",
        parent=styles["Normal"],
        textColor=colors.grey,
        fontSize=9,
        leading=11,
        spaceAfter=12,
    )

    story = []
    story.append(Paragraph("ICRA Results - Merged Report", title_style))
    story.append(Paragraph(f"Directory: {str(INPUT_DIR)}<br/>Files included: {len(files)}", meta_style))
    story.append(Spacer(1, 0.2 * inch))

    page_width, page_height = letter
    usable_width = page_width - doc.leftMargin - doc.rightMargin
    max_image_height = page_height - doc.topMargin - doc.bottomMargin - 2.0 * inch  # leave room for headings

    first_group = True
    for gname in sorted(groups.keys(), key=lambda s: s.lower()):
        if not first_group:
            story.append(PageBreak())
        first_group = False

        story.append(Paragraph(gname, group_style))
        story.append(Spacer(1, 0.1 * inch))

        for p in groups[gname]:
            ext = p.suffix.lower()
            size_kb = p.stat().st_size / 1024.0

            story.append(Paragraph(f"{p.name} <font color='grey' size='9'>({size_kb:.1f} KB)</font>", item_style))

            if ext in IMAGE_EXTS:
                # ReportLab Image auto-reads PNG/JPG; scale to fit page
                img = RLImage(str(p))
                img.hAlign = "LEFT"

                # Scale proportionally to fit usable area
                iw, ih = img.imageWidth, img.imageHeight
                if iw <= 0 or ih <= 0:
                    story.append(Paragraph("Could not read image dimensions.", styles["Normal"]))
                else:
                    scale = min(usable_width / iw, max_image_height / ih, 1.0)
                    img.drawWidth = iw * scale
                    img.drawHeight = ih * scale
                    story.append(img)

                story.append(Spacer(1, 0.15 * inch))

            elif ext in CSV_EXTS:
                data = safe_csv_read(p, MAX_CSV_ROWS, MAX_CSV_COLS)

                truncated_note = ""
                # Heuristic: if we hit max rows, likely truncated
                if len(data) >= MAX_CSV_ROWS:
                    truncated_note = f"<font color='grey' size='9'>Showing first {MAX_CSV_ROWS} rows (may be truncated).</font>"

                tbl = build_csv_table(data)
                block = [tbl]
                if truncated_note:
                    block.append(Spacer(1, 0.08 * inch))
                    block.append(Paragraph(truncated_note, styles["Normal"]))

                story.append(KeepTogether(block))
                story.append(Spacer(1, 0.2 * inch))

    doc.build(story)
    print(f"Created PDF: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()