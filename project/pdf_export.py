"""
pdf_export.py
=============
Institutional-grade equity research PDF generator.

Covers every section visible in the UI:
  1.  Cover Page                       – full-bleed, badge, meta grid
  2.  Table of Contents                – hyperlinked, auto-generated
  3.  Executive Summary & KPIs         – 8-card KPI row + company snapshot
  4.  Income Statement                 – historical table + 2 charts
  5.  Margin Analysis                  – multi-margin trend + commentary
  6.  Balance Sheet                    – dual-column snapshot + donut + trend
  7.  Cash Flow & FCF                  – bridge waterfall + FCF history
  8.  Financial Ratios — Profitability – all sub-ratios with benchmarks
  9.  Financial Ratios — Margins       – full margin set with interpretations
  10. Financial Ratios — Efficiency    – turnover + working capital metrics
  11. Financial Ratios — Liquidity     – current/quick/CCC + commentary
  12. Financial Ratios — Leverage      – debt ratios + Altman Z-Score gauge
  13. Growth & CAGR Analysis           – YoY table + CAGR bar chart + trajectory
  14. Valuation Multiples              – multiples table + EV bridge + PEG
  15. Portfolio Overview (optional)    – weights, allocation donut, performance KPIs
  16. Peer Comparison (optional)       – peer table + scatter
  17. Risk Analysis                    – volatility, beta, drawdown commentary
  18. Legal Disclaimer                 – full regulatory disclosures
"""

import io
import math
import datetime
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter
from matplotlib.patches import FancyBboxPatch

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm, inch
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak, Image, KeepTogether, NextPageTemplate,
    PageTemplate, Frame
)
from reportlab.platypus.flowables import Flowable
from reportlab.pdfgen import canvas as rl_canvas

# ─────────────────────────────────────────────────────────────────
# BRAND / DESIGN CONSTANTS
# ─────────────────────────────────────────────────────────────────
NAVY        = colors.HexColor("#0B1F3A")
NAVY_MID    = colors.HexColor("#1A3A5C")
NAVY_LIGHT  = colors.HexColor("#2D5282")
TEAL        = colors.HexColor("#0D7A8A")
TEAL_LIGHT  = colors.HexColor("#E6F4F6")
TEAL_MID    = colors.HexColor("#0F8FA0")
GOLD        = colors.HexColor("#C9A84C")
GOLD_LIGHT  = colors.HexColor("#FDF6E3")
GOLD_PALE   = colors.HexColor("#F5E6C8")
GREY_DARK   = colors.HexColor("#1F2937")
GREY_MID    = colors.HexColor("#6B7280")
GREY_LIGHT  = colors.HexColor("#F3F4F6")
GREY_LINE   = colors.HexColor("#E5E7EB")
WHITE       = colors.white
GREEN       = colors.HexColor("#059669")
GREEN_LIGHT = colors.HexColor("#D1FAE5")
RED         = colors.HexColor("#DC2626")
RED_LIGHT   = colors.HexColor("#FEE2E2")
AMBER       = colors.HexColor("#D97706")
AMBER_LIGHT = colors.HexColor("#FEF3C7")
SLATE       = colors.HexColor("#334155")
SLATE_LIGHT = colors.HexColor("#94A3B8")

PAGE_W, PAGE_H = A4
MARGIN   = 16 * mm
CONTENT_W = PAGE_W - 2 * MARGIN

# ─────────────────────────────────────────────────────────────────
# MATPLOTLIB THEME
# ─────────────────────────────────────────────────────────────────
CHART_COLORS = ["#0D7A8A", "#C9A84C", "#0B1F3A", "#059669", "#DC2626", "#7C3AED", "#EA580C", "#0891B2"]

MPL_STYLE = {
    "font.family":          "DejaVu Sans",
    "axes.facecolor":       "#FAFBFC",
    "figure.facecolor":     "white",
    "axes.edgecolor":       "#D1D5DB",
    "axes.grid":            True,
    "grid.color":           "#E5E7EB",
    "grid.linestyle":       "--",
    "grid.linewidth":       0.5,
    "axes.spines.top":      False,
    "axes.spines.right":    False,
    "axes.titlesize":       9,
    "axes.titleweight":     "bold",
    "axes.titlecolor":      "#1F2937",
    "axes.labelsize":       8,
    "xtick.labelsize":      7.5,
    "ytick.labelsize":      7.5,
    "legend.fontsize":      7.5,
    "legend.framealpha":    0.85,
    "legend.edgecolor":     "#E5E7EB",
}

def _mpl_to_img(fig, dpi=160):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    buf.seek(0)
    plt.close(fig)
    return buf

def _M(x, pos=None):
    if abs(x) >= 1e12: return f"${x/1e12:.1f}T"
    if abs(x) >= 1e9:  return f"${x/1e9:.1f}B"
    if abs(x) >= 1e6:  return f"${x/1e6:.0f}M"
    return f"${x:,.0f}"

def _Pct(x, pos=None): return f"{x:.1f}%"
def _X(x, pos=None):   return f"{x:.1f}x"

# ─────────────────────────────────────────────────────────────────
# VALUE FORMATTERS
# ─────────────────────────────────────────────────────────────────
def _fv(v, suffix="", prefix="", dec=1, scale=1e6, na="—"):
    try:
        f = float(v)
        if math.isnan(f) or math.isinf(f): return na
        f /= scale
        return f"{prefix}{f:,.{dec}f}{suffix}"
    except: return na

def _fp(v, dec=1, na="—"):
    try:
        f = float(v)
        if math.isnan(f) or math.isinf(f): return na
        return f"{f:.{dec}f}%"
    except: return na

def _fx(v, dec=2, na="—"):
    try:
        f = float(v)
        if math.isnan(f) or math.isinf(f): return na
        return f"{f:.{dec}f}x"
    except: return na

def _fi(v, na="—"):
    try:
        f = float(v)
        if math.isnan(f): return na
        return f"{int(f):,}"
    except: return na

def _color_sign(v, good_pos=True):
    try:
        f = float(v)
        if f > 0: return GREEN if good_pos else RED
        if f < 0: return RED if good_pos else GREEN
    except: pass
    return GREY_MID

def _growth_tag(v):
    try:
        f = float(str(v).replace("%",""))
        s = f"+{f:.1f}%" if f >= 0 else f"{f:.1f}%"
        return s
    except: return "—"

def _safe_series(hist, years):
    if hist is None or (hasattr(hist, "empty") and hist.empty):
        return [None] * len(years)
    result = []
    for y in years:
        found = None
        try:
            for idx in hist.index:
                if pd.to_datetime(idx).year == y:
                    found = float(hist.loc[idx])
                    break
        except: pass
        result.append(found)
    return result

def _years_from_hist(*histories, n=5):
    for h in histories:
        if h is not None and hasattr(h, "index") and not h.empty:
            ys = sorted(set(pd.to_datetime(i).year for i in h.index))
            return ys[-n:]
    yr = datetime.datetime.now().year
    return list(range(yr - n + 1, yr + 1))

# ─────────────────────────────────────────────────────────────────
# CUSTOM FLOWABLES
# ─────────────────────────────────────────────────────────────────
class SectionBanner(Flowable):
    """Full-width section header with icon accent bar."""
    def __init__(self, title, subtitle="", section_num=None, bg=NAVY, fg=WHITE, height=16*mm):
        super().__init__()
        self.title = title
        self.subtitle = subtitle
        self.section_num = section_num
        self.bg = bg; self.fg = fg
        self._height = height
        self.width = CONTENT_W

    def wrap(self, *a): return self.width, self._height

    def draw(self):
        c = self.canv
        # Background
        c.setFillColor(self.bg)
        c.roundRect(0, 0, self.width, self._height, 3, fill=1, stroke=0)
        # Gold accent bar
        c.setFillColor(GOLD)
        c.rect(0, 0, 4, self._height, fill=1, stroke=0)
        # Section number badge
        x_offset = 14
        if self.section_num:
            c.setFillColor(TEAL)
            c.roundRect(14, 4, 18, self._height - 8, 2, fill=1, stroke=0)
            c.setFillColor(WHITE)
            c.setFont("Helvetica-Bold", 7)
            c.drawCentredString(23, self._height / 2 - 3, str(self.section_num))
            x_offset = 40
        # Title
        c.setFillColor(self.fg)
        c.setFont("Helvetica-Bold", 10.5)
        c.drawString(x_offset, self._height - 12, self.title.upper())
        if self.subtitle:
            c.setFont("Helvetica", 7.5)
            c.setFillColor(colors.HexColor("#CBD5E1"))
            c.drawString(x_offset, 4.5, self.subtitle)


class MetricCard(Flowable):
    """KPI card with label, value, optional delta, optional teal top bar."""
    def __init__(self, label, value, delta=None, delta_good=True,
                 width=44*mm, height=22*mm, accent=TEAL, bg=GREY_LIGHT):
        super().__init__()
        self.label = label; self.value = value; self.delta = delta
        self.delta_good = delta_good
        self.width = width; self._height = height
        self.accent = accent; self.bg = bg

    def wrap(self, *a): return self.width, self._height

    def draw(self):
        c = self.canv
        # Shadow simulation (offset rect)
        c.setFillColor(colors.HexColor("#D1D5DB"))
        c.roundRect(1.5, -1.5, self.width - 2, self._height - 2, 3, fill=1, stroke=0)
        # Card body
        c.setFillColor(self.bg)
        c.roundRect(0, 0, self.width - 2, self._height - 2, 3, fill=1, stroke=0)
        # Accent top bar
        c.setFillColor(self.accent)
        c.rect(0, self._height - 4, self.width - 2, 4, fill=1, stroke=0)
        # Label
        c.setFillColor(GREY_MID)
        c.setFont("Helvetica", 6)
        c.drawString(5, self._height - 12, self.label.upper()[:22])
        # Value
        c.setFillColor(NAVY)
        c.setFont("Helvetica-Bold", 12)
        c.drawString(5, 8, str(self.value)[:14])
        # Delta
        if self.delta is not None:
            col = GREEN if self.delta_good else RED
            c.setFillColor(col)
            c.setFont("Helvetica-Bold", 6.5)
            c.drawRightString(self.width - 6, 8, str(self.delta))


class HLine(Flowable):
    def __init__(self, width=None, color=GREY_LINE, thickness=0.5, space_before=3, space_after=3):
        super().__init__()
        self._width = width or CONTENT_W
        self.color = color; self.thickness = thickness
        self.space_before = space_before; self.space_after = space_after

    def wrap(self, *a): return self._width, self.thickness + self.space_before + self.space_after

    def draw(self):
        self.canv.setStrokeColor(self.color)
        self.canv.setLineWidth(self.thickness)
        self.canv.line(0, self.space_after, self._width, self.space_after)


class ColorBar(Flowable):
    """A gradient-like horizontal bar showing a value on a scale (e.g., Z-Score)."""
    def __init__(self, value, lo, hi, label="", width=None, height=10*mm):
        super().__init__()
        self.value = value; self.lo = lo; self.hi = hi; self.label = label
        self.width = width or CONTENT_W * 0.6
        self._height = height

    def wrap(self, *a): return self.width, self._height

    def draw(self):
        c = self.canv
        bar_h = 6; bar_y = 2
        # Background track
        c.setFillColor(GREY_LIGHT)
        c.roundRect(0, bar_y, self.width, bar_h, 2, fill=1, stroke=0)
        # Colored fill
        try:
            frac = min(1, max(0, (float(self.value) - self.lo) / (self.hi - self.lo)))
        except: frac = 0
        fill_color = GREEN if frac > 0.6 else (AMBER if frac > 0.3 else RED)
        c.setFillColor(fill_color)
        fill_w = max(4, self.width * frac)
        c.roundRect(0, bar_y, fill_w, bar_h, 2, fill=1, stroke=0)
        # Pointer
        px = self.width * frac
        c.setFillColor(NAVY)
        c.circle(px, bar_y + bar_h / 2, 3, fill=1, stroke=0)
        # Label
        if self.label:
            c.setFont("Helvetica", 6)
            c.setFillColor(GREY_MID)
            c.drawString(0, bar_y + bar_h + 1.5, self.label)


class TwoColText(Flowable):
    """Two-column key-value pair in a single flowable row."""
    def __init__(self, left, right, width=None, height=8*mm, bold_right=False):
        super().__init__()
        self.left = left; self.right = right
        self.width = width or CONTENT_W; self._height = height
        self.bold_right = bold_right

    def wrap(self, *a): return self.width, self._height

    def draw(self):
        c = self.canv
        c.setFont("Helvetica", 7.5)
        c.setFillColor(GREY_DARK)
        c.drawString(0, 2, self.left)
        c.setFont("Helvetica-Bold" if self.bold_right else "Helvetica", 7.5)
        c.setFillColor(NAVY)
        c.drawRightString(self.width, 2, self.right)


# ─────────────────────────────────────────────────────────────────
# STYLES
# ─────────────────────────────────────────────────────────────────
def _styles():
    def S(name, **kw): return ParagraphStyle(name, **kw)
    return {
        # Cover
        "cover_ticker":  S("cTicker",  fontName="Helvetica-Bold", fontSize=32, textColor=WHITE, leading=38),
        "cover_name":    S("cName",    fontName="Helvetica-Bold", fontSize=20, textColor=WHITE, leading=26),
        "cover_sub":     S("cSub",     fontName="Helvetica",      fontSize=11, textColor=SLATE_LIGHT, leading=15),
        "cover_label":   S("cLabel",   fontName="Helvetica-Bold", fontSize=8,  textColor=GOLD,  leading=12),
        "cover_meta":    S("cMeta",    fontName="Helvetica",      fontSize=8,  textColor=SLATE_LIGHT, leading=11),
        # Section headers
        "h1":  S("h1",  fontName="Helvetica-Bold",    fontSize=14, textColor=NAVY,     spaceBefore=5, spaceAfter=4),
        "h2":  S("h2",  fontName="Helvetica-Bold",    fontSize=10, textColor=NAVY_MID, spaceBefore=4, spaceAfter=3),
        "h3":  S("h3",  fontName="Helvetica-Bold",    fontSize=8.5, textColor=TEAL,    spaceBefore=3, spaceAfter=2),
        "kicker": S("kicker", fontName="Helvetica-Bold", fontSize=6.5, textColor=TEAL, spaceBefore=6, spaceAfter=1),
        # Body
        "body":    S("body",    fontName="Helvetica",        fontSize=8.5, textColor=GREY_DARK, leading=12),
        "body_sm": S("bodySm",  fontName="Helvetica",        fontSize=7.5, textColor=GREY_DARK, leading=11),
        "caption": S("caption", fontName="Helvetica-Oblique", fontSize=6.5, textColor=GREY_MID,  spaceBefore=2),
        "disc":    S("disc",    fontName="Helvetica",        fontSize=6.5, textColor=GREY_MID,  leading=9, alignment=TA_JUSTIFY),
        # Table cells
        "th":      S("th",     fontName="Helvetica-Bold", fontSize=7,   textColor=WHITE,     alignment=TA_CENTER),
        "th_left": S("thL",    fontName="Helvetica-Bold", fontSize=7,   textColor=WHITE,     alignment=TA_LEFT),
        "td":      S("td",     fontName="Helvetica",      fontSize=7.5, textColor=GREY_DARK, alignment=TA_RIGHT),
        "td_l":    S("tdL",    fontName="Helvetica",      fontSize=7.5, textColor=GREY_DARK, alignment=TA_LEFT),
        "td_b":    S("tdB",    fontName="Helvetica-Bold", fontSize=7.5, textColor=NAVY,      alignment=TA_RIGHT),
        "td_bl":   S("tdBL",   fontName="Helvetica-Bold", fontSize=7.5, textColor=NAVY,      alignment=TA_LEFT),
        "td_green":S("tdG",    fontName="Helvetica-Bold", fontSize=7.5, textColor=GREEN,     alignment=TA_RIGHT),
        "td_red":  S("tdR",    fontName="Helvetica-Bold", fontSize=7.5, textColor=RED,       alignment=TA_RIGHT),
        "td_note": S("tdN",    fontName="Helvetica-Oblique", fontSize=6.5, textColor=GREY_MID, alignment=TA_LEFT),
        # TOC
        "toc1": S("toc1", fontName="Helvetica-Bold", fontSize=9,   textColor=NAVY,    spaceBefore=2, spaceAfter=1),
        "toc2": S("toc2", fontName="Helvetica",      fontSize=8,   textColor=GREY_DARK, leftIndent=10),
        # Footer
        "footer": S("foot", fontName="Helvetica", fontSize=7, textColor=GREY_MID, alignment=TA_CENTER),
    }


# ─────────────────────────────────────────────────────────────────
# SHARED TABLE STYLE
# ─────────────────────────────────────────────────────────────────
def _ts(header_rows=1, alt=True, compact=False):
    pad = 3 if compact else 4
    cmds = [
        ("BACKGROUND",   (0, 0), (-1, header_rows - 1), NAVY),
        ("TEXTCOLOR",    (0, 0), (-1, header_rows - 1), WHITE),
        ("FONTNAME",     (0, 0), (-1, header_rows - 1), "Helvetica-Bold"),
        ("FONTSIZE",     (0, 0), (-1, -1), 7 if compact else 7.5),
        ("ALIGN",        (0, 0), (-1, -1), "CENTER"),
        ("ALIGN",        (0, header_rows), (0, -1), "LEFT"),
        ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",   (0, 0), (-1, -1), pad),
        ("BOTTOMPADDING",(0, 0), (-1, -1), pad),
        ("LEFTPADDING",  (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("LINEBELOW",    (0, 0), (-1, 0), 1.5, TEAL),
        ("LINEBELOW",    (0, 1), (-1, -2), 0.25, GREY_LINE),
        ("LINEBELOW",    (0, -1), (-1, -1), 1, NAVY),
        ("GRID",         (0, 0), (-1, -1), 0.1, GREY_LINE),
    ]
    if alt:
        cmds.append(("ROWBACKGROUND", (0, header_rows), (-1, -1), [GREY_LIGHT, WHITE]))
    return TableStyle(cmds)


# ─────────────────────────────────────────────────────────────────
# PAGE HEADERS & FOOTERS
# ─────────────────────────────────────────────────────────────────
def _make_templates(ticker, company_name, report_date):
    """Returns (on_first_page, on_later_pages) canvas callbacks."""

    def on_cover(canvas, doc):
        pass  # Cover handles its own drawing

    def on_page(canvas, doc):
        canvas.saveState()
        pw, ph = A4
        m = MARGIN
        # Top teal rule
        canvas.setStrokeColor(TEAL)
        canvas.setLineWidth(1.2)
        canvas.line(m, ph - 12*mm, pw - m, ph - 12*mm)
        # Header left: company
        canvas.setFont("Helvetica-Bold", 7)
        canvas.setFillColor(NAVY)
        canvas.drawString(m, ph - 9*mm, f"{ticker}  |  {company_name}")
        # Header right: classification
        canvas.setFont("Helvetica", 6.5)
        canvas.setFillColor(GREY_MID)
        canvas.drawRightString(pw - m, ph - 9*mm, "CONFIDENTIAL — INSTITUTIONAL EQUITY RESEARCH")
        # Bottom rule
        canvas.setStrokeColor(GREY_LINE)
        canvas.setLineWidth(0.4)
        canvas.line(m, 11*mm, pw - m, 11*mm)
        # Footer left
        canvas.setFont("Helvetica", 6)
        canvas.setFillColor(GREY_MID)
        canvas.drawString(m, 7*mm, f"Report Date: {report_date}  |  Source: SEC EDGAR & Yahoo Finance")
        # Footer right: page number
        canvas.setFont("Helvetica-Bold", 6.5)
        canvas.setFillColor(NAVY)
        canvas.drawRightString(pw - m, 7*mm, f"Page {doc.page}")
        canvas.restoreState()

    return on_cover, on_page


# ─────────────────────────────────────────────────────────────────
# CHART BUILDERS
# ─────────────────────────────────────────────────────────────────
def _bar_line(years, bars, lines, bar_lbl, line_lbl,
              bar_col=CHART_COLORS[0], line_col=CHART_COLORS[1],
              y1_fmt=_M, y2_fmt=_Pct, title="", figsize=(6.8, 2.7)):
    with plt.rc_context(MPL_STYLE):
        fig, ax1 = plt.subplots(figsize=figsize)
        x = np.arange(len(years))
        bars_c = [b / 1e6 if b else 0 for b in bars]
        ax1.bar(x, bars_c, width=0.48, color=bar_col, alpha=0.82, zorder=3, label=bar_lbl)
        ax1.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"${v:,.0f}M"))
        ax1.set_xticks(x); ax1.set_xticklabels(years)
        ax1.set_ylabel(bar_lbl, fontsize=7.5, color=bar_col)
        ax1.tick_params(axis="y", labelcolor=bar_col, labelsize=7)
        clean_lines = []
        for v in lines:
            try:
                f = float(v)
                clean_lines.append(None if math.isnan(f) else f)
            except: clean_lines.append(None)
        if any(v is not None for v in clean_lines):
            ax2 = ax1.twinx()
            xi = [xi for xi, v in zip(x, clean_lines) if v is not None]
            yi = [v for v in clean_lines if v is not None]
            ax2.plot(xi, yi, color=line_col, marker="o", linewidth=2, markersize=4.5, zorder=4, label=line_lbl)
            ax2.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:.1f}%"))
            ax2.set_ylabel(line_lbl, fontsize=7.5, color=line_col)
            ax2.tick_params(axis="y", labelcolor=line_col, labelsize=7)
            ax2.spines["right"].set_visible(True)
            ax2.spines["right"].set_color("#D1D5DB")
        ax1.set_title(title, pad=6)
        fig.tight_layout()
        return fig


def _multi_line(years, series, title="", y_fmt=_Pct, figsize=(6.8, 2.7)):
    with plt.rc_context(MPL_STYLE):
        fig, ax = plt.subplots(figsize=figsize)
        x = np.arange(len(years))
        for i, (lbl, vals) in enumerate(series.items()):
            clean = []
            for v in vals:
                try: clean.append(None if math.isnan(float(v)) else float(v))
                except: clean.append(None)
            xi = [xi for xi, v in zip(x, clean) if v is not None]
            yi = [v for v in clean if v is not None]
            if xi:
                ax.plot(xi, yi, color=CHART_COLORS[i % len(CHART_COLORS)],
                        marker="o", linewidth=2, markersize=4.5, label=lbl, zorder=3)
        ax.yaxis.set_major_formatter(FuncFormatter(y_fmt))
        ax.set_xticks(x); ax.set_xticklabels(years)
        if len(series) > 1: ax.legend(loc="best")
        ax.set_title(title, pad=6)
        fig.tight_layout()
        return fig


def _waterfall(labels, values, title="", figsize=(5.5, 2.7)):
    with plt.rc_context(MPL_STYLE):
        fig, ax = plt.subplots(figsize=figsize)
        cols = [CHART_COLORS[0] if v >= 0 else "#DC2626" for v in values]
        x = np.arange(len(labels))
        bars = ax.bar(x, values, color=cols, alpha=0.85, width=0.5, zorder=3)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: _M(v * 1e6)))
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7.5, rotation=10, ha="right")
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(abs(v) for v in values)*0.02,
                    _M(val * 1e6), ha="center", va="bottom", fontsize=6.5, color="#374151")
        ax.set_title(title, pad=6)
        fig.tight_layout()
        return fig


def _donut(labels, values, title="", figsize=(3.4, 2.7), colors_list=None):
    with plt.rc_context(MPL_STYLE):
        fig, ax = plt.subplots(figsize=figsize)
        cols = (colors_list or CHART_COLORS)[:len(labels)]
        wedges, _, autotexts = ax.pie(
            values, labels=None, colors=cols, autopct="%1.0f%%", startangle=90,
            wedgeprops={"width": 0.52, "edgecolor": "white", "linewidth": 1.5},
            pctdistance=0.75, textprops={"fontsize": 6.5}
        )
        ax.legend(labels, loc="center left", bbox_to_anchor=(0.95, 0.5), fontsize=7, framealpha=0.9)
        ax.set_title(title, pad=6)
        fig.tight_layout()
        return fig


def _grouped_bar(years, series, title="", figsize=(6.8, 2.7), pct=False):
    with plt.rc_context(MPL_STYLE):
        fig, ax = plt.subplots(figsize=figsize)
        n = len(series)
        x = np.arange(len(years))
        w = 0.7 / n
        for i, (lbl, vals) in enumerate(series.items()):
            clean = [float(v) if v is not None and not math.isnan(float(v if v is not None else float("nan"))) else 0 for v in vals]
            ax.bar(x + i * w - (n - 1) * w / 2, clean, width=w, color=CHART_COLORS[i % len(CHART_COLORS)], alpha=0.85, label=lbl, zorder=3)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:.1f}%" if pct else f"${v:,.0f}M"))
        ax.set_xticks(x); ax.set_xticklabels(years)
        ax.legend(loc="best")
        ax.set_title(title, pad=6)
        fig.tight_layout()
        return fig


def _gauge(value, lo, hi, label="", bands=None, figsize=(3.2, 2.2)):
    """Semicircle gauge chart."""
    with plt.rc_context(MPL_STYLE):
        fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=False))
        ax.set_xlim(-1.2, 1.2); ax.set_ylim(-0.2, 1.2)
        ax.axis("off")
        # Background arc
        theta = np.linspace(np.pi, 0, 100)
        ax.fill_between(np.cos(theta), np.sin(theta) * 0, np.sin(theta), alpha=0.08, color="grey")
        # Color bands
        if bands:
            for blo, bhi, bcol in bands:
                blo_f = max(0, (blo - lo) / (hi - lo))
                bhi_f = min(1, (bhi - lo) / (hi - lo))
                t_lo = np.pi * (1 - blo_f)
                t_hi = np.pi * (1 - bhi_f)
                t = np.linspace(t_lo, t_hi, 50) if t_lo > t_hi else np.linspace(t_hi, t_lo, 50)
                ax.fill_between(np.cos(t), np.sin(t) * 0.6, np.sin(t), alpha=0.35, color=bcol)
        # Needle
        try:
            frac = min(1, max(0, (float(value) - lo) / (hi - lo)))
        except: frac = 0
        ang = np.pi * (1 - frac)
        ax.annotate("", xy=(np.cos(ang) * 0.78, np.sin(ang) * 0.78), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color=NAVY.hexval()[1:] if hasattr(NAVY, "hexval") else "#0B1F3A", lw=2.5))
        ax.plot(0, 0, "o", color="#0B1F3A", markersize=8, zorder=5)
        # Value text
        try: val_str = f"{float(value):.2f}"
        except: val_str = "N/A"
        ax.text(0, 0.15, val_str, ha="center", va="center", fontsize=12, fontweight="bold", color="#0B1F3A")
        ax.text(0, -0.05, label, ha="center", va="center", fontsize=7, color="grey")
        ax.text(-1.1, -0.05, f"{lo}", ha="center", fontsize=6.5, color="grey")
        ax.text( 1.1, -0.05, f"{hi}", ha="center", fontsize=6.5, color="grey")
        fig.tight_layout(pad=0.5)
        return fig


# ─────────────────────────────────────────────────────────────────
# SECTION 1: COVER PAGE
# ─────────────────────────────────────────────────────────────────
def _build_cover(story, styles, ticker, company_name, industry, sector,
                 market_cap_str, report_date, prepared_by="Institutional Equity Research"):

    class CoverPage(Flowable):
        def wrap(self, aw, ah): return CONTENT_W, ah

        def draw(self):
            c = self.canv
            pw, ph = PAGE_W, PAGE_H
            m = MARGIN
            c.saveState()
            c.translate(-m, -15*mm)  # full-bleed from page origin

            # Background gradient simulation
            c.setFillColor(NAVY)
            c.rect(0, 0, pw, ph, fill=1, stroke=0)

            # Diagonal accent panel top-right
            c.setFillColor(NAVY_MID)
            p = c.beginPath()
            p.moveTo(pw * 0.50, ph)
            p.lineTo(pw, ph)
            p.lineTo(pw, ph * 0.30)
            p.close()
            c.drawPath(p, fill=1, stroke=0)

            # Secondary diagonal bottom-left
            c.setFillColor(colors.HexColor("#0A172A"))
            p2 = c.beginPath()
            p2.moveTo(0, 0)
            p2.lineTo(pw * 0.45, 0)
            p2.lineTo(0, ph * 0.20)
            p2.close()
            c.drawPath(p2, fill=1, stroke=0)

            # Gold vertical accent
            c.setFillColor(GOLD)
            c.rect(m, ph * 0.22, 4.5, ph * 0.60, fill=1, stroke=0)

            # Teal horizontal rule
            c.setStrokeColor(TEAL)
            c.setLineWidth(1.5)
            c.line(m + 14, ph * 0.635, pw * 0.68, ph * 0.635)

            # Ticker badge
            bx, by, bw, bh = m + 14, ph * 0.72, 100, 36
            c.setFillColor(TEAL)
            c.roundRect(bx, by, bw, bh, 5, fill=1, stroke=0)
            c.setFillColor(WHITE)
            c.setFont("Helvetica-Bold", 20)
            c.drawCentredString(bx + bw / 2, by + 12, ticker)
            c.setFont("Helvetica", 6.5)
            c.setFillColor(colors.HexColor("#B2EBF2"))
            c.drawCentredString(bx + bw / 2, by + 4, "NYSE / NASDAQ")

            # Company name
            c.setFillColor(WHITE)
            cname = company_name
            while c.stringWidth(cname, "Helvetica-Bold", 22) > pw * 0.62 - m - 20 and len(cname) > 6:
                cname = cname[:-1]
            if cname != company_name: cname = cname.rstrip()[:-3] + "..."
            c.setFont("Helvetica-Bold", 22)
            c.drawString(m + 14, ph * 0.645, cname)

            # Sector & industry
            c.setFont("Helvetica", 11)
            c.setFillColor(colors.HexColor("#94A3B8"))
            c.drawString(m + 14, ph * 0.60, f"{sector}  ·  {industry}")

            # Report type label
            c.setFont("Helvetica-Bold", 8.5)
            c.setFillColor(GOLD)
            c.drawString(m + 14, ph * 0.555, "INSTITUTIONAL EQUITY RESEARCH REPORT")

            # Subtitle
            c.setFont("Helvetica", 9)
            c.setFillColor(colors.HexColor("#64748B"))
            c.drawString(m + 14, ph * 0.53, "Comprehensive Financial Analysis & Valuation")

            # Meta info grid
            meta_y = ph * 0.30
            meta_items = [
                ("MARKET CAP",    market_cap_str),
                ("REPORT DATE",   report_date),
                ("PREPARED BY",   prepared_by[:22]),
                ("CLASSIFICATION","CONFIDENTIAL"),
            ]
            col_w = (pw - 2 * m - 14) / len(meta_items)
            for i, (lbl, val) in enumerate(meta_items):
                mx = m + 14 + i * col_w
                c.setFillColor(colors.HexColor("#1E3A5F"))
                c.roundRect(mx, meta_y - 2, col_w - 10, 38, 4, fill=1, stroke=0)
                c.setFont("Helvetica", 6)
                c.setFillColor(SLATE_LIGHT)
                c.drawString(mx + 7, meta_y + 28, lbl)
                c.setFont("Helvetica-Bold", 8.5)
                c.setFillColor(WHITE)
                c.drawString(mx + 7, meta_y + 13, str(val))

            # Decorative dots pattern (bottom right)
            c.setFillColor(colors.HexColor("#1A3A5C"))
            for row in range(6):
                for col in range(8):
                    cx_ = pw - m - 5 - col * 14
                    cy_ = ph * 0.15 + row * 14
                    c.circle(cx_, cy_, 2.5, fill=1, stroke=0)

            # Bottom disclaimer strip
            c.setFillColor(colors.HexColor("#06111F"))
            c.rect(0, 0, pw, 22*mm, fill=1, stroke=0)
            c.setFillColor(GOLD)
            c.rect(0, 21*mm, pw, 1, fill=1, stroke=0)
            c.setFont("Helvetica", 6)
            c.setFillColor(colors.HexColor("#64748B"))
            c.drawCentredString(pw / 2, 14*mm,
                "This document is prepared for informational purposes only. It does not constitute investment advice, "
                "solicitation, or an offer to buy or sell any security.")
            c.drawCentredString(pw / 2, 8*mm,
                "Financial data sourced from SEC EDGAR and Yahoo Finance. Past performance is not indicative of future results.")
            c.drawCentredString(pw / 2, 3.5*mm,
                f"© {datetime.datetime.now().year}  SEC Equity Analyzer Platform  |  {report_date}")
            c.restoreState()

    story.append(CoverPage())
    story.append(PageBreak())


# ─────────────────────────────────────────────────────────────────
# SECTION 2: TABLE OF CONTENTS
# ─────────────────────────────────────────────────────────────────
def _build_toc(story, styles):
    story.append(SectionBanner("Table of Contents", "Report Navigation Guide", section_num=None, bg=NAVY_MID))
    story.append(Spacer(1, 5*mm))

    toc_items = [
        ("01", "Executive Summary & Key Performance Indicators",   "03"),
        ("02", "Income Statement & Revenue Analysis",              "04"),
        ("03", "Margin Analysis",                                   "05"),
        ("04", "Balance Sheet & Capital Structure",                "06"),
        ("05", "Cash Flow & Free Cash Flow Analysis",              "07"),
        ("06", "Financial Ratios — Profitability",                 "08"),
        ("07", "Financial Ratios — Margins & Efficiency",          "09"),
        ("08", "Financial Ratios — Liquidity",                     "10"),
        ("09", "Financial Ratios — Leverage & Solvency",          "11"),
        ("10", "Growth & CAGR Analysis",                           "12"),
        ("11", "Valuation Multiples",                              "13"),
        ("12", "Risk Analysis",                                    "14"),
        ("13", "Legal Disclaimer & Disclosures",                   "15"),
    ]

    rows = [[Paragraph(n, styles["td_b"]),
             Paragraph(title, styles["td_l"]),
             Paragraph(pg, styles["td"])] for n, title, pg in toc_items]

    tbl = Table(rows, colWidths=[CONTENT_W * 0.08, CONTENT_W * 0.80, CONTENT_W * 0.12])
    tbl.setStyle(TableStyle([
        ("FONTSIZE",     (0, 0), (-1, -1), 8),
        ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING",  (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING",   (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 5),
        ("LINEBELOW",    (0, 0), (-1, -2), 0.3, GREY_LINE),
        ("LINEBELOW",    (0, -1), (-1, -1), 1, TEAL),
        ("ROWBACKGROUND",(0, 0), (-1, -1), [GREY_LIGHT, WHITE]),
        ("TEXTCOLOR",    (0, 0), (0, -1), TEAL),
        ("TEXTCOLOR",    (2, 0), (2, -1), GREY_MID),
    ]))
    story.append(tbl)
    story.append(PageBreak())


# ─────────────────────────────────────────────────────────────────
# SECTION 3: EXECUTIVE SUMMARY
# ─────────────────────────────────────────────────────────────────
def _build_executive_summary(story, styles, summary, metadata, ticker,
                              ltm_data, balance_data):
    story.append(SectionBanner("Executive Summary", "Key Performance Indicators at a Glance", section_num="01"))
    story.append(Spacer(1, 4*mm))

    # ── KPI Cards row ──
    revenue   = _fv(ltm_data.get("revenue"),    suffix="M",   scale=1e6, dec=0)
    ebitda    = _fv(ltm_data.get("ebitda"),      suffix="M",   scale=1e6, dec=0)
    net_inc   = _fv(ltm_data.get("net_income"),  suffix="M",   scale=1e6, dec=0)
    mcap      = _fv(metadata.get("market_cap"),  suffix="B",   scale=1e9, dec=2)
    gm        = _fp(summary.get("Gross Margin %"))
    roic      = _fp(summary.get("ROIC %"))
    pe        = _fx(metadata.get("pe_ltm"))
    fcf_yield = _fp(summary.get("FCF Yield %"))
    ev_ebitda = _fx(summary.get("EV/EBITDA"))
    net_margin= _fp(summary.get("Net Margin %"))
    roe       = _fp(summary.get("ROE %"))
    altman_z  = _fv(summary.get("Altman Z-Score"), scale=1, dec=2)

    kpis = [
        ("Revenue (LTM)",   revenue,    None),
        ("EBITDA (LTM)",    ebitda,     None),
        ("Net Income (LTM)",net_inc,    None),
        ("Market Cap",      mcap,       None),
        ("Gross Margin",    gm,         None),
        ("ROIC",            roic,       None),
        ("P/E (LTM)",       pe,         None),
        ("FCF Yield",       fcf_yield,  None),
        ("EV/EBITDA",       ev_ebitda,  None),
        ("Net Margin",      net_margin, None),
        ("ROE",             roe,        None),
        ("Altman Z-Score",  altman_z,   None),
    ]

    n_cols = 6
    card_w = CONTENT_W / n_cols - 2*mm
    card_h = 22*mm
    accents = [TEAL, GOLD, TEAL, GOLD, TEAL, GOLD, TEAL, GOLD, TEAL, GOLD, TEAL, NAVY_MID]
    rows_cards = []
    for chunk_start in range(0, len(kpis), n_cols):
        chunk = kpis[chunk_start:chunk_start + n_cols]
        row = [MetricCard(lbl, val, width=card_w, height=card_h,
                          accent=accents[(chunk_start + i) % len(accents)])
               for i, (lbl, val, _) in enumerate(chunk)]
        rows_cards.append(row)

    card_tbl = Table(rows_cards, colWidths=[card_w + 2*mm] * n_cols)
    card_tbl.setStyle(TableStyle([
        ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",    (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING",   (0, 0), (-1, -1), 2),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 2),
    ]))
    story.append(card_tbl)
    story.append(Spacer(1, 5*mm))

    # ── Company Snapshot table ──
    story.append(Paragraph("Company Snapshot", styles["h2"]))

    snap_items = [
        ("Company Name",        metadata.get("name", ticker)),
        ("Ticker Symbol",       ticker),
        ("Exchange",            metadata.get("exchange", "—")),
        ("Sector",              metadata.get("sector", "—")),
        ("Industry",            metadata.get("industry", "—")),
        ("Employees",           _fi(metadata.get("employees")) if metadata.get("employees") else "—"),
        ("LTM Revenue",         _fv(ltm_data.get("revenue"),    suffix="M", scale=1e6)),
        ("LTM EBITDA",          _fv(ltm_data.get("ebitda"),     suffix="M", scale=1e6)),
        ("LTM Net Income",      _fv(ltm_data.get("net_income"), suffix="M", scale=1e6)),
        ("LTM OCF",             _fv(ltm_data.get("ocf"),        suffix="M", scale=1e6)),
        ("Total Assets",        _fv(balance_data.get("total_assets"), suffix="M", scale=1e6)),
        ("Total Debt",          _fv(balance_data.get("debt"),         suffix="M", scale=1e6)),
        ("Cash & Equivalents",  _fv(balance_data.get("cash"),         suffix="M", scale=1e6)),
        ("Stockholders' Equity",_fv(balance_data.get("equity"),       suffix="M", scale=1e6)),
    ]

    half = len(snap_items) // 2
    col1, col2 = snap_items[:half], snap_items[half:]

    def _snap_rows(items):
        hdr = [Paragraph("METRIC", styles["th"]), Paragraph("VALUE", styles["th"])]
        rows = [hdr]
        for k, v in items:
            rows.append([Paragraph(k, styles["td_l"]), Paragraph(str(v), styles["td_b"])])
        return rows

    cw = CONTENT_W / 2 - 4*mm
    t1 = Table(_snap_rows(col1), colWidths=[cw * 0.58, cw * 0.42])
    t2 = Table(_snap_rows(col2), colWidths=[cw * 0.58, cw * 0.42])
    for t in (t1, t2): t.setStyle(_ts())
    outer = Table([[t1, Spacer(8*mm, 1), t2]], colWidths=[cw, 8*mm, cw])
    outer.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")]))
    story.append(outer)
    story.append(Spacer(1, 4*mm))

    # ── Business Description (if available) ──
    desc = metadata.get("description") or metadata.get("longBusinessSummary") or ""
    if desc and len(desc) > 40:
        story.append(Paragraph("Business Description", styles["h2"]))
        story.append(Paragraph(desc[:600] + ("..." if len(desc) > 600 else ""), styles["body_sm"]))

    story.append(PageBreak())


# ─────────────────────────────────────────────────────────────────
# SECTION 4: INCOME STATEMENT
# ─────────────────────────────────────────────────────────────────
def _build_income_statement(story, styles, ltm_data, revenue_h, gross_h,
                             ebit_h, ebitda_h, net_income_h, summary=None):
    if summary is None: summary = {}
    story.append(SectionBanner("Income Statement & Revenue Analysis",
                                "Historical Trends · LTM Performance", section_num="02"))
    story.append(Spacer(1, 4*mm))

    years = _years_from_hist(revenue_h, ebitda_h, net_income_h, n=5)
    yr_lbl = [f"FY{str(y)[2:]}" for y in years]

    rev_v  = _safe_series(revenue_h,   years)
    gp_v   = _safe_series(gross_h,     years)
    ebit_v = _safe_series(ebit_h,      years)
    ebd_v  = _safe_series(ebitda_h,    years)
    ni_v   = _safe_series(net_income_h,years)

    def _pct_series(num, den):
        out = []
        for n, d in zip(num, den):
            try: out.append(float(n) / float(d) * 100)
            except: out.append(None)
        return out

    gm_v    = _pct_series(gp_v,  rev_v)
    ebdm_v  = _pct_series(ebd_v, rev_v)
    nim_v   = _pct_series(ni_v,  rev_v)
    ebitm_v = _pct_series(ebit_v,rev_v)

    # Chart 1: Revenue + Gross Margin
    fig1 = _bar_line(yr_lbl, rev_v, gm_v, "Revenue ($M)", "Gross Margin (%)",
                     title="Revenue & Gross Margin Trend")
    img1 = _mpl_to_img(fig1)

    # Chart 2: EBITDA + Net margin
    fig2 = _bar_line(yr_lbl, ebd_v, nim_v, "EBITDA ($M)", "Net Margin (%)",
                     bar_col=CHART_COLORS[1], line_col=CHART_COLORS[4],
                     title="EBITDA & Net Income Margin")
    img2 = _mpl_to_img(fig2)

    cw = CONTENT_W / 2 - 2*mm
    ch = cw * 0.44
    chart_row = Table([[Image(img1, width=cw, height=ch), Image(img2, width=cw, height=ch)]],
                      colWidths=[cw + 2*mm, cw + 2*mm])
    chart_row.setStyle(TableStyle([("ALIGN",(0,0),(-1,-1),"CENTER"), ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
                                   ("LEFTPADDING",(0,0),(-1,-1),1), ("RIGHTPADDING",(0,0),(-1,-1),1)]))
    story.append(chart_row)
    story.append(Spacer(1, 4*mm))

    # Historical income table
    story.append(Paragraph("Historical Income Statement Summary ($M)", styles["h2"]))

    hdr = ["", *yr_lbl, "LTM"]
    rows = [
        ("Revenue",      rev_v,  ltm_data.get("revenue"),         False),
        ("Gross Profit", gp_v,   ltm_data.get("gross_profit"),    False),
        ("EBITDA",       ebd_v,  ltm_data.get("ebitda"),          False),
        ("EBIT",         ebit_v, ltm_data.get("operating_income"),False),
        ("Net Income",   ni_v,   ltm_data.get("net_income"),      False),
        ("—",            None,   None, None),
        ("Gross Margin %",   gm_v,   None, True),
        ("EBITDA Margin %",  ebdm_v, None, True),
        ("EBIT Margin %",    ebitm_v,None, True),
        ("Net Margin %",     nim_v,  None, True),
    ]

    def _mk_row(label, hist, ltm, is_pct):
        if label == "—":
            return [Paragraph("", styles["td"])] * len(hdr)
        def _f(v):
            if v is None: return "—"
            try:
                f = float(v)
                if math.isnan(f): return "—"
                return f"{f:.1f}%" if is_pct else f"${f/1e6:,.0f}M"
            except: return "—"
        cells = [Paragraph(label, styles["td_bl"] if is_pct else styles["td_l"])]
        cells += [Paragraph(_f(v), styles["td"]) for v in (hist or [])]
        cells.append(Paragraph(_f(ltm) if ltm is not None else "—", styles["td_b"]))
        return cells

    cw0 = CONTENT_W * 0.22
    cw_n = (CONTENT_W - cw0) / len(hdr)
    tbl_data = [[Paragraph(h, styles["th"] if h else styles["th"]) for h in hdr]]
    for label, hist, ltm, is_pct in rows:
        tbl_data.append(_mk_row(label, hist, ltm, is_pct))

    inc_tbl = Table(tbl_data, colWidths=[cw0] + [cw_n] * (len(hdr) - 1), repeatRows=1)
    inc_tbl.setStyle(_ts())
    story.append(inc_tbl)

    # ── Section 3: Margin Analysis ──
    story.append(PageBreak())
    story.append(SectionBanner("Margin Analysis", "Profitability Trends Across All Margin Categories", section_num="03"))
    story.append(Spacer(1, 4*mm))

    margin_series = {}
    if any(v is not None for v in gm_v):    margin_series["Gross Margin"]   = gm_v
    if any(v is not None for v in ebdm_v):  margin_series["EBITDA Margin"]  = ebdm_v
    if any(v is not None for v in ebitm_v): margin_series["EBIT Margin"]    = ebitm_v
    if any(v is not None for v in nim_v):   margin_series["Net Margin"]     = nim_v

    if margin_series:
        fig_m = _multi_line(yr_lbl, margin_series,
                            title="All Margin Trends (%)", y_fmt=_Pct, figsize=(7.5, 3.0))
        img_m = _mpl_to_img(fig_m)
        story.append(Image(img_m, width=CONTENT_W * 0.92, height=CONTENT_W * 0.92 * 0.40))
        story.append(Spacer(1, 3*mm))

    # Margin commentary table
    story.append(Paragraph("Margin Summary", styles["h2"]))
    margin_tbl_data = [
        [Paragraph(h, styles["th"]) for h in ["METRIC", "LTM VALUE", "BENCHMARK", "INTERPRETATION"]],
        [Paragraph("Gross Margin",   styles["td_l"]), Paragraph(_fp(summary.get("Gross Margin %")),   styles["td_b"]),
         Paragraph("40–70%+",        styles["td"]),   Paragraph("Revenue minus COGS, pre-operating expense", styles["td_note"])],
        [Paragraph("EBITDA Margin",  styles["td_l"]), Paragraph(_fp(summary.get("EBITDA Margin %")),  styles["td_b"]),
         Paragraph("> 15%",          styles["td"]),   Paragraph("Operating cash profitability proxy", styles["td_note"])],
        [Paragraph("EBIT Margin",    styles["td_l"]), Paragraph(_fp(summary.get("EBIT Margin %")),    styles["td_b"]),
         Paragraph("> 10%",          styles["td"]),   Paragraph("After D&A; true operating profit", styles["td_note"])],
        [Paragraph("Net Margin",     styles["td_l"]), Paragraph(_fp(summary.get("Net Margin %")),     styles["td_b"]),
         Paragraph("> 5%",           styles["td"]),   Paragraph("Bottom-line earnings efficiency", styles["td_note"])],
        [Paragraph("FCF Margin",     styles["td_l"]), Paragraph(_fp(summary.get("LFCF Margin %")),    styles["td_b"]),
         Paragraph("> 10%",          styles["td"]),   Paragraph("Cash available to equity holders", styles["td_note"])],
        [Paragraph("SG&A Margin",    styles["td_l"]), Paragraph(_fp(summary.get("SG&A Margin %")),    styles["td_b"]),
         Paragraph("< 30%",          styles["td"]),   Paragraph("Sales & admin as % of revenue", styles["td_note"])],
        [Paragraph("R&D Margin",     styles["td_l"]), Paragraph(_fp(summary.get("R&D Margin %")),     styles["td_b"]),
         Paragraph("Varies",         styles["td"]),   Paragraph("Innovation investment intensity", styles["td_note"])],
        [Paragraph("CapEx % Rev",    styles["td_l"]), Paragraph(_fp(summary.get("CapEx % Revenue")),  styles["td_b"]),
         Paragraph("< 10%",          styles["td"]),   Paragraph("Capital intensity relative to sales", styles["td_note"])],
    ]
    m_tbl = Table(margin_tbl_data, colWidths=[CONTENT_W*0.22, CONTENT_W*0.13, CONTENT_W*0.13, CONTENT_W*0.48],
                  repeatRows=1)
    m_tbl.setStyle(_ts())
    story.append(m_tbl)
    story.append(PageBreak())


# ─────────────────────────────────────────────────────────────────
# SECTION 5: BALANCE SHEET
# ─────────────────────────────────────────────────────────────────
def _build_balance_sheet(story, styles, balance_data, assets_h, equity_h, liab_h):
    story.append(SectionBanner("Balance Sheet & Capital Structure",
                                "Asset Quality · Solvency · Capital Allocation", section_num="04"))
    story.append(Spacer(1, 4*mm))

    bd = balance_data
    story.append(Paragraph("Balance Sheet Snapshot (Latest Quarter, $M)", styles["h2"]))

    def _bv(k): return _fv(bd.get(k), suffix="M", scale=1e6)

    bs_rows = [
        [Paragraph("ASSETS", styles["th_left"]),   Paragraph("$M", styles["th"]),
         Paragraph("LIABILITIES & EQUITY", styles["th_left"]), Paragraph("$M", styles["th"])],
        [Paragraph("Cash & Equivalents",   styles["td_l"]), Paragraph(_bv("cash"),               styles["td_b"]),
         Paragraph("Current Liabilities",  styles["td_l"]), Paragraph(_bv("current_liabilities"),styles["td_b"])],
        [Paragraph("Accounts Receivable",  styles["td_l"]), Paragraph(_bv("accounts_receivable"),styles["td"]),
         Paragraph("Short-Term Debt",      styles["td_l"]), Paragraph(_bv("short_term_debt"),     styles["td"])],
        [Paragraph("Inventory",            styles["td_l"]), Paragraph(_bv("inventory"),           styles["td"]),
         Paragraph("Long-Term Debt",       styles["td_l"]), Paragraph(_bv("long_term_debt"),      styles["td"])],
        [Paragraph("Current Assets",       styles["td_l"]), Paragraph(_bv("current_assets"),      styles["td_b"]),
         Paragraph("Total Debt",           styles["td_l"]), Paragraph(_bv("debt"),                styles["td_b"])],
        [Paragraph("Net PP&E",             styles["td_l"]), Paragraph(_bv("ppe"),                 styles["td"]),
         Paragraph("Retained Earnings",    styles["td_l"]), Paragraph(_bv("retained_earnings"),   styles["td"])],
        [Paragraph("Goodwill & Intangibles",styles["td_l"]),Paragraph(_bv("goodwill"),            styles["td"]),
         Paragraph("Additional Paid-In Cap",styles["td_l"]),Paragraph(_bv("additional_paid_in"),  styles["td"])],
        [Paragraph("TOTAL ASSETS",         styles["td_bl"]),Paragraph(_bv("total_assets"),        styles["td_b"]),
         Paragraph("TOTAL EQUITY",         styles["td_bl"]),Paragraph(_bv("equity"),              styles["td_b"])],
    ]

    half_w = CONTENT_W / 2 - 3*mm
    bs_tbl = Table(bs_rows, colWidths=[half_w * 0.60, half_w * 0.40, half_w * 0.60, half_w * 0.40])
    bs_tbl.setStyle(_ts())
    story.append(bs_tbl)
    story.append(Spacer(1, 4*mm))

    # Capital structure donut + asset trend side-by-side
    eq_v   = float(bd.get("equity") or 0)
    debt_v = float(bd.get("debt")   or 0)
    cash_v = float(bd.get("cash")   or 0)
    net_debt = max(0, debt_v - cash_v)

    donut_img, trend_img = None, None

    if eq_v > 0 or net_debt > 0:
        dlabels, dvals = [], []
        if eq_v > 0:   dlabels.append(f"Equity\n${eq_v/1e6:,.0f}M");   dvals.append(eq_v)
        if net_debt > 0: dlabels.append(f"Net Debt\n${net_debt/1e6:,.0f}M"); dvals.append(net_debt)
        if cash_v > 0: dlabels.append(f"Cash\n${cash_v/1e6:,.0f}M");   dvals.append(cash_v)
        if dvals:
            fig_d = _donut(dlabels, dvals, title="Capital Structure")
            donut_img = _mpl_to_img(fig_d)

    if assets_h is not None and hasattr(assets_h, "index") and not assets_h.empty:
        years_t = _years_from_hist(assets_h, n=5)
        yr_t = [f"FY{str(y)[2:]}" for y in years_t]
        a_v = _safe_series(assets_h, years_t)
        e_v = _safe_series(equity_h, years_t) if equity_h is not None else [None] * len(years_t)
        l_v = _safe_series(liab_h,   years_t) if liab_h   is not None else [None] * len(years_t)
        fig_t = _multi_line(yr_t, {
            "Total Assets":     [v/1e6 if v else None for v in a_v],
            "Equity":           [v/1e6 if v else None for v in e_v],
            "Total Liabilities":[v/1e6 if v else None for v in l_v],
        }, title="Assets, Liabilities & Equity ($M)", y_fmt=lambda v, p: f"${v:,.0f}M",
                            figsize=(4.5, 2.7))
        trend_img = _mpl_to_img(fig_t)

    if donut_img and trend_img:
        dw = CONTENT_W * 0.36; tw = CONTENT_W * 0.60; h_ = dw * 0.82
        row = Table([[Image(donut_img, width=dw, height=h_), Image(trend_img, width=tw, height=h_)]],
                    colWidths=[dw + 4*mm, tw + 4*mm])
        row.setStyle(TableStyle([("ALIGN",(0,0),(-1,-1),"CENTER"), ("VALIGN",(0,0),(-1,-1),"MIDDLE")]))
        story.append(row)
    elif donut_img:
        dw = CONTENT_W * 0.45; h_ = dw * 0.75
        story.append(Image(donut_img, width=dw, height=h_))
    elif trend_img:
        tw = CONTENT_W * 0.85; h_ = tw * 0.38
        story.append(Image(trend_img, width=tw, height=h_))

    story.append(PageBreak())


# ─────────────────────────────────────────────────────────────────
# SECTION 6: CASH FLOW
# ─────────────────────────────────────────────────────────────────
def _build_cashflow(story, styles, ltm_data, lfcf_h, revenue_h, ufcf_h=None):
    story.append(SectionBanner("Cash Flow & Free Cash Flow Analysis",
                                "Operating Cash Flow · CapEx · FCF Generation", section_num="05"))
    story.append(Spacer(1, 4*mm))

    ocf   = ltm_data.get("ocf", float("nan"))
    capex = ltm_data.get("capex", float("nan"))
    try:    fcf_ltm = float(ocf) - float(capex)
    except: fcf_ltm = float("nan")

    # Waterfall
    wf_items, wf_vals = [], []
    for lbl, val in [("Op. Cash Flow", ocf), ("CapEx (−)", capex), ("Free Cash Flow", fcf_ltm)]:
        try:
            v = float(val)
            if not math.isnan(v):
                wf_items.append(lbl)
                wf_vals.append(v / 1e6)
        except: pass

    waterfall_img = None
    if wf_items:
        fig_wf = _waterfall(wf_items, wf_vals, title="LTM Cash Flow Bridge ($M)")
        waterfall_img = _mpl_to_img(fig_wf)

    # FCF history
    fcf_hist_img = None
    if lfcf_h is not None and hasattr(lfcf_h, "index") and not lfcf_h.empty:
        years_f = _years_from_hist(lfcf_h, n=6)
        yr_f = [f"FY{str(y)[2:]}" for y in years_f]
        f_v = [v/1e6 if v else None for v in _safe_series(lfcf_h, years_f)]
        fig_fcf = _multi_line(yr_f, {"Levered FCF ($M)": f_v},
                              title="Free Cash Flow History ($M)", y_fmt=lambda v, p: f"${v:,.0f}M",
                              figsize=(4.5, 2.7))
        fcf_hist_img = _mpl_to_img(fig_fcf)

    if waterfall_img and fcf_hist_img:
        ww = CONTENT_W * 0.44; fw = CONTENT_W * 0.52; h_ = ww * 0.65
        row = Table([[Image(waterfall_img, width=ww, height=h_), Image(fcf_hist_img, width=fw, height=h_)]],
                    colWidths=[ww + 4*mm, fw + 4*mm])
        row.setStyle(TableStyle([("ALIGN",(0,0),(-1,-1),"CENTER"), ("VALIGN",(0,0),(-1,-1),"MIDDLE")]))
        story.append(row)
    elif waterfall_img:
        ww = CONTENT_W * 0.55; story.append(Image(waterfall_img, width=ww, height=ww * 0.65))
    elif fcf_hist_img:
        fw = CONTENT_W * 0.75; story.append(Image(fcf_hist_img, width=fw, height=fw * 0.42))

    story.append(Spacer(1, 4*mm))

    # Cash flow metrics table
    story.append(Paragraph("Cash Flow Metrics Summary", styles["h2"]))

    rev_ltm = ltm_data.get("revenue") or 1
    try:    fcf_yield_v = _fp(float(fcf_ltm) / float(metadata_placeholder := ltm_data.get("market_cap", 1)) * 100 if ltm_data.get("market_cap") else float("nan"))
    except: fcf_yield_v = "—"

    try:    capex_pct = _fp(abs(float(capex)) / float(rev_ltm) * 100 if rev_ltm else float("nan"))
    except: capex_pct = "—"

    cf_rows = [
        [Paragraph(h, styles["th"]) for h in ["METRIC", "LTM VALUE", "FORMULA", "COMMENTARY"]],
        [Paragraph("Operating Cash Flow",       styles["td_l"]), Paragraph(_fv(ocf,     suffix="M", scale=1e6), styles["td_b"]),
         Paragraph("Net Income + D&A ± WC",     styles["td"]),   Paragraph("Core operational cash generation", styles["td_note"])],
        [Paragraph("Capital Expenditures",      styles["td_l"]), Paragraph(_fv(capex,   suffix="M", scale=1e6), styles["td"]),
         Paragraph("Property & equipment spend",styles["td"]),   Paragraph("Investment in long-term assets", styles["td_note"])],
        [Paragraph("Levered Free Cash Flow",    styles["td_l"]), Paragraph(_fv(fcf_ltm, suffix="M", scale=1e6), styles["td_b"]),
         Paragraph("OCF − CapEx",               styles["td"]),   Paragraph("Cash available after maintenance capex", styles["td_note"])],
        [Paragraph("CapEx as % of Revenue",     styles["td_l"]), Paragraph(capex_pct,                            styles["td"]),
         Paragraph("CapEx / Revenue",           styles["td"]),   Paragraph("Capital intensity; < 10% = asset-light", styles["td_note"])],
    ]

    cf_tbl = Table(cf_rows, colWidths=[CONTENT_W*0.26, CONTENT_W*0.16, CONTENT_W*0.22, CONTENT_W*0.32],
                   repeatRows=1)
    cf_tbl.setStyle(_ts())
    story.append(cf_tbl)
    story.append(PageBreak())


# ─────────────────────────────────────────────────────────────────
# SECTIONS 7-10: FINANCIAL RATIOS (All 4 pages)
# ─────────────────────────────────────────────────────────────────
def _build_ratios_profitability(story, styles, summary):
    story.append(SectionBanner("Financial Ratios — Profitability",
                                "Returns on Capital · Assets · Equity", section_num="06"))
    story.append(Spacer(1, 4*mm))

    hdr = [Paragraph(h, styles["th"]) for h in ["RATIO", "VALUE", "BENCHMARK", "FORMULA", "INTERPRETATION"]]

    def _row(label, key, fmt_fn, bench, formula, interp):
        val = fmt_fn(summary.get(key, float("nan")))
        return [Paragraph(label, styles["td_l"]), Paragraph(val, styles["td_b"]),
                Paragraph(bench, styles["td"]),   Paragraph(formula, styles["td_note"]),
                Paragraph(interp, styles["td_note"])]

    rows = [hdr,
        _row("Return on Assets (ROA)",           "ROA %",     _fp, "> 5%",   "Net Income / Total Assets",
             "Measures how efficiently assets generate profit. Higher = better asset utilization."),
        _row("Return on Invested Capital (ROIC)", "ROIC %",    _fp, "> 10%",  "NOPAT / Invested Capital",
             "Premier capital efficiency metric. Consistently > WACC = value creation."),
        _row("Return on Equity (ROE)",            "ROE %",     _fp, "> 15%",  "Net Income / Equity",
             "Return to shareholders. Must be evaluated alongside leverage level."),
        _row("Return on Capital Employed (RCE)",  "RCE %",     _fp, "> 10%",  "EBIT / Capital Employed",
             "Efficiency of total capital deployed. Compare to cost of capital."),
    ]

    cw = [CONTENT_W*0.24, CONTENT_W*0.10, CONTENT_W*0.12, CONTENT_W*0.18, CONTENT_W*0.32]
    tbl = Table(rows, colWidths=cw, repeatRows=1)
    tbl.setStyle(_ts())
    story.append(tbl)
    story.append(Spacer(1, 5*mm))

    # Visual ROE vs ROIC vs ROA chart
    metrics = {"ROA": summary.get("ROA %"), "ROIC": summary.get("ROIC %"), "ROE": summary.get("ROE %"), "RCE": summary.get("RCE %")}
    vals = [float(v) if v and not (isinstance(v, float) and math.isnan(v)) else 0 for v in metrics.values()]
    labels = list(metrics.keys())
    benchmarks = [5, 10, 15, 10]

    with plt.rc_context(MPL_STYLE):
        fig, ax = plt.subplots(figsize=(6, 2.4))
        x = np.arange(len(labels))
        bars = ax.bar(x, vals, width=0.4, color=[CHART_COLORS[0] if v >= b else "#DC2626"
                                                  for v, b in zip(vals, benchmarks)], alpha=0.85, zorder=3)
        ax.plot(x, benchmarks, "D--", color="#C9A84C",
                markersize=7, linewidth=1.5, label="Benchmark", zorder=4)
        ax.set_xticks(x); ax.set_xticklabels(labels)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:.1f}%"))
        ax.legend(loc="upper right"); ax.set_title("Return Metrics vs. Benchmarks", pad=6)
        fig.tight_layout()
        img_ret = _mpl_to_img(fig)

    iw = CONTENT_W * 0.65; ih = iw * 0.38
    story.append(Image(img_ret, width=iw, height=ih))
    story.append(PageBreak())


def _build_ratios_efficiency(story, styles, summary):
    story.append(SectionBanner("Financial Ratios — Margins & Efficiency",
                                "Margin Quality · Turnover · Working Capital Cycle", section_num="07"))
    story.append(Spacer(1, 4*mm))

    hdr = [Paragraph(h, styles["th"]) for h in ["RATIO", "VALUE", "BENCHMARK", "NOTES"]]

    def _row(label, key, fmt_fn, bench, notes):
        val = fmt_fn(summary.get(key, float("nan")))
        return [Paragraph(label, styles["td_l"]), Paragraph(val, styles["td_b"]),
                Paragraph(bench, styles["td"]),   Paragraph(notes, styles["td_note"])]

    # Asset turnover section
    story.append(Paragraph("Asset Turnover Ratios", styles["h2"]))
    rows_eff = [hdr,
        _row("Total Asset Turnover",        "Total Asset Turnover",   _fx, "0.8–1.5x", "Revenue / Total Assets — efficiency of asset base"),
        _row("Accounts Receivable Turnover","AR Turnover",            _fx, "> 8x",     "Revenue / AR — collection speed"),
        _row("Inventory Turnover",          "Inventory Turnover",     _fx, "> 5x",     "COGS / Inventory — stock movement speed"),
        _row("Fixed Asset Turnover",        "Fixed Asset Turnover",   _fx, "> 2x",     "Revenue / Net PP&E — capex productivity"),
    ]
    cw = [CONTENT_W*0.30, CONTENT_W*0.12, CONTENT_W*0.13, CONTENT_W*0.41]
    tbl1 = Table(rows_eff, colWidths=cw, repeatRows=1); tbl1.setStyle(_ts())
    story.append(tbl1); story.append(Spacer(1, 4*mm))

    # Working capital section
    story.append(Paragraph("Working Capital & Cash Conversion Cycle", styles["h2"]))
    rows_wc = [hdr,
        _row("Days Sales Outstanding (DSO)",     "Avg Days Sales Outstanding",    lambda v: _fv(v,suffix=" days",scale=1), "30–60 days", "AR / Revenue × 365"),
        _row("Days Inventory Outstanding (DIO)", "Avg Days Inventory Outstanding",lambda v: _fv(v,suffix=" days",scale=1), "Varies",     "Inventory / COGS × 365"),
        _row("Days Payable Outstanding (DPO)",   "Avg Days Payable Outstanding",  lambda v: _fv(v,suffix=" days",scale=1), "Varies",     "AP / COGS × 365"),
        _row("Cash Conversion Cycle (CCC)",      "Cash Conversion Cycle",         lambda v: _fv(v,suffix=" days",scale=1), "< 45 days",  "DSO + DIO − DPO; negative = excellent"),
    ]
    tbl2 = Table(rows_wc, colWidths=cw, repeatRows=1); tbl2.setStyle(_ts())
    story.append(tbl2)
    story.append(PageBreak())


def _build_ratios_liquidity(story, styles, summary):
    story.append(SectionBanner("Financial Ratios — Liquidity",
                                "Short-Term Solvency · Cash Position · Liquidity Buffers", section_num="08"))
    story.append(Spacer(1, 4*mm))

    hdr = [Paragraph(h, styles["th"]) for h in ["RATIO", "VALUE", "BENCHMARK", "FORMULA", "INTERPRETATION"]]

    def _row(label, key, fmt_fn, bench, formula, interp):
        val = fmt_fn(summary.get(key, float("nan")))
        return [Paragraph(label, styles["td_l"]), Paragraph(val, styles["td_b"]),
                Paragraph(bench, styles["td"]),   Paragraph(formula, styles["td_note"]),
                Paragraph(interp, styles["td_note"])]

    rows = [hdr,
        _row("Current Ratio",   "Current Ratio",  _fx, "1.5–3.0x", "Current Assets / CL",              "Short-term obligation coverage. < 1 signals stress."),
        _row("Quick Ratio",     "Quick Ratio",    _fx, "> 1.0x",   "(CA − Inventory) / CL",            "Excludes inventory; more conservative solvency test."),
        _row("Cash Ratio",      "Cash Ratio",     _fx, "> 0.5x",   "Cash / Current Liabilities",        "Most conservative liquidity; immediate payment ability."),
    ]
    cw = [CONTENT_W*0.22, CONTENT_W*0.10, CONTENT_W*0.12, CONTENT_W*0.20, CONTENT_W*0.32]
    tbl = Table(rows, colWidths=cw, repeatRows=1); tbl.setStyle(_ts())
    story.append(tbl)
    story.append(Spacer(1, 5*mm))

    # Liquidity visualization
    liq_metrics = {
        "Current Ratio": (summary.get("Current Ratio"), 1.5),
        "Quick Ratio":   (summary.get("Quick Ratio"),   1.0),
    }
    with plt.rc_context(MPL_STYLE):
        fig, ax = plt.subplots(figsize=(5.5, 2.3))
        labels_l = list(liq_metrics.keys())
        vals_l   = [float(v[0]) if v[0] and not (isinstance(v[0],float) and math.isnan(v[0])) else 0 for v in liq_metrics.values()]
        bench_l  = [v[1] for v in liq_metrics.values()]
        x = np.arange(len(labels_l))
        bars = ax.bar(x, vals_l, width=0.35, color=[CHART_COLORS[0] if v >= b else "#DC2626"
                                                     for v, b in zip(vals_l, bench_l)], alpha=0.85, zorder=3)
        ax.axhline(y=1.0, color="#DC2626", linestyle="--", linewidth=1.2, label="Min Threshold (1.0x)", zorder=2)
        ax.set_xticks(x); ax.set_xticklabels(labels_l)
        for bar, val in zip(bars, vals_l):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f"{val:.2f}x",
                    ha="center", va="bottom", fontsize=8, fontweight="bold")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:.1f}x"))
        ax.legend(); ax.set_title("Liquidity Ratios vs. Minimum Thresholds", pad=6)
        fig.tight_layout()
        liq_img = _mpl_to_img(fig)
    iw = CONTENT_W * 0.60
    story.append(Image(liq_img, width=iw, height=iw * 0.43))
    story.append(PageBreak())


def _build_ratios_leverage(story, styles, summary):
    story.append(SectionBanner("Financial Ratios — Leverage & Solvency",
                                "Debt Coverage · Interest Service · Bankruptcy Risk", section_num="09"))
    story.append(Spacer(1, 4*mm))

    hdr = [Paragraph(h, styles["th"]) for h in ["RATIO", "VALUE", "BENCHMARK", "FORMULA", "INTERPRETATION"]]

    def _row(label, key, fmt_fn, bench, formula, interp):
        val = fmt_fn(summary.get(key, float("nan")))
        return [Paragraph(label, styles["td_l"]), Paragraph(val, styles["td_b"]),
                Paragraph(bench, styles["td"]),   Paragraph(formula, styles["td_note"]),
                Paragraph(interp, styles["td_note"])]

    rows = [hdr,
        _row("Total Debt / Equity",       "Total D/E",          _fx, "< 1.0x",  "Total Debt / Equity",         "Financial leverage; higher = more debt-funded."),
        _row("Total Debt / Capital",      "Total D/Capital",    _fx, "< 0.5x",  "Debt / (Debt + Equity)",      "Proportion of capital from debt financing."),
        _row("Long-Term Debt / Equity",   "LT D/E",             _fx, "< 0.8x",  "LT Debt / Equity",            "Long-horizon leverage; excludes short-term."),
        _row("Total Liabilities / Assets","Total Liab/Assets",  _fx, "< 0.6x",  "Total Liab / Total Assets",   "Balance sheet solvency; < 60% is conservative."),
        _row("Interest Coverage (EBIT)",  "EBIT/Interest",      _fx, "> 3.0x",  "EBIT / Interest Expense",     "Ability to cover interest from earnings; < 1.5x = stress."),
        _row("Interest Coverage (EBITDA)","EBITDA/Interest",    _fx, "> 4.0x",  "EBITDA / Interest Expense",   "Cash-based coverage; preferred by lenders."),
        _row("Net Debt / EBITDA",         "Net Debt/EBITDA",    _fx, "< 3.0x",  "Net Debt / EBITDA",           "Years of EBITDA to repay debt. > 4x = elevated risk."),
        _row("Altman Z-Score",            "Altman Z-Score",     lambda v: _fv(v,scale=1,dec=2), "> 2.99", "Weighted financial ratios", "Z > 2.99 = Safe; 1.81–2.99 = Grey; < 1.81 = Distress."),
    ]
    cw = [CONTENT_W*0.24, CONTENT_W*0.10, CONTENT_W*0.12, CONTENT_W*0.18, CONTENT_W*0.32]
    tbl = Table(rows, colWidths=cw, repeatRows=1); tbl.setStyle(_ts())
    story.append(tbl)
    story.append(Spacer(1, 4*mm))

    # Altman Z-Score gauge
    story.append(Paragraph("Altman Z-Score Analysis", styles["h2"]))
    z = summary.get("Altman Z-Score")
    try:
        z_val = float(z)
        bands = [(0, 1.81, "#DC2626"), (1.81, 2.99, "#D97706"), (2.99, 6, "#059669")]
        fig_z = _gauge(z_val, 0, 6, "Altman Z-Score",
                       bands=[(0, 1.81, "#DC2626"), (1.81, 2.99, "#D97706"), (2.99, 6, "#059669")])
        img_z = _mpl_to_img(fig_z)
        zw = CONTENT_W * 0.38; zh = zw * 0.72

        zone = "SAFE ZONE (> 2.99)" if z_val > 2.99 else ("GREY ZONE (1.81–2.99)" if z_val > 1.81 else "DISTRESS ZONE (< 1.81)")
        zone_color = "GREEN" if z_val > 2.99 else ("AMBER" if z_val > 1.81 else "RED")

        z_commentary = Table([
            [Image(img_z, width=zw, height=zh),
             Table([
                 [Paragraph("Z-SCORE INTERPRETATION", styles["h3"])],
                 [Paragraph(f"<b>Current Z-Score: {z_val:.2f}</b>", styles["body"])],
                 [Paragraph(f"Status: {zone}", styles["body"])],
                 [Spacer(1, 3*mm)],
                 [Paragraph("• <b>&gt; 2.99</b> — Safe Zone: Low bankruptcy risk", styles["body_sm"])],
                 [Paragraph("• <b>1.81–2.99</b> — Grey Zone: Monitor closely", styles["body_sm"])],
                 [Paragraph("• <b>&lt; 1.81</b> — Distress Zone: High default risk", styles["body_sm"])],
                 [Spacer(1, 2*mm)],
                 [Paragraph("Altman Z-Score uses five financial ratios weighted to predict corporate distress probability within 2 years.", styles["body_sm"])],
             ], colWidths=[CONTENT_W * 0.54])]
        ], colWidths=[zw + 4*mm, CONTENT_W * 0.54])
        z_commentary.setStyle(TableStyle([("VALIGN",(0,0),(-1,-1),"TOP"), ("LEFTPADDING",(0,0),(-1,-1),4)]))
        story.append(z_commentary)
    except: pass

    story.append(PageBreak())


# ─────────────────────────────────────────────────────────────────
# SECTION 11: GROWTH & CAGR
# ─────────────────────────────────────────────────────────────────
def _build_growth(story, styles, summary, revenue_h, gross_h, ebit_h, ebitda_h, net_income_h):
    story.append(SectionBanner("Growth & CAGR Analysis",
                                "Year-over-Year Growth · Compound Annual Growth Rates", section_num="10"))
    story.append(Spacer(1, 4*mm))

    # Growth rates table
    story.append(Paragraph("Year-over-Year & CAGR Summary", styles["h2"]))

    def _gfmt(key):
        v = summary.get(key, float("nan"))
        return _fp(v)

    def _gcol(val_str):
        try:
            f = float(val_str.replace("%","").replace("—","").strip())
            col = "#059669" if f > 0 else "#DC2626"
            return f'<font color="{col}">{val_str}</font>'
        except: return val_str

    growth_rows = [
        [Paragraph(h, styles["th"]) for h in ["METRIC", "YoY Growth", "2-Yr CAGR", "3-Yr CAGR", "5-Yr CAGR"]],
    ]

    metrics_g = [
        ("Revenue",     "Revenue YoY %",     "Revenue"),
        ("Gross Profit","Gross Profit YoY %","Gross Profit"),
        ("EBITDA",      "EBITDA YoY %",      "EBITDA"),
        ("EBIT",        "EBIT YoY %",        "EBIT"),
        ("Net Income",  "Net Income YoY %",  "Net Income"),
        ("EPS (Basic)", "EPS YoY %",         "EPS (Basic)"),
    ]

    for label, yoy_key, base in metrics_g:
        yoy  = _fp(summary.get(yoy_key, float("nan")))
        c2   = _fp(summary.get(f"{base} 2yr CAGR %", float("nan")))
        c3   = _fp(summary.get(f"{base} 3yr CAGR %", float("nan")))
        c5   = _fp(summary.get(f"{base} 5yr CAGR %", float("nan")))
        growth_rows.append([
            Paragraph(label, styles["td_l"]),
            Paragraph(_gcol(yoy), styles["td_b"]),
            Paragraph(_gcol(c2),  styles["td"]),
            Paragraph(_gcol(c3),  styles["td"]),
            Paragraph(_gcol(c5),  styles["td"]),
        ])

    cw = [CONTENT_W*0.26, CONTENT_W*0.185, CONTENT_W*0.185, CONTENT_W*0.185, CONTENT_W*0.185]
    g_tbl = Table(growth_rows, colWidths=cw, repeatRows=1)
    g_tbl.setStyle(_ts())
    story.append(g_tbl)
    story.append(Spacer(1, 4*mm))

    # Revenue trajectory chart with YoY annotations
    if revenue_h is not None and hasattr(revenue_h, "index") and not revenue_h.empty:
        years_r = _years_from_hist(revenue_h, n=6)
        yr_r = [f"FY{str(y)[2:]}" for y in years_r]
        rev_v = [v/1e6 if v else 0 for v in _safe_series(revenue_h, years_r)]

        with plt.rc_context(MPL_STYLE):
            fig, ax = plt.subplots(figsize=(7.2, 2.8))
            x = np.arange(len(yr_r))
            bars = ax.bar(x, rev_v, width=0.5, color=CHART_COLORS[0], alpha=0.83, zorder=3)
            # YoY annotations
            for i in range(1, len(rev_v)):
                try:
                    yoy_g = (rev_v[i] - rev_v[i-1]) / abs(rev_v[i-1]) * 100 if rev_v[i-1] else 0
                    col = "#059669" if yoy_g >= 0 else "#DC2626"
                    ax.text(x[i], rev_v[i] + max(rev_v) * 0.015, f"{yoy_g:+.1f}%",
                            ha="center", va="bottom", fontsize=7, color=col, fontweight="bold")
                except: pass
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"${v:,.0f}M"))
            ax.set_xticks(x); ax.set_xticklabels(yr_r)
            ax.set_title("Revenue Trajectory with YoY Growth Rates", pad=6)
            fig.tight_layout()
            rev_img = _mpl_to_img(fig)

        story.append(Image(rev_img, width=CONTENT_W * 0.92, height=CONTENT_W * 0.92 * 0.38))

    # Multi-metric CAGR comparison bar
    cagr_labels = ["Revenue", "Gross Profit", "EBITDA", "Net Income"]
    cagr_2yr = [float(summary.get(f"{m} 2yr CAGR %", 0) or 0) for m in cagr_labels]
    cagr_3yr = [float(summary.get(f"{m} 3yr CAGR %", 0) or 0) for m in cagr_labels]

    with plt.rc_context(MPL_STYLE):
        fig, ax = plt.subplots(figsize=(6.5, 2.5))
        x = np.arange(len(cagr_labels)); w = 0.3
        ax.bar(x - w/2, cagr_2yr, width=w, color=CHART_COLORS[0], alpha=0.85, label="2-Yr CAGR", zorder=3)
        ax.bar(x + w/2, cagr_3yr, width=w, color=CHART_COLORS[1], alpha=0.85, label="3-Yr CAGR", zorder=3)
        ax.axhline(0, color="grey", linewidth=0.8)
        ax.set_xticks(x); ax.set_xticklabels(cagr_labels)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:.1f}%"))
        ax.legend(loc="best")
        ax.set_title("CAGR Comparison by Metric", pad=6)
        fig.tight_layout()
        cagr_img = _mpl_to_img(fig)

    story.append(Spacer(1, 3*mm))
    story.append(Image(cagr_img, width=CONTENT_W * 0.78, height=CONTENT_W * 0.78 * 0.38))
    story.append(PageBreak())


# ─────────────────────────────────────────────────────────────────
# SECTION 12: VALUATION MULTIPLES
# ─────────────────────────────────────────────────────────────────
def _build_multiples(story, styles, summary, metadata):
    story.append(SectionBanner("Valuation Multiples",
                                "Trading Multiples · Relative Valuation · PEG Analysis", section_num="11"))
    story.append(Spacer(1, 4*mm))

    # Primary multiples table
    story.append(Paragraph("Valuation Multiple Summary", styles["h2"]))

    mult_rows = [
        [Paragraph(h, styles["th"]) for h in ["MULTIPLE", "VALUE", "INTERPRETATION", "INVESTOR NOTE"]],
        [Paragraph("Price / Earnings (LTM P/E)", styles["td_l"]),
         Paragraph(_fx(metadata.get("pe_ltm")), styles["td_b"]),
         Paragraph("Price per $1 of LTM earnings", styles["td"]),
         Paragraph("Compare to sector median; context-dependent", styles["td_note"])],
        [Paragraph("EV / EBITDA", styles["td_l"]),
         Paragraph(_fx(summary.get("EV/EBITDA")), styles["td_b"]),
         Paragraph("Enterprise value vs. cash earnings", styles["td"]),
         Paragraph("Primary LBO/M&A benchmark; 8–12x typical", styles["td_note"])],
        [Paragraph("EV / Revenue", styles["td_l"]),
         Paragraph(_fx(summary.get("EV/Revenue")), styles["td_b"]),
         Paragraph("Valuation relative to top-line", styles["td"]),
         Paragraph("Key for growth companies; < 3x = reasonable", styles["td_note"])],
        [Paragraph("EV / EBIT", styles["td_l"]),
         Paragraph(_fx(summary.get("EV/EBIT")), styles["td_b"]),
         Paragraph("Post-D&A enterprise value metric", styles["td"]),
         Paragraph("Capital structure neutral profitability multiple", styles["td_note"])],
        [Paragraph("Price / Book (P/B)", styles["td_l"]),
         Paragraph(_fx(metadata.get("pb_ratio")), styles["td_b"]),
         Paragraph("Market price vs. accounting equity", styles["td"]),
         Paragraph("< 1x may suggest undervaluation", styles["td_note"])],
        [Paragraph("Price / FCF", styles["td_l"]),
         Paragraph(_fx(summary.get("P/FCF")), styles["td_b"]),
         Paragraph("Equity value vs. free cash flow", styles["td"]),
         Paragraph("FCF-based alternative to P/E", styles["td_note"])],
        [Paragraph("FCF Yield", styles["td_l"]),
         Paragraph(_fp(summary.get("FCF Yield %")), styles["td_b"]),
         Paragraph("FCF as % of market cap", styles["td"]),
         Paragraph("Inverse of P/FCF; > 5% = attractive", styles["td_note"])],
        [Paragraph("Dividend Yield", styles["td_l"]),
         Paragraph(_fp(metadata.get("dividend_yield_pct")), styles["td"]),
         Paragraph("Annual dividend / share price", styles["td"]),
         Paragraph("Income return component of total return", styles["td_note"])],
        [Paragraph("PEG Ratio (LTM)", styles["td_l"]),
         Paragraph(_fx(summary.get("PEG (PE LTM)")), styles["td_b"]),
         Paragraph("P/E adjusted for growth", styles["td"]),
         Paragraph("< 1.0 may indicate undervaluation vs. growth", styles["td_note"])],
        [Paragraph("PEG Ratio (Lynch)", styles["td_l"]),
         Paragraph(_fx(summary.get("PEG (Lynch)")), styles["td_b"]),
         Paragraph("P/E vs. earnings growth + yield", styles["td"]),
         Paragraph("Peter Lynch's growth-adjusted valuation", styles["td_note"])],
    ]

    cw = [CONTENT_W*0.25, CONTENT_W*0.10, CONTENT_W*0.23, CONTENT_W*0.38]
    m_tbl = Table(mult_rows, colWidths=cw, repeatRows=1)
    m_tbl.setStyle(_ts())
    story.append(m_tbl)
    story.append(Spacer(1, 4*mm))

    # Multiples bar chart
    mult_labels = ["EV/EBITDA", "EV/Revenue", "P/FCF"]
    mult_vals = [
        float(summary.get("EV/EBITDA") or 0),
        float(summary.get("EV/Revenue") or 0),
        float(summary.get("P/FCF") or 0),
    ]
    # Industry benchmarks (general reference)
    mult_bench = [10, 3, 20]

    with plt.rc_context(MPL_STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.5))
        for ax, lbl, val, bench in zip(axes, mult_labels, mult_vals, mult_bench):
            col = CHART_COLORS[0] if val and val <= bench * 1.5 else "#DC2626"
            ax.barh([lbl], [val], color=col, alpha=0.85, height=0.4)
            if bench:
                ax.axvline(bench, color=CHART_COLORS[1], linestyle="--", linewidth=1.5, label=f"Bench: {bench}x")
            try: ax.text(val + 0.1, 0, f"{val:.1f}x", va="center", fontsize=9, fontweight="bold", color="#1F2937")
            except: pass
            ax.set_title(lbl, fontsize=8)
            ax.legend(fontsize=6.5)
            ax.set_xlim(0, max(val * 1.5, bench * 1.5) if val else bench * 2)
        plt.suptitle("Key Multiples vs. Reference Benchmarks", fontsize=9, fontweight="bold", color="#1F2937")
        fig.tight_layout()
        mv_img = _mpl_to_img(fig)

    story.append(Image(mv_img, width=CONTENT_W * 0.92, height=CONTENT_W * 0.92 * 0.33))
    story.append(Spacer(1, 4*mm))

    # Valuation commentary
    story.append(Paragraph("Institutional Valuation Commentary", styles["h2"]))
    commentary = (
        "The valuation multiples presented should be interpreted within the context of the company's "
        "industry, capital structure, stage of growth, and macroeconomic environment. "
        "<b>EV/EBITDA</b> is the most widely referenced metric in M&amp;A and leveraged buyout analysis, "
        "as it strips out capital structure differences and D&amp;A policy. "
        "<b>P/E</b> remains the dominant retail investor metric but must be adjusted for one-time items. "
        "<b>FCF Yield</b> and <b>Price/FCF</b> offer an intrinsic value perspective particularly relevant "
        "for capital-light, high-margin businesses. "
        "The <b>PEG ratio</b> contextualizes P/E by growth rate — a PEG below 1.0 historically signals "
        "potential undervaluation relative to growth prospects. "
        "Recipients should cross-reference these multiples against sector medians, historical trading "
        "ranges, and comparable company analyses before drawing investment conclusions."
    )
    story.append(Paragraph(commentary, styles["body"]))
    story.append(PageBreak())


# ─────────────────────────────────────────────────────────────────
# SECTION 13: RISK ANALYSIS
# ─────────────────────────────────────────────────────────────────
def _build_risk_analysis(story, styles, summary, metadata):
    story.append(SectionBanner("Risk Analysis",
                                "Volatility · Credit · Operational Risk Factors", section_num="12"))
    story.append(Spacer(1, 4*mm))

    story.append(Paragraph("Financial Risk Indicators", styles["h2"]))

    risk_rows = [
        [Paragraph(h, styles["th"]) for h in ["RISK FACTOR", "METRIC", "VALUE", "SIGNAL", "COMMENTARY"]],
        [Paragraph("Credit / Leverage Risk",   styles["td_l"]),
         Paragraph("Net Debt / EBITDA",         styles["td"]),
         Paragraph(_fx(summary.get("Net Debt/EBITDA")), styles["td_b"]),
         Paragraph("< 3.0x = Low", styles["td_note"]),
         Paragraph("Primary lender covenant metric; > 4x = elevated refinancing risk", styles["td_note"])],
        [Paragraph("Interest Coverage Risk",   styles["td_l"]),
         Paragraph("EBIT / Interest",           styles["td"]),
         Paragraph(_fx(summary.get("EBIT/Interest")), styles["td_b"]),
         Paragraph("> 3.0x = Safe", styles["td_note"]),
         Paragraph("Ability to service debt from earnings; < 1.5x = distress signal", styles["td_note"])],
        [Paragraph("Liquidity Risk",            styles["td_l"]),
         Paragraph("Current Ratio",             styles["td"]),
         Paragraph(_fx(summary.get("Current Ratio")), styles["td_b"]),
         Paragraph("> 1.5x = Low", styles["td_note"]),
         Paragraph("Near-term obligation coverage; < 1.0x = immediate concern", styles["td_note"])],
        [Paragraph("Solvency Risk",             styles["td_l"]),
         Paragraph("Altman Z-Score",            styles["td"]),
         Paragraph(_fv(summary.get("Altman Z-Score"), scale=1, dec=2), styles["td_b"]),
         Paragraph("> 2.99 = Safe", styles["td_note"]),
         Paragraph("Composite bankruptcy predictor; grey zone 1.81–2.99", styles["td_note"])],
        [Paragraph("Capital Structure Risk",    styles["td_l"]),
         Paragraph("Total D/E Ratio",           styles["td"]),
         Paragraph(_fx(summary.get("Total D/E")), styles["td_b"]),
         Paragraph("< 1.0x = Low", styles["td_note"]),
         Paragraph("Debt reliance; higher leverage amplifies both gains and losses", styles["td_note"])],
        [Paragraph("Profitability Risk",        styles["td_l"]),
         Paragraph("ROIC vs. Cost of Capital",  styles["td"]),
         Paragraph(_fp(summary.get("ROIC %")), styles["td_b"]),
         Paragraph("> 10% = Positive", styles["td_note"]),
         Paragraph("ROIC consistently above WACC signals durable competitive advantage", styles["td_note"])],
    ]

    cw = [CONTENT_W*0.19, CONTENT_W*0.17, CONTENT_W*0.10, CONTENT_W*0.13, CONTENT_W*0.37]
    r_tbl = Table(risk_rows, colWidths=cw, repeatRows=1)
    r_tbl.setStyle(_ts())
    story.append(r_tbl)
    story.append(Spacer(1, 4*mm))

    # Risk radar / spider (as bar for simplicity & reliability)
    risk_metrics = {
        "Leverage\n(D/E)":     min(3.0, abs(float(summary.get("Total D/E") or 0))),
        "Coverage\n(EBIT/Int)":min(10.0, abs(float(summary.get("EBIT/Interest") or 0))),
        "Liquidity\n(Current)":min(4.0, abs(float(summary.get("Current Ratio") or 0))),
        "Profitability\n(ROIC)":min(30.0, abs(float(summary.get("ROIC %") or 0))),
        "FCF Margin\n(%)":      min(30.0, abs(float(summary.get("LFCF Margin %") or 0))),
    }

    with plt.rc_context(MPL_STYLE):
        fig, ax = plt.subplots(figsize=(6.5, 2.5))
        lbl_r = list(risk_metrics.keys())
        val_r = list(risk_metrics.values())
        x = np.arange(len(lbl_r))
        # Color code: leverage is bad high, coverage/liquidity/profitability good high
        bad_high = [True, False, False, False, False]
        cols_r = []
        for i, (v, bh) in enumerate(zip(val_r, bad_high)):
            if bh:
                cols_r.append("#059669" if v < 1 else ("#D97706" if v < 2 else "#DC2626"))
            else:
                cols_r.append("#059669" if v > 5 else ("#D97706" if v > 2 else "#DC2626"))
        ax.bar(x, val_r, color=cols_r, alpha=0.82, width=0.45, zorder=3)
        ax.set_xticks(x); ax.set_xticklabels(lbl_r, fontsize=7.5)
        ax.set_title("Risk Metric Snapshot", pad=6)
        fig.tight_layout()
        risk_img = _mpl_to_img(fig)

    story.append(Image(risk_img, width=CONTENT_W * 0.75, height=CONTENT_W * 0.75 * 0.38))
    story.append(PageBreak())


# ─────────────────────────────────────────────────────────────────
# SECTION 14: LEGAL DISCLAIMER
# ─────────────────────────────────────────────────────────────────
def _build_disclaimer(story, styles, ticker, company_name, report_date):
    story.append(SectionBanner("Legal Disclaimer & Disclosures",
                                "Important Information — Please Read Carefully", section_num="13", bg=SLATE))
    story.append(Spacer(1, 6*mm))

    blocks = [
        ("<b>CONFIDENTIALITY NOTICE</b>",
         f"This report has been prepared by the SEC Equity Analyzer platform and is intended solely for "
         f"the use of the named recipient. Any reproduction, redistribution, or disclosure to third parties "
         f"without prior written consent is strictly prohibited."),

        ("<b>NOT INVESTMENT ADVICE</b>",
         f"Nothing contained herein constitutes investment advice, a recommendation, solicitation, or an offer "
         f"to buy or sell any security or financial instrument. This document is prepared for informational and "
         f"analytical purposes only and does not take into account the specific investment objectives, financial "
         f"situation, tax position, or particular needs of any individual investor or institution."),

        ("<b>DATA SOURCES & ACCURACY</b>",
         f"All financial data referenced in this report has been sourced from SEC EDGAR filings, Yahoo Finance "
         f"(yfinance), and other publicly available information as of {report_date}. While every effort has been "
         f"made to ensure accuracy and completeness, the platform makes no representations or warranties, express "
         f"or implied, as to the reliability of such data. Figures may differ from audited financial statements."),

        ("<b>FORWARD-LOOKING STATEMENTS</b>",
         f"Certain statements, projections, or analyses in this report may constitute forward-looking statements "
         f"involving known and unknown risks and uncertainties. Actual results, performance, or events may differ "
         f"materially from those expressed or implied. Past performance is not indicative, and should not be "
         f"relied upon as an indicator, of future results."),

        ("<b>NO REGULATORY ENDORSEMENT</b>",
         f"This report has not been reviewed, approved, or endorsed by any securities regulator, exchange, or "
         f"governmental authority. It does not constitute a prospectus, offering memorandum, or placement "
         f"document. Recipients should seek independent professional, legal, and tax advice tailored to their "
         f"specific circumstances before making any investment decision."),

        ("<b>SUBJECT COMPANY</b>",
         f"{company_name} ({ticker}). Analysis prepared using financial data as of {report_date}. "
         f"The platform has no position in the securities of the subject company and receives no compensation "
         f"from the subject company or any third party in connection with the preparation of this report."),
    ]

    for heading, body in blocks:
        story.append(Paragraph(heading, styles["h3"]))
        story.append(Paragraph(body, styles["disc"]))
        story.append(Spacer(1, 3*mm))

    story.append(HLine(color=GOLD, thickness=1.2))
    story.append(Spacer(1, 4*mm))

    footer_text = (
        f"© {datetime.datetime.now().year}  SEC Equity Analyzer Platform  ·  "
        f"Powered by SEC EDGAR & Yahoo Finance  ·  "
        f"Report generated: {report_date}"
    )
    story.append(Paragraph(footer_text, styles["footer"]))


# ─────────────────────────────────────────────────────────────────
# OPTIONAL: PEER COMPARISON
# ─────────────────────────────────────────────────────────────────
def _build_peer_comparison(story, styles, peers, ticker, summary):
    """Optional section: rendered only if peers list is provided."""
    if not peers:
        return
    story.append(PageBreak())
    story.append(SectionBanner("Peer Comparison Analysis",
                                "Comparable Company Benchmarking", section_num="~"))
    story.append(Spacer(1, 4*mm))

    hdr = [Paragraph(h, styles["th"]) for h in
           ["COMPANY", "TICKER", "MKT CAP", "EV/EBITDA", "EV/REV", "P/E", "GROSS MGN", "EBITDA MGN", "ROIC", "MATCH SCORE"]]

    peer_rows = [hdr]
    for p in peers[:12]:
        is_subject = p.get("ticker","") == ticker
        style_n = styles["td_bl"] if is_subject else styles["td_l"]
        style_v = styles["td_b"]  if is_subject else styles["td"]
        peer_rows.append([
            Paragraph((p.get("name","—") or "—")[:22], style_n),
            Paragraph(p.get("ticker","—"),        style_v),
            Paragraph(_fv(p.get("market_cap"),suffix="B",scale=1e9,dec=1), style_v),
            Paragraph(_fx(p.get("ev_ebitda")),    style_v),
            Paragraph(_fx(p.get("ev_revenue")),   style_v),
            Paragraph(_fx(p.get("pe_ratio")),      style_v),
            Paragraph(_fp(p.get("gross_margin")), style_v),
            Paragraph(_fp(p.get("ebitda_margin")),style_v),
            Paragraph(_fp(p.get("roic")),         style_v),
            Paragraph(f"{p.get('composite_score',0)*100:.0f}%" if p.get("composite_score") else "—", style_v),
        ])

    cw = [CONTENT_W*0.18, CONTENT_W*0.065, CONTENT_W*0.09,
          CONTENT_W*0.08, CONTENT_W*0.08, CONTENT_W*0.07,
          CONTENT_W*0.09, CONTENT_W*0.10, CONTENT_W*0.07, CONTENT_W*0.10]
    p_tbl = Table(peer_rows, colWidths=cw, repeatRows=1)
    style_peer = _ts(compact=True)
    # Highlight subject row
    for i, p in enumerate(peers[:12], 1):
        if p.get("ticker","") == ticker:
            style_peer.add("BACKGROUND", (0, i), (-1, i), TEAL_LIGHT)
            style_peer.add("TEXTCOLOR",  (0, i), (-1, i), NAVY)
    p_tbl.setStyle(style_peer)
    story.append(p_tbl)


# ─────────────────────────────────────────────────────────────────
# OPTIONAL: PORTFOLIO SECTION
# ─────────────────────────────────────────────────────────────────
def _build_portfolio_section(story, styles, portfolio_data):
    """Optional: rendered only if portfolio_data dict is provided."""
    if not portfolio_data:
        return
    story.append(PageBreak())
    story.append(SectionBanner("Portfolio Overview",
                                "Holdings Analysis · Return Attribution · Risk Metrics", section_num="~"))
    story.append(Spacer(1, 4*mm))

    kpi_items = [
        ("Total Invested",      portfolio_data.get("net_invested",    "—")),
        ("Current Value",       portfolio_data.get("current_value",   "—")),
        ("Total Return ($)",    portfolio_data.get("total_return_dollars","—")),
        ("Total Return (%)",    portfolio_data.get("total_return_pct", "—")),
        ("XIRR (Annualised)",   portfolio_data.get("xirr",            "—")),
        ("Max Drawdown",        portfolio_data.get("max_drawdown",    "—")),
        ("Sharpe Ratio",        portfolio_data.get("sharpe",          "—")),
        ("Beta",                portfolio_data.get("beta",            "—")),
    ]

    n = 4
    card_w = CONTENT_W / n - 2*mm
    rows_c = []
    for i in range(0, len(kpi_items), n):
        chunk = kpi_items[i:i+n]
        rows_c.append([MetricCard(lbl, str(val), width=card_w, height=22*mm) for lbl, val in chunk])

    ct = Table(rows_c, colWidths=[card_w + 2*mm] * n)
    ct.setStyle(TableStyle([("ALIGN",(0,0),(-1,-1),"CENTER"),("VALIGN",(0,0),(-1,-1),"MIDDLE"),
                             ("TOPPADDING",(0,0),(-1,-1),3),("BOTTOMPADDING",(0,0),(-1,-1),3)]))
    story.append(ct)

    # Holdings table
    holdings = portfolio_data.get("holdings", [])
    if holdings:
        story.append(Spacer(1, 5*mm))
        story.append(Paragraph("Portfolio Holdings", styles["h2"]))
        hdr = [Paragraph(h, styles["th"]) for h in ["TICKER", "WEIGHT", "VALUE", "RETURN", "SECTOR"]]
        h_rows = [hdr]
        for h in holdings[:20]:
            h_rows.append([
                Paragraph(h.get("ticker","—"), styles["td_b"]),
                Paragraph(_fp(h.get("weight")), styles["td"]),
                Paragraph(_fv(h.get("value"), suffix="K", scale=1e3), styles["td"]),
                Paragraph(_fp(h.get("return")), styles["td"]),
                Paragraph(h.get("sector","—")[:18], styles["td_l"]),
            ])
        ht = Table(h_rows, colWidths=[CONTENT_W*0.12, CONTENT_W*0.12, CONTENT_W*0.16, CONTENT_W*0.14, CONTENT_W*0.40],
                   repeatRows=1)
        ht.setStyle(_ts())
        story.append(ht)


# ─────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────
def generate_equity_report_pdf(
    ticker:              str,
    company_name:        str,
    metadata:            dict,
    ltm_data:            dict,
    balance_data:        dict,
    summary:             dict,
    revenue_history      = None,
    gross_profit_history = None,
    ebit_history         = None,
    ebitda_history       = None,
    net_income_history   = None,
    lfcf_history         = None,
    ufcf_history         = None,
    total_assets_history = None,
    total_liabilities_history = None,
    equity_history       = None,
    peers                = None,   # List[dict] from capital_iq_style_peer_finder
    portfolio_data       = None,   # dict from performance.py
    prepared_by: str     = "Institutional Equity Research Platform",
) -> bytes:
    """
    Generate a comprehensive institutional-grade equity research PDF.

    Parameters match the data structures already in the app's session state.
    All parameters except ticker/company_name/metadata/ltm_data/balance_data/summary
    are optional — sections degrade gracefully when data is absent.

    Returns:
        bytes: Complete PDF byte string for st.download_button.
    """
    buf         = io.BytesIO()
    report_date = datetime.datetime.now().strftime("%B %d, %Y")
    mcap_str    = _fv(metadata.get("market_cap"), suffix="B", scale=1e9, dec=2)
    industry    = metadata.get("industry", "—")
    sector      = metadata.get("sector",   "—")

    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=16*mm, bottomMargin=15*mm,
        title=f"{ticker} — Institutional Equity Research",
        author="SEC Equity Analyzer",
        subject=f"Comprehensive Financial Analysis: {company_name}",
        creator="SEC Equity Analyzer Platform",
    )

    ST = _styles()
    story = []

    # ── 1. Cover ──────────────────────────────────────────────────
    _build_cover(story, ST, ticker, company_name, industry, sector,
                 mcap_str, report_date, prepared_by)

    # ── 2. Table of Contents ──────────────────────────────────────
    _build_toc(story, ST)

    # ── 3. Executive Summary ──────────────────────────────────────
    _build_executive_summary(story, ST, summary, metadata, ticker, ltm_data, balance_data)

    # ── 4. Income Statement + 5. Margin Analysis ─────────────────
    _build_income_statement(story, ST, ltm_data,
                            revenue_history, gross_profit_history,
                            ebit_history, ebitda_history, net_income_history,
                            summary=summary)

    # ── 6. Balance Sheet ──────────────────────────────────────────
    _build_balance_sheet(story, ST, balance_data,
                         total_assets_history, equity_history, total_liabilities_history)

    # ── 7. Cash Flow ──────────────────────────────────────────────
    _build_cashflow(story, ST, ltm_data, lfcf_history, revenue_history, ufcf_history)

    # ── 8-11. Financial Ratios (all 4 categories) ─────────────────
    _build_ratios_profitability(story, ST, summary)
    _build_ratios_efficiency(story, ST, summary)
    _build_ratios_liquidity(story, ST, summary)
    _build_ratios_leverage(story, ST, summary)

    # ── 12. Growth & CAGR ─────────────────────────────────────────
    _build_growth(story, ST, summary,
                  revenue_history, gross_profit_history,
                  ebit_history, ebitda_history, net_income_history)

    # ── 13. Valuation Multiples ───────────────────────────────────
    _build_multiples(story, ST, summary, metadata)

    # ── 14. Risk Analysis ─────────────────────────────────────────
    _build_risk_analysis(story, ST, summary, metadata)

    # ── Optional: Peer Comparison ─────────────────────────────────
    if peers:
        _build_peer_comparison(story, ST, peers, ticker, summary)

    # ── Optional: Portfolio Section ───────────────────────────────
    if portfolio_data:
        _build_portfolio_section(story, ST, portfolio_data)

    # ── Final: Legal Disclaimer ───────────────────────────────────
    story.append(PageBreak())
    _build_disclaimer(story, ST, ticker, company_name, report_date)

    # ── Build ─────────────────────────────────────────────────────
    on_cover, on_page = _make_templates(ticker, company_name, report_date)
    doc.build(story, onFirstPage=on_cover, onLaterPages=on_page)

    buf.seek(0)
    return buf.read()
