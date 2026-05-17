"""
report_generator.py
────────────────────
Generates a clinical PDF summary report for EchoGlow sessions.
Requires: fpdf2  (pip install fpdf2)

NOTE: fpdf2 default fonts (Helvetica) only support Latin-1.
      All strings here are deliberately ASCII/Latin-1 safe.
"""

import time
from collections import Counter
from datetime import datetime
from typing import List


def _ts(t: float) -> str:
    return datetime.fromtimestamp(t).strftime("%H:%M:%S")


def _dt(t: float) -> str:
    return datetime.fromtimestamp(t).strftime("%Y-%m-%d %H:%M")


# Severity RGB colours
SEV_RGB = {
    "critical": (200, 30,  30),
    "urgent":   (200, 100,  0),
    "monitor":  (140, 100,  0),
    "normal":   (40,  140, 90),
}

# ASCII-safe classification labels (no emoji / special Unicode)
CLS_LABEL = {
    "calm":                "Calm",
    "hunger":              "Hunger Cry",
    "discomfort":          "Discomfort",
    "pain":                "Pain Cry",
    "stridor":             "[URGENT] Stridor",
    "wheezing":            "[URGENT] Wheezing",
    "respiratory_distress":"[CRITICAL] Resp. Distress",
    "unknown":             "Unknown",
}

# Severity prefix tag
SEV_TAG = {
    "critical": "[CRITICAL]",
    "urgent":   "[URGENT]",
    "monitor":  "[MONITOR]",
    "normal":   "[OK]",
}


class ReportGenerator:

    def generate(self, alerts: List[dict], hours: int,
                 session: dict, generated_at: float) -> bytes:
        try:
            from fpdf import FPDF
        except ImportError:
            raise RuntimeError(
                "fpdf2 is required for PDF reports. Run:  pip install fpdf2"
            )

        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=18)
        pdf.set_margins(15, 15, 15)
        pdf.add_page()

        W = pdf.w - 30   # usable width (A4 = 210mm, margins=15 each side)

        # ── Header bar ────────────────────────────────────────────────────────
        pdf.set_fill_color(10, 60, 50)
        pdf.rect(0, 0, pdf.w, 26, style="F")

        pdf.set_y(6)
        pdf.set_font("Helvetica", "B", 15)
        pdf.set_text_color(0, 220, 170)
        pdf.cell(0, 8, "EchoGlow  Neonatal Monitor", ln=True, align="L")

        pdf.set_font("Helvetica", "", 8)
        pdf.set_text_color(160, 210, 200)
        period = f"Last {hours} Hour{'s' if hours > 1 else ''}"
        pdf.cell(0, 5,
                 f"Clinical Summary Report  |  {period}  |  "
                 f"Generated: {_dt(generated_at)}",
                 ln=True)

        pdf.set_y(32)
        pdf.set_text_color(0, 0, 0)

        # ── Summary stat boxes ────────────────────────────────────────────────
        total   = len(alerts)
        crit    = sum(1 for a in alerts if a.get("severity") == "critical")
        urgent  = sum(1 for a in alerts if a.get("severity") == "urgent")
        monitor = sum(1 for a in alerts if a.get("severity") == "monitor")
        resp_c  = sum(1 for a in alerts if a.get("respiratory_concern"))

        start_t = session.get("start_time")
        dur_s   = int(time.time() - start_t) if start_t else 0
        dur_str = (f"{dur_s // 3600}h {(dur_s % 3600) // 60}m"
                   if dur_s >= 3600 else f"{dur_s // 60}m {dur_s % 60}s")

        # Draw 5 stat boxes side by side
        boxes = [
            ("TOTAL ALERTS",    total,   50, 120, 100),
            ("CRITICAL",        crit,   200,  30,  30),
            ("URGENT",          urgent, 200, 100,   0),
            ("MONITOR",         monitor,140, 100,   0),
            ("RESP. CONCERNS",  resp_c,  80,  80, 160),
        ]
        bw = (W - 4 * 2) / 5          # box width with 2mm gaps
        x0 = 15
        for label, val, r, g, b in boxes:
            pdf.set_xy(x0, pdf.get_y())
            pdf.set_fill_color(r, g, b)
            pdf.set_text_color(255, 255, 255)
            pdf.set_font("Helvetica", "B", 16)
            pdf.cell(bw, 14, str(val), align="C", fill=True)
            x0 += bw + 2

        pdf.ln(14)
        x0 = 15
        for label, *_ in boxes:
            pdf.set_xy(x0, pdf.get_y())
            pdf.set_font("Helvetica", "", 7)
            pdf.set_text_color(80, 80, 80)
            pdf.cell(bw, 5, label, align="C")
            x0 += bw + 2
        pdf.ln(8)

        # Session meta line
        pdf.set_font("Helvetica", "", 8)
        pdf.set_text_color(80, 80, 80)
        mode    = session.get("mode", "-").upper()
        count   = session.get("analysis_count", 0)
        model   = session.get("model", "gemma4:4b")
        pdf.cell(0, 6,
                 f"Mode: {mode}   Duration: {dur_str}   "
                 f"Gemma analyses: {count}   Model: {model}",
                 ln=True)

        pdf.ln(3)
        pdf.set_draw_color(0, 180, 140)
        pdf.set_line_width(0.4)
        pdf.line(15, pdf.get_y(), 15 + W, pdf.get_y())
        pdf.ln(4)

        # ── Alert log table ───────────────────────────────────────────────────
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(20, 20, 20)
        pdf.cell(0, 7,
                 f"Alert Log  ({total} event{'s' if total != 1 else ''})",
                 ln=True)
        pdf.ln(1)

        if not alerts:
            pdf.set_font("Helvetica", "I", 9)
            pdf.set_text_color(140, 140, 140)
            pdf.cell(0, 8, "No alerts recorded during this period.", ln=True)
        else:
            # Column widths: Time | Classification | Severity | Resp | Note
            cols = [26, 44, 26, 10, 74]
            hdrs = ["Time", "Classification", "Severity", "R?", "Clinical Note"]

            pdf.set_fill_color(15, 80, 65)
            pdf.set_text_color(255, 255, 255)
            pdf.set_font("Helvetica", "B", 8)
            for hdr, cw in zip(hdrs, cols):
                pdf.cell(cw, 6, hdr, fill=True, border=0)
            pdf.ln(6)

            pdf.set_font("Helvetica", "", 8)
            alt = False
            for a in sorted(alerts, key=lambda x: x.get("timestamp", 0)):
                sev  = a.get("severity", "normal")
                cls  = a.get("cry_classification", "unknown")
                note = (a.get("clinical_note") or a.get("action_recommended") or "")[:68]
                resp = "Yes" if a.get("respiratory_concern") else "-"
                rc, gc, bc = SEV_RGB.get(sev, (60, 60, 60))

                r_bg, g_bg, b_bg = (242, 250, 247) if alt else (255, 255, 255)
                pdf.set_fill_color(r_bg, g_bg, b_bg)
                pdf.set_text_color(60, 60, 60)
                pdf.cell(cols[0], 5.5, _ts(a.get("timestamp", 0)), fill=True)

                pdf.set_text_color(30, 30, 30)
                pdf.cell(cols[1], 5.5, CLS_LABEL.get(cls, cls), fill=True)

                pdf.set_text_color(rc, gc, bc)
                pdf.set_font("Helvetica", "B", 8)
                pdf.cell(cols[2], 5.5, SEV_TAG.get(sev, sev.upper()), fill=True)
                pdf.set_font("Helvetica", "", 8)

                pdf.cell(cols[3], 5.5, resp, fill=True, align="C")

                pdf.set_text_color(60, 60, 60)
                pdf.cell(cols[4], 5.5, note, fill=True)
                pdf.ln(5.5)
                alt = not alt

        pdf.ln(5)
        pdf.set_draw_color(200, 200, 200)
        pdf.set_line_width(0.3)
        pdf.line(15, pdf.get_y(), 15 + W, pdf.get_y())
        pdf.ln(4)

        # ── Classification breakdown ───────────────────────────────────────────
        if alerts:
            pdf.set_font("Helvetica", "B", 11)
            pdf.set_text_color(20, 20, 20)
            pdf.cell(0, 7, "Classification Breakdown", ln=True)

            cls_counts = Counter(a.get("cry_classification", "unknown") for a in alerts)
            bar_w = 80

            pdf.set_font("Helvetica", "", 9)
            for cls, cnt in sorted(cls_counts.items(), key=lambda x: -x[1]):
                pct = cnt / max(total, 1)
                label = CLS_LABEL.get(cls, cls)
                is_urgent = cls in ("stridor", "respiratory_distress", "pain", "wheezing")
                rc, gc, bc = (200, 30, 30) if is_urgent else (40, 140, 90)

                pdf.set_text_color(60, 60, 60)
                pdf.cell(52, 5.5, label)
                pdf.cell(10, 5.5, str(cnt), align="R")
                pdf.cell(4,  5.5, "")

                # Background bar
                pdf.set_fill_color(220, 240, 235)
                pdf.cell(bar_w, 4, "", fill=True)
                bar_x = pdf.get_x() - bar_w
                bar_y = pdf.get_y()

                # Coloured fill
                if pct > 0:
                    pdf.set_fill_color(rc, gc, bc)
                    pdf.rect(bar_x, bar_y, bar_w * pct, 4, style="F")

                # Percentage label
                pdf.set_xy(bar_x + bar_w, bar_y)
                pdf.set_text_color(80, 80, 80)
                pdf.cell(14, 5.5, f" {round(pct*100)}%", ln=True)

            pdf.ln(4)

        # ── Reasoning snapshot (last 3 notable alerts) ─────────────────────────
        notable = [a for a in alerts if a.get("severity") in ("urgent","critical")][-3:]
        if notable:
            pdf.set_font("Helvetica", "B", 11)
            pdf.set_text_color(20, 20, 20)
            pdf.cell(0, 7, "Clinical Reasoning Snapshot (last 3 notable alerts)", ln=True)

            for a in notable:
                sev = a.get("severity", "normal")
                rc, gc, bc = SEV_RGB.get(sev, (60,60,60))
                pdf.set_font("Helvetica", "B", 9)
                pdf.set_text_color(rc, gc, bc)
                cls = CLS_LABEL.get(a.get("cry_classification","unknown"), "Unknown")
                pdf.set_x(15)
                pdf.cell(0, 6,
                         f"{SEV_TAG.get(sev,'')} {cls}  @  {_ts(a.get('timestamp',0))}",
                         ln=True)
                pdf.set_font("Helvetica", "", 8)
                pdf.set_text_color(60, 60, 60)
                pdf.set_x(15)
                pdf.multi_cell(W, 4.5, a.get("reasoning","")[:180])
                pdf.set_font("Helvetica", "I", 8)
                pdf.set_text_color(100, 100, 100)
                pdf.set_x(15)
                pdf.multi_cell(W, 4.5,
                               f"Action: {a.get('action_recommended','')[:120]}")
                pdf.ln(3)

        # ── Disclaimer ────────────────────────────────────────────────────────
        pdf.ln(2)
        pdf.set_font("Helvetica", "I", 7)
        pdf.set_text_color(160, 160, 160)
        pdf.set_x(15)
        pdf.multi_cell(W, 4.5,
            "CLINICAL DISCLAIMER: EchoGlow is an AI-assisted monitoring aid powered by "
            "Gemma 4 (Google DeepMind) running locally via Ollama. It does NOT replace "
            "qualified clinical judgment. All alerts must be evaluated by competent medical "
            "personnel. This report is for informational purposes only and must not be used "
            "as the sole basis for any clinical decision. Patient data never leaves the "
            "local device.")

        # ── Page footer ───────────────────────────────────────────────────────
        pdf.set_y(-12)
        pdf.set_font("Helvetica", "", 7)
        pdf.set_text_color(180, 180, 180)
        pdf.cell(0, 5,
                 f"EchoGlow v1.0  |  Gemma 4 Challenge  |  "
                 f"Page {pdf.page_no()}  |  {_dt(generated_at)}",
                 align="C")

        return bytes(pdf.output())
