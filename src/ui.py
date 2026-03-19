import tkinter as tk
from tkinter import ttk
import threading


# ── Palette ──────────────────────────────────────────────────────────────────
BG          = "#0D0F14"
CARD_BG     = "#13161E"
BORDER      = "#1E2230"
TEXT_PRI    = "#E8EAF0"
TEXT_SEC    = "#6B7280"
TEXT_MONO   = "#A0A8B8"
ACCENT      = "#4F6EF7"

RISK_COLORS = {
    "High Risk"     : {"bg": "#2A1018", "border": "#C0392B", "text": "#FF6B6B", "dot": "#FF4444"},
    "Moderate Risk" : {"bg": "#2A1F0A", "border": "#D4800A", "text": "#FFB347", "dot": "#FF8C00"},
    "Low Risk"      : {"bg": "#0A1F2A", "border": "#1A7A9A", "text": "#5BC8F5", "dot": "#3AB4E8"},
    "Normal"        : {"bg": "#111A14", "border": "#1A6B30", "text": "#4ADE80", "dot": "#22C55E"},
}

FONT_TITLE  = ("Courier New", 11, "bold")
FONT_LABEL  = ("Courier New", 9)
FONT_MONO   = ("Courier New", 10, "bold")
FONT_SMALL  = ("Courier New", 8)
FONT_HEAD   = ("Courier New", 13, "bold")


def show_results(summary_df, eval_metrics: dict | None = None):
    """
    Opens a tkinter window showing per-stock manipulation results.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Must have: symbol, is_manipulated, avg_risk_score, max_risk_score,
                   manipulation_rate, risk_category
    eval_metrics : dict | None
        Optional — keys: precision, recall, f1, auprc, roc_auc,
                         total_days, fraud_days, flagged
    """

    root = tk.Tk()
    root.title("Stock Manipulation Detector")
    root.configure(bg=BG)
    root.resizable(True, True)

    # Center window on screen
    w, h = 780, 600
    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    root.geometry(f"{w}x{h}+{(sw - w) // 2}+{(sh - h) // 2}")

    # ── Outer scroll container ────────────────────────────────────────────────
    canvas = tk.Canvas(root, bg=BG, highlightthickness=0)
    scrollbar = ttk.Scrollbar(root, orient="vertical", command=canvas.yview)
    canvas.configure(yscrollcommand=scrollbar.set)

    scrollbar.pack(side="right", fill="y")
    canvas.pack(side="left", fill="both", expand=True)

    frame = tk.Frame(canvas, bg=BG)
    frame_id = canvas.create_window((0, 0), window=frame, anchor="nw")

    def _on_resize(event):
        canvas.itemconfig(frame_id, width=event.width)
    canvas.bind("<Configure>", _on_resize)

    def _on_frame_change(event):
        canvas.configure(scrollregion=canvas.bbox("all"))
    frame.bind("<Configure>", _on_frame_change)

    root.bind_all("<MouseWheel>", lambda e: canvas.yview_scroll(-1 * (e.delta // 120), "units"))

    # ── Header ────────────────────────────────────────────────────────────────
    header = tk.Frame(frame, bg=BG, pady=24)
    header.pack(fill="x", padx=32)

    tk.Label(
        header, text="◈  STOCK MANIPULATION DETECTOR",
        font=("Courier New", 15, "bold"),
        bg=BG, fg=ACCENT
    ).pack(anchor="w")

    tk.Label(
        header,
        text="AI-powered anomaly detection  ·  IsolationForest + rule-based scoring",
        font=FONT_SMALL, bg=BG, fg=TEXT_SEC
    ).pack(anchor="w", pady=(4, 0))

    _divider(frame)

    # ── Stock cards ───────────────────────────────────────────────────────────
    tk.Label(
        frame, text="ANALYSIS RESULTS",
        font=FONT_SMALL, bg=BG, fg=TEXT_SEC
    ).pack(anchor="w", padx=32, pady=(20, 8))

    for _, row in summary_df.iterrows():
        _stock_card(frame, row)

    # ── Model metrics panel (only if eval ran) ────────────────────────────────
    if eval_metrics:
        _divider(frame)
        tk.Label(
            frame, text="MODEL PERFORMANCE",
            font=FONT_SMALL, bg=BG, fg=TEXT_SEC
        ).pack(anchor="w", padx=32, pady=(20, 8))
        _metrics_panel(frame, eval_metrics)

    # ── Footer ────────────────────────────────────────────────────────────────
    tk.Label(
        frame,
        text="Ground truth: NSE bulk deal flags  ·  Labels are proxy signals, not legal findings",
        font=FONT_SMALL, bg=BG, fg=TEXT_SEC
    ).pack(pady=(24, 8))

    tk.Label(
        frame, text=f"Stocks analysed: {len(summary_df)}  ·  "
                    f"Flagged: {summary_df['is_manipulated'].sum()}",
        font=FONT_SMALL, bg=BG, fg=TEXT_SEC
    ).pack(pady=(0, 24))

    root.mainloop()


# ── Stock card ────────────────────────────────────────────────────────────────

def _stock_card(parent, row):
    cat    = row["risk_category"]
    colors = RISK_COLORS.get(cat, RISK_COLORS["Normal"])

    outer = tk.Frame(parent, bg=BORDER, padx=1, pady=1)
    outer.pack(fill="x", padx=32, pady=6)

    card = tk.Frame(outer, bg=colors["bg"], padx=20, pady=16)
    card.pack(fill="both")

    # Top row: symbol + category badge
    top = tk.Frame(card, bg=colors["bg"])
    top.pack(fill="x")

    # Dot indicator
    tk.Label(top, text="●", font=("Courier New", 14),
             bg=colors["bg"], fg=colors["dot"]).pack(side="left", padx=(0, 8))

    tk.Label(top, text=row["symbol"],
             font=("Courier New", 14, "bold"),
             bg=colors["bg"], fg=TEXT_PRI).pack(side="left")

    # Badge
    badge = tk.Frame(top, bg=colors["border"], padx=8, pady=2)
    badge.pack(side="right")
    tk.Label(badge, text=cat.upper(),
             font=("Courier New", 8, "bold"),
             bg=colors["border"], fg=colors["text"]).pack()

    # Manipulated flag
    flag_text = "⚠  MANIPULATION DETECTED" if row["is_manipulated"] else "✓  NO MANIPULATION DETECTED"
    flag_color = colors["text"] if row["is_manipulated"] else "#4ADE80"
    tk.Label(card, text=flag_text,
             font=("Courier New", 9, "bold"),
             bg=colors["bg"], fg=flag_color).pack(anchor="w", pady=(10, 12))

    # Metric row
    metrics = tk.Frame(card, bg=colors["bg"])
    metrics.pack(fill="x")

    _metric_box(metrics, "AVG RISK",     f"{row['avg_risk_score']:.1f}",  "/100", colors)
    _metric_box(metrics, "PEAK RISK",    f"{row['max_risk_score']:.1f}",  "/100", colors)
    _metric_box(metrics, "MANIP. RATE",  f"{row['manipulation_rate']:.1f}", "%",  colors)

    # Risk bar
    tk.Frame(card, bg=BORDER, height=1).pack(fill="x", pady=(14, 8))
    _risk_bar(card, row["avg_risk_score"], colors)


def _metric_box(parent, label, value, suffix, colors):
    box = tk.Frame(parent, bg=CARD_BG, padx=14, pady=10)
    box.pack(side="left", padx=(0, 8))

    tk.Label(box, text=label,
             font=FONT_SMALL, bg=CARD_BG, fg=TEXT_SEC).pack(anchor="w")

    val_row = tk.Frame(box, bg=CARD_BG)
    val_row.pack(anchor="w")
    tk.Label(val_row, text=value,
             font=("Courier New", 18, "bold"),
             bg=CARD_BG, fg=colors["text"]).pack(side="left")
    tk.Label(val_row, text=suffix,
             font=FONT_SMALL, bg=CARD_BG, fg=TEXT_SEC).pack(side="left", padx=(2, 0))


def _risk_bar(parent, score, colors):
    bar_frame = tk.Frame(parent, bg=colors["bg"])
    bar_frame.pack(fill="x")

    tk.Label(bar_frame, text="RISK LEVEL",
             font=FONT_SMALL, bg=colors["bg"], fg=TEXT_SEC).pack(anchor="w")

    track = tk.Frame(bar_frame, bg=BORDER, height=6)
    track.pack(fill="x", pady=(4, 0))
    track.update_idletasks()

    fill_pct = min(score / 100, 1.0)

    def _draw():
        width = track.winfo_width()
        if width > 1:
            fill_w = max(int(width * fill_pct), 4)
            tk.Frame(track, bg=colors["dot"], width=fill_w, height=6).place(x=0, y=0)

    track.after(50, _draw)


# ── Model metrics panel ────────────────────────────────────────────────────────

def _metrics_panel(parent, m):
    panel = tk.Frame(parent, bg=CARD_BG, padx=24, pady=20)
    panel.pack(fill="x", padx=32, pady=(0, 8))

    # Top stats row
    stats = tk.Frame(panel, bg=CARD_BG)
    stats.pack(fill="x", pady=(0, 16))

    for label, value in [
        ("TOTAL DAYS",    str(m.get("total_days", "—"))),
        ("FRAUD DAYS",    str(m.get("fraud_days", "—"))),
        ("FLAGGED",       str(m.get("flagged", "—"))),
        ("TRUE HITS",     str(m.get("tp", "—"))),
        ("MISSED",        str(m.get("fn", "—"))),
    ]:
        col = tk.Frame(stats, bg=CARD_BG, padx=16)
        col.pack(side="left")
        tk.Label(col, text=label, font=FONT_SMALL, bg=CARD_BG, fg=TEXT_SEC).pack(anchor="w")
        tk.Label(col, text=value, font=("Courier New", 16, "bold"),
                 bg=CARD_BG, fg=TEXT_PRI).pack(anchor="w")

    tk.Frame(panel, bg=BORDER, height=1).pack(fill="x", pady=(0, 14))

    # Score row
    scores = tk.Frame(panel, bg=CARD_BG)
    scores.pack(fill="x")

    metrics_to_show = [
        ("PRECISION",  m.get("precision"), "of flags raised that were real"),
        ("RECALL",     m.get("recall"),    "of fraud days caught"),
        ("F1 SCORE",   m.get("f1"),        "harmonic mean"),
        ("AUPRC",      m.get("auprc"),     f"baseline ≈ {m.get('baseline', 0.078):.3f}"),
        ("ROC-AUC",    m.get("roc_auc"),   "0.5 = random"),
    ]

    for label, val, hint in metrics_to_show:
        if val is None:
            continue
        col = tk.Frame(scores, bg=CARD_BG, padx=12)
        col.pack(side="left")

        color = _score_color(label, val)
        tk.Label(col, text=label, font=FONT_SMALL, bg=CARD_BG, fg=TEXT_SEC).pack(anchor="w")
        tk.Label(col, text=f"{val:.3f}", font=("Courier New", 16, "bold"),
                 bg=CARD_BG, fg=color).pack(anchor="w")
        tk.Label(col, text=hint, font=FONT_SMALL, bg=CARD_BG, fg=TEXT_SEC).pack(anchor="w")


def _score_color(metric, val):
    if metric == "AUPRC":
        return "#4ADE80" if val >= 0.5 else "#FFB347" if val >= 0.25 else "#FF6B6B"
    if metric in ("PRECISION", "RECALL", "F1 SCORE"):
        return "#4ADE80" if val >= 0.6 else "#FFB347" if val >= 0.3 else "#FF6B6B"
    if metric == "ROC-AUC":
        return "#4ADE80" if val >= 0.7 else "#FFB347" if val >= 0.55 else "#FF6B6B"
    return TEXT_MONO


# ── Helpers ───────────────────────────────────────────────────────────────────

def _divider(parent):
    tk.Frame(parent, bg=BORDER, height=1).pack(fill="x", padx=32, pady=(8, 0))