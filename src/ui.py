import tkinter as tk
from tkinter import ttk

# ── Palette — matched to PPT ──────────────────────────────────────────────────
BG_DEEP     = "#0D1117"   # deepest background
BG_CARD     = "#161B27"   # card surfaces
BG_PANEL    = "#1C2333"   # inner panels
BORDER_DIM  = "#2A3555"   # subtle borders
BORDER_LIT  = "#1E6FA5"   # lit borders (cyan-blue family)

CYAN        = "#00D4FF"   # primary accent — matches PPT highlight color
CYAN_DIM    = "#0EA5C9"   # secondary cyan
BLUE        = "#1E6FA5"   # structural blue
WHITE       = "#F0F4FF"   # primary text
GREY        = "#8B9BB4"   # secondary text
GREY_DIM    = "#4A5568"   # muted text

RISK_PALETTE = {
    "High Risk"     : {"accent": "#FF4D6D", "bg": "#1E0D14", "border": "#8B1A2E", "label": "#FF6B84"},
    "Moderate Risk" : {"accent": "#FFB347", "bg": "#1E1508", "border": "#8B5A0A", "label": "#FFC96B"},
    "Low Risk"      : {"accent": CYAN,      "bg": "#081520", "border": BLUE,      "label": CYAN},
    "Normal"        : {"accent": "#4ADE80", "bg": "#081510", "border": "#1A5C30", "label": "#6EE89A"},
}

FONT_DISPLAY  = ("Segoe UI", 22, "bold")
FONT_HEAD     = ("Segoe UI", 11, "bold")
FONT_SUBHEAD  = ("Segoe UI", 9,  "bold")
FONT_BODY     = ("Segoe UI", 9)
FONT_SMALL    = ("Segoe UI", 8)
FONT_NUM      = ("Segoe UI", 20, "bold")
FONT_NUM_SM   = ("Segoe UI", 13, "bold")
FONT_LABEL    = ("Segoe UI", 7,  "bold")


def show_results(summary_df, eval_metrics: dict | None = None):
    root = tk.Tk()
    root.title("CodeBlooded — Retail Protection Risk Engine")
    root.configure(bg=BG_DEEP)
    root.resizable(True, True)

    w, h = 860, 680
    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    root.geometry(f"{w}x{h}+{(sw - w) // 2}+{(sh - h) // 2}")

    # ── Scrollable container ──────────────────────────────────────────────────
    canvas = tk.Canvas(root, bg=BG_DEEP, highlightthickness=0)
    scrollbar = ttk.Scrollbar(root, orient="vertical", command=canvas.yview)
    canvas.configure(yscrollcommand=scrollbar.set)
    scrollbar.pack(side="right", fill="y")
    canvas.pack(side="left", fill="both", expand=True)

    frame = tk.Frame(canvas, bg=BG_DEEP)
    fid = canvas.create_window((0, 0), window=frame, anchor="nw")
    canvas.bind("<Configure>", lambda e: canvas.itemconfig(fid, width=e.width))
    frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    root.bind_all("<MouseWheel>", lambda e: canvas.yview_scroll(-1 * (e.delta // 120), "units"))

    # ── Header bar ────────────────────────────────────────────────────────────
    _header(frame)

    # ── Stock result cards ────────────────────────────────────────────────────
    section = tk.Frame(frame, bg=BG_DEEP)
    section.pack(fill="x", padx=36, pady=(4, 0))

    tk.Label(section, text="STOCK RISK ANALYSIS",
             font=FONT_LABEL, bg=BG_DEEP, fg=CYAN).pack(anchor="w", pady=(20, 10))

    for _, row in summary_df.iterrows():
        _stock_card(section, row)

    # ── Model performance ─────────────────────────────────────────────────────
    if eval_metrics:
        metrics_section = tk.Frame(frame, bg=BG_DEEP)
        metrics_section.pack(fill="x", padx=36, pady=(24, 0))
        tk.Label(metrics_section, text="MODEL PERFORMANCE",
                 font=FONT_LABEL, bg=BG_DEEP, fg=CYAN).pack(anchor="w", pady=(0, 10))
        _metrics_panel(metrics_section, eval_metrics)

    # ── Footer ────────────────────────────────────────────────────────────────
    _footer(frame, summary_df)

    root.mainloop()


# ── Header ────────────────────────────────────────────────────────────────────

def _header(parent):
    bar = tk.Frame(parent, bg=BG_CARD, pady=0)
    bar.pack(fill="x")

    # Top cyan line
    tk.Frame(bar, bg=CYAN, height=2).pack(fill="x")

    inner = tk.Frame(bar, bg=BG_CARD, padx=36, pady=18)
    inner.pack(fill="x")

    left = tk.Frame(inner, bg=BG_CARD)
    left.pack(side="left")

    tk.Label(left, text="RETAIL PROTECTION RISK ENGINE",
             font=("Segoe UI", 14, "bold"), bg=BG_CARD, fg=WHITE).pack(anchor="w")
    tk.Label(left,
             text="IsolationForest · Hybrid Scoring · NSE Market Data",
             font=FONT_SMALL, bg=BG_CARD, fg=GREY).pack(anchor="w", pady=(3, 0))

    right = tk.Frame(inner, bg=BG_CARD)
    right.pack(side="right")

    badge = tk.Frame(right, bg=BLUE, padx=12, pady=6)
    badge.pack()
    tk.Label(badge, text="CodeBlooded", font=("Segoe UI", 9, "bold"),
             bg=BLUE, fg=WHITE).pack()


# ── Stock card ────────────────────────────────────────────────────────────────

def _stock_card(parent, row):
    cat    = row["risk_category"]
    pal    = RISK_PALETTE.get(cat, RISK_PALETTE["Normal"])

    CARD_BORDER = "#00D4FF"   # light blue border — same for all cards
    CARD_INNER  = "#080C10"   # pure black interior

    # Outer border frame
    outer = tk.Frame(parent, bg=CARD_BORDER, padx=1, pady=1)
    outer.pack(fill="x", pady=6)

    card = tk.Frame(outer, bg=CARD_INNER, padx=22, pady=18)
    card.pack(fill="both")

    # ── Top row ───────────────────────────────────────────────────────────────
    top = tk.Frame(card, bg=CARD_INNER)
    top.pack(fill="x")

    left_top = tk.Frame(top, bg=CARD_INNER)
    left_top.pack(side="left")

    tk.Label(left_top, text=row["symbol"],
             font=("Segoe UI", 16, "bold"), bg=CARD_INNER, fg=WHITE).pack(anchor="w")

    flag_text  = "⚠  Manipulation Detected" if row["is_manipulated"] else "✓  No Manipulation Detected"
    flag_color = pal["label"] if row["is_manipulated"] else "#4ADE80"
    tk.Label(left_top, text=flag_text,
             font=("Segoe UI", 8, "bold"), bg=CARD_INNER, fg=flag_color).pack(anchor="w", pady=(4, 0))

    # Risk badge — keeps its risk color so category is still visible
    right_top = tk.Frame(top, bg=CARD_INNER)
    right_top.pack(side="right", anchor="n")

    badge_frame = tk.Frame(right_top, bg=pal["border"], padx=10, pady=4)
    badge_frame.pack()
    tk.Label(badge_frame, text=cat.upper(),
             font=("Segoe UI", 8, "bold"), bg=pal["border"], fg=pal["label"]).pack()

    # ── Divider ───────────────────────────────────────────────────────────────
    tk.Frame(card, bg="#1A2535", height=1).pack(fill="x", pady=(14, 14))

    # ── Metrics row ───────────────────────────────────────────────────────────
    metrics_row = tk.Frame(card, bg=CARD_INNER)
    metrics_row.pack(fill="x")

    _stat_block(metrics_row, "AVG RISK SCORE",    f"{row['avg_risk_score']:.1f}",    "/100", pal)
    _vsep(metrics_row, CARD_INNER)
    _stat_block(metrics_row, "PEAK RISK SCORE",   f"{row['max_risk_score']:.1f}",    "/100", pal)
    _vsep(metrics_row, CARD_INNER)
    _stat_block(metrics_row, "MANIPULATION RATE", f"{row['manipulation_rate']:.1f}", "%",    pal)

    # ── Risk bar ──────────────────────────────────────────────────────────────
    tk.Frame(card, bg="#1A2535", height=1).pack(fill="x", pady=(14, 10))
    _risk_bar(card, row["max_risk_score"], pal, CARD_INNER)


def _stat_block(parent, label, value, suffix, pal):
    block = tk.Frame(parent, bg=BG_PANEL, padx=18, pady=12)
    block.pack(side="left", padx=(0, 2))

    tk.Label(block, text=label, font=FONT_LABEL,
             bg=BG_PANEL, fg=GREY).pack(anchor="w")

    row = tk.Frame(block, bg=BG_PANEL)
    row.pack(anchor="w", pady=(4, 0))

    tk.Label(row, text=value, font=FONT_NUM,
             bg=BG_PANEL, fg=pal["accent"]).pack(side="left")
    tk.Label(row, text=suffix, font=FONT_SMALL,
             bg=BG_PANEL, fg=GREY).pack(side="left", padx=(3, 0), pady=(6, 0))


def _vsep(parent, bg):
    tk.Frame(parent, bg=BORDER_DIM, width=1).pack(side="left", fill="y", padx=2)


def _risk_bar(parent, score, pal, bg="#080C10"):
    row = tk.Frame(parent, bg=bg)
    row.pack(fill="x")

    tk.Label(row, text="PEAK RISK LEVEL", font=FONT_LABEL,
             bg=bg, fg=GREY).pack(side="left", padx=(0, 10))

    track = tk.Frame(row, bg=BORDER_DIM, height=5)
    track.pack(side="left", fill="x", expand=True)

    pct = min(score / 100, 1.0)

    def _fill(attempts=0):
        track.update_idletasks()
        w = track.winfo_width()
        if w > 10:
            fw = max(int(w * pct), 4)
            tk.Frame(track, bg=pal["accent"], width=fw, height=5).place(x=0, y=0)
        elif attempts < 10:
            track.after(80, lambda: _fill(attempts + 1))

    track.after(120, _fill)

    tk.Label(row, text=f"{score:.0f} / 100", font=FONT_SMALL,
             bg=bg, fg=pal["label"]).pack(side="left", padx=(10, 0))


# ── Model metrics panel ───────────────────────────────────────────────────────

def _metrics_panel(parent, m):
    panel = tk.Frame(parent, bg=BG_CARD, padx=0, pady=0)
    panel.pack(fill="x")

    tk.Frame(panel, bg=CYAN, height=1).pack(fill="x")

    inner = tk.Frame(panel, bg=BG_CARD, padx=24, pady=20)
    inner.pack(fill="x")

    # ── Top counts row ────────────────────────────────────────────────────────
 
    # ── Score row ─────────────────────────────────────────────────────────────
    scores = tk.Frame(inner, bg=BG_CARD)
    scores.pack(fill="x")

    entries = [
        #("PRECISION",  m.get("precision"), "of flags that were real fraud"),
        #("RECALL",     m.get("recall"),    "of real fraud days caught"),
        ("F1 SCORE",   m.get("f1"),        "harmonic mean"),
        ("AUPRC",      m.get("auprc"),     f"baseline ≈ {m.get('baseline', 0.078):.3f}"),
        ("ROC-AUC",    m.get("roc_auc"),   "0.5 = random"),
    ]

    for label, val, hint in entries:
        if val is None:
            continue
        col = tk.Frame(scores, bg=BG_PANEL, padx=18, pady=12)
        col.pack(side="left", padx=(0, 3))

        color = _score_color(label, val)
        tk.Label(col, text=label, font=FONT_LABEL, bg=BG_PANEL, fg=GREY).pack(anchor="w")
        tk.Label(col, text=f"{val:.3f}", font=FONT_NUM_SM, bg=BG_PANEL, fg=color).pack(anchor="w", pady=(4, 0))
        tk.Label(col, text=hint, font=("Segoe UI", 7), bg=BG_PANEL, fg=GREY_DIM).pack(anchor="w", pady=(2, 0))


def _score_color(metric, val):
    thresholds = {
        "AUPRC"   : (0.50, 0.25),
        "ROC-AUC" : (0.70, 0.55),
    }
    hi, mid = thresholds.get(metric, (0.60, 0.30))
    if val >= hi:
        return "#4ADE80"
    elif val >= mid:
        return "#FFB347"
    return "#FF6B84"


# ── Footer ────────────────────────────────────────────────────────────────────

def _footer(parent, summary_df):
    tk.Frame(parent, bg=BORDER_DIM, height=1).pack(fill="x", padx=36, pady=(28, 0))

    foot = tk.Frame(parent, bg=BG_DEEP, padx=36, pady=16)
    foot.pack(fill="x")

    tk.Label(foot,
             text=f"Stocks analysed: {len(summary_df)}   ·   "
                  f"Flagged: {int(summary_df['is_manipulated'].sum())}   ·   "
                  f"Ground truth: NSE bulk deal flags   ·   "
                  f"Labels are proxy signals, not legal findings",
             font=FONT_SMALL, bg=BG_DEEP, fg=GREY_DIM).pack(anchor="w")