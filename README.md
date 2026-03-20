<div align="center">

<h1>🛡️ AI Powered Retail Protection Risk Engine</h1>

<p><strong>An unsupervised machine learning system that detects stock market manipulation and pump-and-dump schemes on NSE-listed Indian equities.</strong></p>

<p>
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/scikit--learn-IsolationForest-orange?style=flat-square&logo=scikitlearn&logoColor=white"/>
  <img src="https://img.shields.io/badge/Market-NSE%20India-blue?style=flat-square"/>
  <img src="https://img.shields.io/badge/Status-Prototype-cyan?style=flat-square"/>
  <img src="https://img.shields.io/badge/Team-CodeBlooded-blueviolet?style=flat-square"/>
</p>

<br/>

> **ByteVerse 1.0 Hackathon** · Team CodeBlooded · ICFAI University, Dehradun  
> Theme: AI for Digital Finance & Cyber Security

</div>

---

## 📌 The Problem

India's retail investors lose **crores every year** to manipulated microcap stocks. Pump-and-dump operators inflate a stock's price with fake volume and coordinated buying, then exit — leaving retail investors holding worthless shares.

| Scam | Loss |
|------|------|
| Harshad Mehta (1992) | ₹1,00,000+ Cr market erosion |
| Ketan Parekh (2001) | ₹1,200–2,000 Cr |
| Sahara India (2010s) | ₹24,000 Cr raised illegally |
| Microcap P&D (2015–present) | ₹500–2,000+ Cr per cycle |

**No accessible, real-time tool exists to warn retail investors before they enter a manipulated stock.**

---

## 🧠 How It Works

The system combines a rule-based signal engine with an unsupervised ML model (IsolationForest) into a hybrid scoring pipeline. Every stock is scored daily across 6 engineered manipulation signals. The ML model learns anomaly patterns; the rule engine provides interpretable flags. Both are blended into a single final risk score.

### Mathematical Workflow

![Workflow Diagram](docs/workflow_diagram.png)

### Pipeline Summary

```
Raw OHLCV + Delivery Data
        ↓
  Feature Engineering (6 signals)
        ↓
  Manipulation Score S ∈ {0…6}
        ↓
  ┌─────────────────┐    ┌──────────────────┐
  │ IsolationForest │    │  Rule Normaliser │
  │  decision_fn()  │    │  S / S_max × 100 │
  └────────┬────────┘    └────────┬─────────┘
           └──────────┬───────────┘
                      ↓
      final = 0.6 × ML + 0.4 × Rule
                      ↓
       Normal → Low Risk → Moderate → High Risk
```

---

## ⚙️ Feature Engineering

Six manipulation signals are engineered per trading day:

| Feature | Formula | What it detects |
|---------|---------|-----------------|
| **Volume multiplier** | `V_t / avg(V, 20d)` | Abnormal trading activity |
| **Upper circuit streak** | Consecutive days with `pct_change ≥ 4.9%` | Coordinated price ramping |
| **Delivery divergence** | `price > 5%` AND `delivery% < 5d avg` | Speculative / intraday buying |
| **Price–volume correlation** | `rolling_corr(P, V, 10d) < -0.3` | Price moving without genuine demand |
| **Volume–price divergence** | `volume > 2× avg` AND `pct_change ≤ 0` | Distribution phase (smart money exiting) |
| **Price z-score** | `(P - μ₂₀) / σ₂₀ > 2` | Statistically extreme price move |

A composite **manipulation score S** is the integer sum of all signals (max = 6).  
`is_manipulated = 1` if `S ≥ 2`.

---

## 🤖 Model

```python
IsolationForest(
    n_estimators  = 200,
    contamination = 0.10,   # matches ~7.8% observed fraud rate
    random_state  = 42
)
```

The model trains **only on engineered features** — raw OHLCV columns are intentionally excluded. IsolationForest would otherwise treat normal price levels as anomalies.

### Hybrid Scoring

```
R_ml   = minmax_norm(decision_function output) × 100   [inverted — high = anomalous]
R_rule = (S / S_max) × 100

final_risk_score = 0.6 × R_ml + 0.4 × R_rule
```

### Risk Categories

| Score | Category |
|-------|----------|
| ≥ 80 | 🔴 High Risk |
| 60–79 | 🟠 Moderate Risk |
| 30–59 | 🔵 Low Risk |
| < 30 | 🟢 Normal |

---

## 📊 Evaluation

Ground truth labels are derived from `bulk_deal_flag == YES` rows in NSE data (57 fraud-labeled days across 730 total, 7.8% positive rate).

| Metric | Value | Notes |
|--------|-------|-------|
| AUPRC | 0.154 | Baseline ≈ 0.078 (2× random) |
| ROC-AUC | 0.561 | Above random (0.5) |
| Precision | 0.203 | 1 in 5 flags is real fraud |
| Recall | 0.211 | Catching ~21% of fraud days |
| F1 | 0.207 | |

> **AUPRC is the primary metric** for imbalanced fraud detection. Random baseline on this dataset is 0.078. The model is at 0.154 — 2× random with only 2 labeled stocks and no feature scaling yet.

---

## 🖥️ UI

A tkinter desktop window launches after analysis with per-stock cards showing risk scores, manipulation rate, and a peak risk level bar.

The terminal retains full verbose output including missed fraud days, false alarm counts, and the complete evaluation report.

---

## 🏗️ Tech Stack

![Tech Stack](docs/tech_stack_diagram.png)

| Layer | Tool |
|-------|------|
| Data source | NSE India, Kaggle |
| Data processing | Pandas, NumPy |
| ML model | scikit-learn (IsolationForest) |
| Market data API | yfinance |
| UI | tkinter |
| Storage *(planned)* | PostgreSQL |
| Adaptive learning *(planned)* | Feedback loop on new labeled data |

---

## 📁 Project Structure

```
AI_fake_stock_detector/
│
├── src/
│   ├── main.py            # Orchestrator — runs the full pipeline
│   ├── data_loader.py     # Loads all stock CSVs from data/
│   ├── feature.py         # Engineers 6 manipulation signals
│   ├── train.py           # Trains / loads IsolationForest model
│   ├── risk_scoring.py    # Hybrid ML + rule blending
│   ├── ground_truth.py    # Extracts fraud labels from bulk_deal_flag
│   ├── evaluate.py        # Precision, recall, AUPRC evaluation
│   └── ui.py              # tkinter results window
│
├── data/
│   ├── cleaned_PCJEWELLER_PumpDump.csv
│   ├── cleaned_RPOWER_PumpDump_Synthetic_365Days.csv
│   └── labeled_ground_truth.csv   # auto-generated
│
├── models/
│   └── isolation_forest_YYYY_MM_DD.pkl
│
├── docs/
│   ├── workflow_diagram.png
│   └── tech_stack_diagram.png
│
└── requirements.txt
```

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install -r requirements.txt
```

```
pandas
numpy
scikit-learn
yfinance
joblib
matplotlib
```

### Running

```bash
cd src
python main.py
```

The pipeline will:
1. Load all CSVs from `data/`
2. Fetch live market cap from yfinance
3. Engineer manipulation features
4. Train (or load) the IsolationForest model
5. Score every stock daily, aggregate to per-stock summary
6. Build ground truth labels and run evaluation
7. Launch the results UI window

### Force retrain

In `main.py`, set:
```python
model = train_model(df, MODEL_PATH, force_retrain=True)
```

---

## 📈 Known Limitations & Roadmap

| Limitation | Fix planned |
|------------|-------------|
| `bulk_deal_flag` labels include clean institutional trades | Composite label: bulk deal AND price spike AND volume spike |
| No feature scaling | `StandardScaler` before IsolationForest fit |
| Train and eval on same data | Time-based train/test split (70/30 on dates) |
| Only 2 labeled stocks | Expand to 20+ NSE microcap stocks |
| Static CSV input | Dynamic yfinance pipeline by ticker + date range |
| `add_pump_dump_features()` unused | Wire in pump strength, dump strength, z-score features |

---

## 🔬 Why This Matters

Most manipulation detection tools are:
- Proprietary (SEBI internal systems)
- Expensive (Bloomberg Terminal)
- Reactive (post-facto enforcement)

This engine is designed to be **retail-first, real-time, and embeddable** — a risk score API that any broker platform (Groww, Zerodha, Upstox) could surface directly to retail investors before they buy.

---

## 👥 Team

**CodeBlooded** · ByteVerse 1.0 · ICFAI University, Dehradun

| Name | Role |
|------|------|
| Sameer Husain | ML pipeline, feature engineering, evaluation |
| Anwesha Rudra | Data collection, cleaning, research |
| Aarav Raj | Architecture, UI, presentation |

---



---

<div align="center">
  <sub>Built with ❤️ at ByteVerse 1.0 · ICFAI University Dehradun</sub>
</div>
