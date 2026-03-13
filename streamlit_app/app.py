"""
CreditAI – Dashboard de Prédiction de Crédit
Projet 7 – OpenClassRoom
Conformité WCAG 2.1 AA (palette IBM colorblind-safe, labels textuels, descriptions accessibles)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import requests
import base64
import json
from pathlib import Path

# ════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="CreditAI Dashboard",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session State ────────────────────────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = "landing"
if "active_view" not in st.session_state:
    st.session_state.active_view = "overview"
if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None
if "pred_loaded_sk_id" not in st.session_state:
    st.session_state.pred_loaded_sk_id = None

# ════════════════════════════════════════════════════════════════════════════
# CHEMINS
# ════════════════════════════════════════════════════════════════════════════
_BASE     = Path(__file__).parent
DATA_PATH = _BASE / "data" / "sample_clients.csv"
PRED_PATH = _BASE / "data" / "predictions.csv"
_CFG_PATH = _BASE / "data" / "config.json"
if not DATA_PATH.exists():
    st.error("⚠️ Fichier sample_clients.csv introuvable dans data/")
    st.stop()

# ════════════════════════════════════════════════════════════════════════════
# CONFIGURATION DYNAMIQUE  (config.json généré par le notebook)
# ════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def load_config() -> dict:
    """Charge la config du modèle depuis config.json (généré par le notebook)."""
    if _CFG_PATH.exists():
        with open(_CFG_PATH) as f:
            return json.load(f)
    # Fallback si le fichier est absent
    return {
        "threshold":     0.09594,
        "score_thresh":  904,
        "model_version": "LightGBM_v4",
        "auc_val":       0.785,
        "date_trained":  "N/A",
    }

CFG          = load_config()
THRESHOLD    = float(CFG.get("threshold", 0.09594))
SCORE_THRESH = int(CFG.get("score_thresh", round((1 - THRESHOLD) * 1000)))

# ════════════════════════════════════════════════════════════════════════════
# CONSTANTES
# ════════════════════════════════════════════════════════════════════════════
API_URL = "https://projet7-mlops-implementation-modele.onrender.com/predict"

# ── Palette WCAG 2.1 AA – IBM colorblind-safe ────────────────────────────────
# Deutéranopie + protanopie + tritanopie compatibles
# Ne jamais utiliser rouge/vert seuls pour distinguer des informations
C_ACCEPTED = "#0072B2"  # Bleu         → Accordé / Faible risque
C_REFUSED = "#D55E00"  # Vermillon    → Refusé / Risque élevé
C_WARNING = "#E69F00"  # Ambre        → Risque modéré
C_NEUTRAL = "#6b7280"  # Gris         → Neutre / Médiane
C_PRIMARY = "#2563eb"  # Bleu marque  → Boutons, accentuation
C_DARK = "#111827"
C_BG = "#f9fafb"
C_WHITE = "#ffffff"
C_BORDER = "#e5e7eb"

# ── Variables comparaison (sous-ensemble lisible) ─────────────────────────────
COMPARE_FEATURES = {
    "AGE_YEARS": "Âge (années)",
    "EMPLOYMENT_YEARS": "Ancienneté emploi (années)",
    "AMT_INCOME_TOTAL": "Revenu annuel (€)",
    "AMT_CREDIT": "Montant du crédit (€)",
    "AMT_ANNUITY": "Annuité mensuelle (€/mois)",
    "EXT_SOURCE_2": "Score externe 2",
    "EXT_SOURCE_3": "Score externe 3",
    "CNT_CHILDREN": "Nombre d'enfants",
    "CNT_FAM_MEMBERS": "Membres du foyer",
    "AMT_GOODS_PRICE": "Prix du bien (€)",
}

# ── Données statiques (vue d'ensemble) ───────────────────────────────────────
RISK_DATA = [
    {"name": "Faible", "value": 65, "color": C_ACCEPTED},
    {"name": "Moyen", "value": 25, "color": C_WARNING},
    {"name": "Élevé", "value": 10, "color": C_REFUSED},
]
APPROVAL_DATA = {
    "mois": ["Jan", "Fév", "Mar", "Avr", "Mai", "Juin"],
    "taux": [82, 85, 88, 84, 90, 87],
}
SCORE_DATA = {
    "range": ["0-300", "300-500", "500-700", "700-1000"],
    "count": [45, 120, 180, 95],
}
HISTORY_DATA = pd.DataFrame(
    {
        "ID": ["#1234", "#1235", "#1236", "#1237", "#1238", "#1239", "#1240"],
        "Date": [
            "05/02/2026",
            "04/02/2026",
            "03/02/2026",
            "03/02/2026",
            "02/02/2026",
            "02/02/2026",
            "01/02/2026",
        ],
        "Client": [
            "Jean Dupont",
            "Marie Martin",
            "Pierre Bernard",
            "Sophie Leclerc",
            "Luc Petit",
            "Émilie Richard",
            "Thomas Moreau",
        ],
        "Montant": [
            "50 000 €",
            "25 000 €",
            "35 000 €",
            "18 000 €",
            "42 000 €",
            "15 000 €",
            "28 000 €",
        ],
        "Durée": [
            "36 mois",
            "24 mois",
            "48 mois",
            "12 mois",
            "60 mois",
            "24 mois",
            "36 mois",
        ],
        "Score": [742, 812, 565, 695, 485, 778, 652],
        "Décision": [
            "Approuvé",
            "Approuvé",
            "Refusé",
            "Approuvé",
            "Refusé",
            "Approuvé",
            "Approuvé",
        ],
    }
)


# ════════════════════════════════════════════════════════════════════════════
# CHARGEMENT DONNÉES
# ════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def load_clients() -> pd.DataFrame:
    """Charge et enrichit le fichier client (100 dossiers pré-traités)."""
    if not DATA_PATH.exists():
        return pd.DataFrame()
    df = pd.read_csv(DATA_PATH)
    # Variables dérivées lisibles
    if "DAYS_BIRTH" in df.columns:
        df["AGE_YEARS"] = (-df["DAYS_BIRTH"] / 365).round(1)
    if "DAYS_EMPLOYED" in df.columns:
        df["EMPLOYMENT_YEARS"] = df["DAYS_EMPLOYED"].apply(
            lambda x: round(-x / 365, 1) if pd.notna(x) and x < 0 else 0.0
        )
    if "CODE_GENDER" in df.columns:
        df["GENDER_LABEL"] = (
            df["CODE_GENDER"].map({1: "Homme", 0: "Femme"}).fillna("N/R")
        )
    return df


@st.cache_data(show_spinner=False)
def load_predictions() -> pd.DataFrame:
    """Charge les scores pré-calculés (LightGBM local, instantané)."""
    if PRED_PATH.exists():
        return pd.read_csv(PRED_PATH)
    return pd.DataFrame()


def get_client_result(client_idx: int) -> dict:
    """Retourne le résultat du modèle pour un client (depuis CSV pré-calculé)."""
    preds = load_predictions()
    if preds.empty or client_idx >= len(preds):
        return {"error": "no_predictions", "probability": None, "decision": None}
    row = preds.iloc[client_idx]
    return {
        "probability": float(row["probability"]),
        "decision": int(row["decision"]),
        "score": int(row["score"]),
        "threshold": float(row["threshold"]),
    }


# ════════════════════════════════════════════════════════════════════════════
# FONCTIONS UTILITAIRES
# ════════════════════════════════════════════════════════════════════════════
def interpret_risk(probability: float) -> tuple:
    """Retourne (label_court, explication_html, couleur)."""
    seuil_pct = THRESHOLD * 100
    prob_pct = probability * 100
    margin = abs(probability - THRESHOLD) * 100

    if probability < THRESHOLD * 0.5:
        return (
            "Risque très faible",
            (
                f"Ce client présente un risque <strong>très faible</strong> de défaut de paiement. "
                f"Sa probabilité de défaut ({prob_pct:.1f}%) est bien en dessous du seuil de décision "
                f"({seuil_pct:.1f}%), avec une marge de sécurité de <strong>{margin:.1f} points</strong>."
            ),
            C_ACCEPTED,
        )
    elif probability < THRESHOLD:
        return (
            "Risque faible",
            (
                f"Ce client présente un <strong>risque faible</strong> de défaut. "
                f"Sa probabilité ({prob_pct:.1f}%) est en dessous du seuil d'acceptation ({seuil_pct:.1f}%), "
                f"avec une marge de <strong>{margin:.1f} points</strong>."
            ),
            C_ACCEPTED,
        )
    elif probability < THRESHOLD * 1.5:
        return (
            "Risque modéré",
            (
                f"Ce client présente un <strong>risque modéré</strong>. Sa probabilité de défaut ({prob_pct:.1f}%) "
                f"dépasse le seuil de décision ({seuil_pct:.1f}%) de <strong>{margin:.1f} points</strong>. "
                f"Une analyse approfondie est recommandée avant toute décision."
            ),
            C_WARNING,
        )
    else:
        return (
            "Risque élevé",
            (
                f"Ce client présente un <strong>risque élevé</strong> de défaut de paiement. "
                f"Sa probabilité ({prob_pct:.1f}%) dépasse significativement le seuil ({seuil_pct:.1f}%) "
                f"de <strong>{margin:.1f} points</strong>."
            ),
            C_REFUSED,
        )


def format_value(feature: str, value) -> str:
    """Formate une valeur brute en texte lisible."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "N/A"
    v = float(value)
    if feature in {"AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE"}:
        return f"{v:,.0f} €"
    if feature in {"AGE_YEARS", "EMPLOYMENT_YEARS"}:
        return f"{v:.1f} ans"
    if feature in {"EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"}:
        return f"{v:.3f}"
    if feature in {"FLAG_OWN_REALTY", "FLAG_OWN_CAR", "CODE_GENDER"}:
        return "Oui" if v == 1 else "Non"
    return f"{v:.1f}"


# ════════════════════════════════════════════════════════════════════════════
# SVG ICONS (Lucide-style, 24×24, stroke-based)
# ════════════════════════════════════════════════════════════════════════════
_ICON_GRID = (
    "<rect x='3' y='3' width='7' height='7'/>"
    "<rect x='14' y='3' width='7' height='7'/>"
    "<rect x='14' y='14' width='7' height='7'/>"
    "<rect x='3' y='14' width='7' height='7'/>"
)
_ICON_USER = (
    "<path d='M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2'/>"
    "<circle cx='12' cy='7' r='4'/>"
)
_ICON_FILEPLUS = (
    "<path d='M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z'/>"
    "<polyline points='14 2 14 8 20 8'/>"
    "<line x1='12' y1='18' x2='12' y2='12'/>"
    "<line x1='9' y1='15' x2='15' y2='15'/>"
)
_ICON_CLOCK = "<circle cx='12' cy='12' r='10'/><polyline points='12 6 12 12 16 14'/>"
_ICON_SETTINGS = (
    "<circle cx='12' cy='12' r='3'/>"
    "<path d='M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06"
    "a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09"
    "A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06"
    "A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09"
    "A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06"
    "A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09"
    "a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06"
    "A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z'/>"
)
_ICON_LOGOUT = (
    "<path d='M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4'/>"
    "<polyline points='16 17 21 12 16 7'/>"
    "<line x1='21' y1='12' x2='9' y2='12'/>"
)


def _svg_b64(paths: str, color: str = "#64748b") -> str:
    """Return base64-encoded SVG data URI for CSS url()."""
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" '
        f'stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
        f"{paths}</svg>"
    )
    return base64.b64encode(svg.encode()).decode()


# ════════════════════════════════════════════════════════════════════════════
# CSS
# ════════════════════════════════════════════════════════════════════════════
def inject_landing_css():
    st.markdown(
        """
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
        * { font-family: 'Inter', sans-serif !important; }
        .stApp { background: linear-gradient(135deg, #1e3a8a 0%, #1e40af 50%, #111827 100%) !important; }
        [data-testid="stSidebar"]  { display: none !important; }
        [data-testid="stHeader"]   { background: transparent !important; }
        [data-testid="stToolbar"]  { display: none !important; }
        .landing-logo {
            width:120px; height:120px;
            background: linear-gradient(135deg, #3b82f6, #1d4ed8);
            border-radius:30px; display:flex; align-items:center; justify-content:center;
            margin:0 auto 36px; transform:rotate(8deg);
            box-shadow:0 0 50px rgba(59,130,246,.55), 0 16px 40px rgba(0,0,0,.3);
            animation:float 4s ease-in-out infinite, glow 3s ease-in-out infinite;
        }
        @keyframes float { 0%,100% { transform:rotate(8deg) translateY(0px); } 50% { transform:rotate(8deg) translateY(-10px); } }
        @keyframes glow  { 0%,100% { box-shadow:0 0 50px rgba(59,130,246,.55),0 16px 40px rgba(0,0,0,.3); } 50% { box-shadow:0 0 90px rgba(59,130,246,.85),0 20px 60px rgba(0,0,0,.4); } }
        .landing-badge  { display:inline-flex; align-items:center; gap:8px; padding:7px 18px; background:rgba(255,255,255,.1); backdrop-filter:blur(12px); border:1px solid rgba(255,255,255,.18); border-radius:24px; color:rgba(255,255,255,.92); font-size:13px; font-weight:500; margin-bottom:18px; letter-spacing:.2px; }
        .landing-title  { font-size:76px; font-weight:800; text-align:center; margin-bottom:14px; letter-spacing:-3px; color:white; }
        .landing-subtitle { font-size:20px; color:rgba(255,255,255,.65); text-align:center; font-weight:300; margin-bottom:48px; letter-spacing:.2px; }
        .landing-stats  { display:flex; justify-content:center; gap:0px; margin-top:52px; border-top:1px solid rgba(255,255,255,.1); padding-top:40px; }
        .landing-stat   { text-align:center; flex:1; max-width:180px; }
        .landing-stat + .landing-stat {{ border-left:1px solid rgba(255,255,255,.1); }}
        .landing-stat-value { font-size:42px; font-weight:800; color:white; text-shadow:0 2px 10px rgba(0,0,0,.25); letter-spacing:-1px; }
        .landing-stat-label { font-size:11px; color:rgba(255,255,255,.5); text-transform:uppercase; letter-spacing:1.2px; font-weight:600; margin-top:4px; }
        .stButton > button {
            background:white !important; color:#1d4ed8 !important; border:none !important;
            border-radius:14px !important; padding:18px 48px !important; font-size:17px !important;
            font-weight:700 !important; box-shadow:0 8px 32px rgba(0,0,0,.25) !important;
            transition:all .3s cubic-bezier(.4,0,.2,1) !important; height:auto !important; min-height:58px !important;
            letter-spacing:.1px !important;
        }
        .stButton > button:hover { transform:translateY(-3px) !important; box-shadow:0 16px 48px rgba(0,0,0,.35) !important; }
    </style>""",
        unsafe_allow_html=True,
    )


def inject_dashboard_css(active_view):
    # DOM : stVerticalBlock > stElementContainer:nth-child(n)
    # 1=logo, 2=nav-label, 3=overview, 4=client, 5=prediction, 6=history
    # 7=spacer, 8=hr, 9=settings, 10=logout
    nav_child = {"overview": 3, "client": 4, "prediction": 5, "history": 6}
    active_child = nav_child.get(active_view, 3)

    # ── Build per-button SVG icon CSS ─────────────────────────────────────────
    _nav_icons = [
        (3, _ICON_GRID),
        (4, _ICON_USER),
        (5, _ICON_FILEPLUS),
        (6, _ICON_CLOCK),
        (9, _ICON_SETTINGS),
        (10, _ICON_LOGOUT),
    ]
    icon_rules = ""
    for child, paths in _nav_icons:
        clr = "#93c5fd" if child == active_child else "#64748b"
        b64 = _svg_b64(paths, clr)
        icon_rules += (
            f'\n        [data-testid="stSidebarUserContent"] [data-testid="stVerticalBlock"]'
            f' > [data-testid="stElementContainer"]:nth-child({child})'
            f' button[data-testid="stBaseButton-secondary"]::before {{'
            f'\n            background-image:url("data:image/svg+xml;base64,{b64}") !important;'
            f"\n        }}"
        )

    st.markdown(
        f"""
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
        /* Force light color-scheme to prevent OS/browser dark mode on BaseWeb */
        :root {{ color-scheme:light !important; }}
        * {{ font-family:'Inter', sans-serif !important; box-sizing:border-box; color-scheme:light !important; }}
        .stApp {{ background-color:#f3f4f6 !important; }}
        [data-testid="stHeader"]  {{ background:transparent !important; }}
        [data-testid="stToolbar"] {{ display:none !important; }}
        .main .block-container {{ padding-top:0 !important; padding-bottom:40px !important; max-width:100% !important; }}

        /* ════════════════════════════════════════════════════
           SIDEBAR
        ════════════════════════════════════════════════════ */
        [data-testid="stSidebar"] {{ background-color:#0f172a !important; border-right:none !important; box-shadow:2px 0 12px rgba(0,0,0,.15); }}
        [data-testid="stSidebarContent"] {{ background-color:#0f172a !important; padding:0 !important; }}
        [data-testid="stSidebar"] > div:first-child {{ background-color:#0f172a !important; }}

        /* Nav items – default */
        [data-testid="stSidebar"] button[data-testid="stBaseButton-secondary"] {{
            background-color:transparent !important; color:#64748b !important;
            border:none !important; border-radius:8px !important; border-left:3px solid transparent !important;
            padding:11px 14px 11px 42px !important; font-size:14px !important; font-weight:500 !important;
            text-align:left !important; width:100% !important; height:auto !important;
            min-height:44px !important; justify-content:flex-start !important;
            transition:all .18s ease !important; position:relative !important;
        }}
        [data-testid="stSidebar"] button[data-testid="stBaseButton-secondary"]:hover {{
            background-color:rgba(255,255,255,.06) !important; color:#e2e8f0 !important;
            border-left-color:rgba(255,255,255,.2) !important;
        }}
        /* Nav item actif – left accent + subtle glow */
        [data-testid="stSidebarUserContent"] [data-testid="stVerticalBlock"] > [data-testid="stElementContainer"]:nth-child({active_child}) button[data-testid="stBaseButton-secondary"] {{
            background-color:rgba(37,99,235,.15) !important; color:#93c5fd !important;
            border-left:3px solid #3b82f6 !important; font-weight:600 !important;
        }}
        [data-testid="stSidebarUserContent"] [data-testid="stVerticalBlock"] > [data-testid="stElementContainer"]:nth-child({active_child}) button[data-testid="stBaseButton-secondary"]:hover {{
            background-color:rgba(37,99,235,.2) !important;
        }}

        /* Nav icon ::before shared base */
        [data-testid="stSidebarUserContent"] [data-testid="stVerticalBlock"] > [data-testid="stElementContainer"]:nth-child(3) button[data-testid="stBaseButton-secondary"]::before,
        [data-testid="stSidebarUserContent"] [data-testid="stVerticalBlock"] > [data-testid="stElementContainer"]:nth-child(4) button[data-testid="stBaseButton-secondary"]::before,
        [data-testid="stSidebarUserContent"] [data-testid="stVerticalBlock"] > [data-testid="stElementContainer"]:nth-child(5) button[data-testid="stBaseButton-secondary"]::before,
        [data-testid="stSidebarUserContent"] [data-testid="stVerticalBlock"] > [data-testid="stElementContainer"]:nth-child(6) button[data-testid="stBaseButton-secondary"]::before,
        [data-testid="stSidebarUserContent"] [data-testid="stVerticalBlock"] > [data-testid="stElementContainer"]:nth-child(9) button[data-testid="stBaseButton-secondary"]::before,
        [data-testid="stSidebarUserContent"] [data-testid="stVerticalBlock"] > [data-testid="stElementContainer"]:nth-child(10) button[data-testid="stBaseButton-secondary"]::before {{
            content:''; position:absolute; left:14px; top:50%; transform:translateY(-50%);
            width:18px; height:18px; background-size:contain; background-repeat:no-repeat;
            background-position:center; pointer-events:none;
        }}
        {icon_rules}

        /* ════════════════════════════════════════════════════
           HEADER
        ════════════════════════════════════════════════════ */
        .dash-header {{
            background:white;
            border-bottom:1px solid #e5e7eb;
            padding:18px 32px;
            margin-bottom:28px;
            display:flex; justify-content:space-between; align-items:center;
            box-shadow:0 1px 4px rgba(0,0,0,.06);
        }}
        .dash-header-title {{ font-size:19px; font-weight:700; color:#111827; margin-bottom:2px; letter-spacing:-.3px; }}
        .dash-header-subtitle {{ font-size:13px; color:#9ca3af; font-weight:400; }}

        /* ════════════════════════════════════════════════════
           KPI CARDS
        ════════════════════════════════════════════════════ */
        .kpi-card {{
            background:white; border:1px solid #e5e7eb; border-radius:14px;
            padding:20px 20px 16px; box-shadow:0 1px 3px rgba(0,0,0,.07);
            min-height:130px; display:flex; justify-content:space-between; align-items:flex-start;
            transition:all .25s cubic-bezier(.4,0,.2,1);
            border-top:3px solid var(--kpi-accent,#e5e7eb);
        }}
        .kpi-card:hover {{ transform:translateY(-3px); box-shadow:0 12px 28px -6px rgba(0,0,0,.12); }}
        .kpi-label {{ font-size:12px; color:#9ca3af; font-weight:600; text-transform:uppercase; letter-spacing:.5px; margin-bottom:10px; }}
        .kpi-value {{ font-size:30px; font-weight:800; color:#111827; line-height:1; margin-bottom:10px; letter-spacing:-.5px; }}
        .kpi-icon  {{ width:44px; height:44px; border-radius:10px; display:flex; align-items:center; justify-content:center; font-size:20px; flex-shrink:0; }}
        .kpi-delta {{ font-size:12px; font-weight:600; display:flex; align-items:center; gap:3px; }}
        .kpi-delta-label {{ font-size:11px; color:#9ca3af; font-weight:400; }}

        /* ════════════════════════════════════════════════════
           PERFORMANCE CARD
        ════════════════════════════════════════════════════ */
        .perf-card {{
            background:white; border:1px solid #e5e7eb; border-radius:14px;
            padding:24px; box-shadow:0 1px 3px rgba(0,0,0,.07);
        }}
        .perf-title {{ font-size:16px; font-weight:700; color:#111827; margin-bottom:20px; letter-spacing:-.2px; }}
        .metric-grid {{ display:grid; grid-template-columns:repeat(4,1fr); gap:12px; }}
        .metric-item {{
            background:linear-gradient(135deg,#f8faff,#eff6ff);
            border:1px solid #dbeafe;
            padding:18px 14px; border-radius:10px; text-align:center; transition:all .2s;
        }}
        .metric-item:hover {{ background:linear-gradient(135deg,#eff6ff,#dbeafe); transform:translateY(-2px); box-shadow:0 4px 12px rgba(37,99,235,.12); }}
        .metric-val  {{ font-size:28px; font-weight:800; color:#2563eb; line-height:1; margin-bottom:5px; letter-spacing:-.5px; }}
        .metric-name {{ font-size:13px; font-weight:600; color:#1e40af; margin-bottom:3px; }}
        .metric-desc {{ font-size:11px; color:#64748b; }}

        /* ════════════════════════════════════════════════════
           CHART CARDS
        ════════════════════════════════════════════════════ */
        .chart-card {{
            background:white; border:1px solid #e5e7eb; border-radius:14px;
            padding:24px; box-shadow:0 1px 3px rgba(0,0,0,.07); transition:all .25s;
        }}
        .chart-card:hover {{ transform:translateY(-2px); box-shadow:0 8px 24px -4px rgba(0,0,0,.1); }}
        .chart-title {{ font-size:16px; font-weight:700; color:#111827; margin-bottom:3px; letter-spacing:-.2px; }}
        .chart-subtitle {{ font-size:12px; color:#9ca3af; margin-bottom:12px; }}

        /* ════════════════════════════════════════════════════
           SECTION LABEL
        ════════════════════════════════════════════════════ */
        .section-label {{
            font-size:11px; font-weight:700; color:#6366f1; text-transform:uppercase;
            letter-spacing:1px; margin-bottom:16px; display:flex; align-items:center; gap:8px;
        }}
        .section-label::before {{
            content:''; width:3px; height:14px; background:linear-gradient(180deg,#6366f1,#3b82f6);
            border-radius:2px; display:inline-block;
        }}

        /* ════════════════════════════════════════════════════
           PROFILE CARDS
        ════════════════════════════════════════════════════ */
        .profile-card {{
            background:white; border:1px solid #e5e7eb; border-radius:12px;
            padding:16px 10px; text-align:center; margin-bottom:10px;
            box-shadow:0 1px 3px rgba(0,0,0,.05); transition:all .2s;
        }}
        .profile-card:hover {{ transform:translateY(-2px); box-shadow:0 6px 16px rgba(0,0,0,.09); border-color:#c7d2fe; }}

        /* ════════════════════════════════════════════════════
           FORMS – INPUTS
        ════════════════════════════════════════════════════ */
        .stTextInput input, [data-testid="stNumberInput"] input {{
            border:1px solid #e2e8f0 !important; border-radius:8px !important;
            font-size:14px !important; color:#111827 !important; background:white !important;
            padding:10px 14px !important; transition:border-color .15s, box-shadow .15s !important;
        }}
        .stTextInput input:focus, [data-testid="stNumberInput"] input:focus {{
            border-color:#3b82f6 !important; box-shadow:0 0 0 3px rgba(59,130,246,.12) !important;
            outline:none !important;
        }}
        .stTextInput label, [data-testid="stNumberInput"] label,
        [data-testid="stSelectbox"] label {{
            font-size:13px !important; font-weight:600 !important; color:#475569 !important;
            text-transform:uppercase !important; letter-spacing:.4px !important;
        }}

        /* Masquer +/- number inputs */
        button[data-testid="stNumberInputStepDown"],
        button[data-testid="stNumberInputStepUp"] {{ display:none !important; }}

        /* Selectbox trigger */
        [data-testid="stSelectbox"] > div > div {{
            background-color:white !important; border:1px solid #e2e8f0 !important;
            border-radius:8px !important; color:#111827 !important;
        }}
        [data-testid="stSelectbox"] [data-baseweb="select"] > div {{
            background-color:white !important; border:1px solid #e2e8f0 !important;
            border-radius:8px !important; color:#111827 !important; min-height:44px !important;
        }}
        [data-testid="stSelectbox"] [data-baseweb="select"] > div:hover {{
            border-color:#3b82f6 !important;
        }}
        [data-testid="stSelectbox"] span, [data-testid="stSelectbox"] div[class*="placeholder"] {{
            color:#111827 !important;
        }}

        /* Dropdown menu (BaseWeb + ARIA) – force white regardless of color-scheme */
        [data-baseweb="menu"] {{
            background-color:white !important; border:1px solid #e5e7eb !important;
            border-radius:10px !important; box-shadow:0 8px 24px rgba(0,0,0,.12) !important;
        }}
        [data-baseweb="menu"] li {{
            background-color:white !important; color:#374151 !important;
            font-size:14px !important; padding:10px 14px !important;
        }}
        [data-baseweb="menu"] li:hover {{ background-color:#f1f5ff !important; color:#2563eb !important; }}
        [role="listbox"] {{
            background-color:white !important; border:1px solid #e5e7eb !important;
            border-radius:10px !important; box-shadow:0 8px 24px rgba(0,0,0,.12) !important;
        }}
        [role="option"] {{
            background-color:white !important; color:#374151 !important;
            font-size:14px !important; padding:10px 14px !important;
        }}
        [role="option"]:hover, [role="option"][aria-selected="true"] {{
            background-color:#f1f5ff !important; color:#2563eb !important;
        }}

        /* ════════════════════════════════════════════════════
           BUTTONS
        ════════════════════════════════════════════════════ */
        /* Primary CTA */
        .predict-wrap .stButton > button {{
            background:linear-gradient(135deg,#3b82f6,#2563eb) !important; color:white !important;
            border:none !important; border-radius:10px !important; font-size:15px !important;
            font-weight:700 !important; padding:14px !important; height:auto !important;
            min-height:50px !important; box-shadow:0 4px 12px rgba(37,99,235,.35) !important;
            letter-spacing:.1px !important; transition:all .2s !important;
        }}
        .predict-wrap .stButton > button:hover {{
            box-shadow:0 6px 20px rgba(37,99,235,.45) !important; transform:translateY(-1px) !important;
        }}

        /* All secondary buttons default to white (sidebar overrides via higher specificity) */
        button[data-testid="stBaseButton-secondary"] {{
            background:white !important; color:#374151 !important;
            border:1px solid #d1d5db !important; border-radius:8px !important;
            font-size:14px !important; font-weight:600 !important;
            padding:10px 16px !important; height:auto !important; min-height:40px !important;
            box-shadow:0 1px 2px rgba(0,0,0,.05) !important; transition:all .15s !important;
        }}
        button[data-testid="stBaseButton-secondary"]:hover {{
            background:#f9fafb !important; border-color:#9ca3af !important;
            box-shadow:0 2px 6px rgba(0,0,0,.08) !important;
        }}

        /* Sidebar permanently open – force open & hide collapse controls */
        [data-testid="stSidebarCollapseButton"] {{ display:none !important; }}
        [data-testid="collapsedControl"]        {{ display:none !important; }}
        button[data-testid="stBaseButton-headerNoPadding"] {{ display:none !important; }}
        /* Override Streamlit's collapsed-state transform so sidebar is always visible */
        section[data-testid="stSidebar"],
        section[data-testid="stSidebar"][aria-expanded="false"] {{
            transform:none !important; margin-left:0 !important;
            left:0 !important; min-width:244px !important;
            visibility:visible !important; pointer-events:auto !important;
        }}
        section[data-testid="stSidebar"] > div:first-child,
        section[data-testid="stSidebar"][aria-expanded="false"] > div:first-child {{
            transform:none !important; transition:none !important;
            min-width:244px !important;
        }}
        [data-testid="stSidebarContent"] {{
            display:flex !important; opacity:1 !important; visibility:visible !important;
        }}

        /* Sidebar HR */
        [data-testid="stSidebar"] hr {{ border-color:rgba(255,255,255,.08) !important; margin:8px 16px !important; }}

        /* Input placeholder text – visible gray */
        [data-testid="stNumberInput"] input::placeholder,
        .stTextInput input::placeholder {{
            color:#9ca3af !important; opacity:1 !important;
        }}

        /* Streamlit metrics */
        [data-testid="stMetric"] {{ background:white; border:1px solid #e5e7eb; border-radius:10px; padding:14px 16px; }}
        [data-testid="stMetricLabel"] {{ font-size:12px !important; color:#6b7280 !important; font-weight:600 !important; text-transform:uppercase !important; letter-spacing:.4px !important; }}
        [data-testid="stMetricValue"] {{ font-size:22px !important; font-weight:800 !important; color:#111827 !important; letter-spacing:-.3px !important; }}
        [data-testid="stMetricDelta"] {{ font-size:12px !important; font-weight:600 !important; }}
    </style>""",
        unsafe_allow_html=True,
    )


# ════════════════════════════════════════════════════════════════════════════
# GRAPHIQUES
# ════════════════════════════════════════════════════════════════════════════
def make_donut_chart():
    """Donut – Distribution des risques (palette WCAG)."""
    fig = go.Figure(
        go.Pie(
            labels=[d["name"] for d in RISK_DATA],
            values=[d["value"] for d in RISK_DATA],
            hole=0.65,
            marker=dict(
                colors=[d["color"] for d in RISK_DATA], line=dict(color="#fff", width=2)
            ),
            textinfo="none",
            hovertemplate="<b>%{label}</b><br>%{value}%<extra></extra>",
        )
    )
    fig.update_layout(
        height=260,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="white",
        plot_bgcolor="white",
        showlegend=True,
        legend=dict(
            orientation="v",
            x=0.75,
            y=0.5,
            xanchor="left",
            yanchor="middle",
            font=dict(size=13, color="#374151"),
            itemsizing="constant",
        ),
    )
    return fig


def make_line_chart():
    """Courbe – Tendances d'approbation."""
    fig = go.Figure(
        go.Scatter(
            x=APPROVAL_DATA["mois"],
            y=APPROVAL_DATA["taux"],
            mode="lines+markers",
            line=dict(color=C_PRIMARY, width=3),
            marker=dict(size=8, color=C_PRIMARY, line=dict(color="white", width=2)),
            hovertemplate="<b>%{x}</b><br>Taux: %{y}%<extra></extra>",
            fill="tozeroy",
            fillcolor="rgba(37,99,235,.06)",
        )
    )
    fig.update_layout(
        height=280,
        margin=dict(l=52, r=16, t=16, b=36),
        paper_bgcolor="white",
        plot_bgcolor="white",
        xaxis=dict(showgrid=False, tickfont=dict(size=12, color="#6b7280")),
        yaxis=dict(
            showgrid=True,
            gridcolor="#f3f4f6",
            tickfont=dict(size=12, color="#6b7280"),
            ticksuffix="%",
            range=[70, 100],
        ),
        hovermode="x unified",
    )
    return fig


def make_bar_chart():
    """Barres – Distribution des scores."""
    fig = go.Figure(
        go.Bar(
            x=SCORE_DATA["range"],
            y=SCORE_DATA["count"],
            marker=dict(color=C_PRIMARY, line=dict(width=0)),
            hovertemplate="<b>Score %{x}</b><br>%{y} demandes<extra></extra>",
            text=SCORE_DATA["count"],
            textposition="outside",
            textfont=dict(size=13, color="#374151"),
        )
    )
    fig.update_layout(
        height=290,
        margin=dict(l=48, r=16, t=50, b=36),
        paper_bgcolor="white",
        plot_bgcolor="white",
        xaxis=dict(showgrid=False, tickfont=dict(size=12, color="#6b7280")),
        yaxis=dict(
            showgrid=True,
            gridcolor="#f3f4f6",
            tickfont=dict(size=12, color="#6b7280"),
            range=[0, max(SCORE_DATA["count"]) * 1.22],
        ),
        bargap=0.3,
    )
    return fig


def make_client_gauge(probability: float) -> go.Figure:
    """Jauge de probabilité de défaut (WCAG : palette blue/vermillion, labels numériques)."""
    is_accepted = probability < THRESHOLD
    bar_color = C_ACCEPTED if is_accepted else C_REFUSED
    seuil_pct = round(THRESHOLD * 100, 1)
    prob_pct = round(probability * 100, 1)

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=prob_pct,
            number={
                "font": {"size": 46, "color": C_DARK, "family": "Inter"},
                "suffix": "%",
            },
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickwidth": 2,
                    "tickcolor": "#d1d5db",
                    "tickvals": [0, 15, seuil_pct, 50, 75, 100],
                    "ticktext": [
                        "0%",
                        "15%",
                        f"Seuil\n{seuil_pct}%",
                        "50%",
                        "75%",
                        "100%",
                    ],
                    "tickfont": {"size": 10, "color": "#6b7280"},
                },
                "bar": {"color": bar_color, "thickness": 0.28},
                "bgcolor": "#f9fafb",
                "borderwidth": 0,
                "steps": [
                    {
                        "range": [0, seuil_pct],
                        "color": "#dbeafe",
                    },  # Zone accordée (bleu clair)
                    {
                        "range": [seuil_pct, 55],
                        "color": "#fef3c7",
                    },  # Zone de vigilance (ambre clair)
                    {
                        "range": [55, 100],
                        "color": "#ffedd5",
                    },  # Zone refus (orange clair)
                ],
                "threshold": {
                    "line": {"color": "#374151", "width": 4},
                    "thickness": 0.85,
                    "value": seuil_pct,
                },
            },
            title={
                "text": "Probabilité de défaut<br><span style='font-size:11px;color:#6b7280'>Zone bleue = accordé · Seuil = ligne noire</span>",
                "font": {"size": 13, "color": "#6b7280"},
            },
        )
    )
    fig.update_layout(
        height=270,
        margin={"t": 50, "b": 10, "l": 20, "r": 20},
        paper_bgcolor="white",
        font={"family": "Inter"},
    )
    return fig


def make_comparison_chart(
    df: pd.DataFrame,
    feature: str,
    client_value: float,
    group_label: str = "Tous les clients",
    feature_label: str = "",
) -> go.Figure:
    """
    Histogramme WCAG : population (bleu) + marqueur client (vermillon).
    WCAG 1.4.1 : couleur + forme + texte (jamais couleur seule).
    WCAG 1.4.3 : contraste texte >= 4.5:1.
    """
    vals = df[feature].dropna()
    if len(vals) == 0:
        return go.Figure()
    percentile = (vals < client_value).mean() * 100

    fig = go.Figure()

    # Population – histogramme bleu (couleur + label)
    fig.add_trace(
        go.Histogram(
            x=vals,
            nbinsx=35,
            name=f"Population ({group_label})",
            marker=dict(
                color=C_ACCEPTED, opacity=0.60, line=dict(color="white", width=0.5)
            ),
            hovertemplate=f"Valeur: %{{x}}<br>Nombre de clients: %{{y}}<extra></extra>",
        )
    )

    # Médiane – ligne pointillée grise
    median_val = vals.median()
    fig.add_vline(
        x=median_val,
        line_color=C_NEUTRAL,
        line_width=1.5,
        line_dash="dot",
        annotation=dict(
            text=f"Médiane :<br>{format_value(feature, median_val)}",
            font=dict(size=11, color=C_NEUTRAL),
        ),
        annotation_position="bottom right",
    )

    # Client – ligne vermillon pleine + annotation texte détaillée
    fig.add_vline(
        x=client_value,
        line_color=C_REFUSED,
        line_width=3,
        line_dash="solid",
        annotation=dict(
            text=(
                f"<b>📍 Ce client</b><br>"
                f"{format_value(feature, client_value)}<br>"
                f"Percentile {percentile:.0f}e"
            ),
            bgcolor="white",
            bordercolor=C_REFUSED,
            borderwidth=2,
            borderpad=6,
            font=dict(size=12, color=C_REFUSED),
            yanchor="bottom",
        ),
        annotation_position="top",
    )

    lbl = feature_label or feature
    fig.update_layout(
        title={"text": f"Distribution : {lbl}", "font": {"size": 15, "color": C_DARK}},
        xaxis_title=lbl,
        yaxis_title="Nombre de clients",
        plot_bgcolor="white",
        paper_bgcolor="white",
        font={"family": "Inter", "size": 13},
        bargap=0.05,
        showlegend=True,
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
        },
        margin={"t": 55, "b": 40, "l": 40, "r": 20},
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="#f3f4f6")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="#f3f4f6")
    return fig


def make_score_gauge(score: int) -> go.Figure:
    """Jauge de score (page Nouvelle Prédiction)."""
    color = (
        C_ACCEPTED
        if score >= SCORE_THRESH
        else (C_WARNING if score >= 550 else C_REFUSED)
    )
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number={"font": {"size": 52, "color": C_DARK, "family": "Inter"}},
            gauge={
                "axis": {
                    "range": [0, 1000],
                    "tickfont": {"size": 11, "color": "#9ca3af"},
                    "nticks": 6,
                },
                "bar": {"color": color, "thickness": 0.25},
                "bgcolor": "#f9fafb",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 400], "color": "#ffedd5"},
                    {"range": [400, 550], "color": "#fef3c7"},
                    {"range": [550, SCORE_THRESH], "color": "#dbeafe"},
                    {"range": [SCORE_THRESH, 1000], "color": "#bbf7d0"},
                ],
                "threshold": {
                    "line": {"color": "#374151", "width": 3},
                    "thickness": 0.75,
                    "value": SCORE_THRESH,
                },
            },
        )
    )
    fig.update_layout(
        height=220,
        margin=dict(l=30, r=30, t=20, b=0),
        paper_bgcolor="white",
        font={"family": "Inter"},
    )
    return fig


# ════════════════════════════════════════════════════════════════════════════
# COMPOSANTS COMMUNS
# ════════════════════════════════════════════════════════════════════════════
def render_chart_card(title, subtitle, fig, height=360):
    chart_html = pio.to_html(
        fig, include_plotlyjs="cdn", full_html=False, config={"displayModeBar": False}
    )
    st.components.v1.html(
        f"""
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
    <div style="background:white;border:1px solid #e5e7eb;border-radius:14px;
                padding:22px 24px;box-shadow:0 1px 4px rgba(0,0,0,.06);font-family:'Inter',sans-serif;">
        <div style="font-size:15px;font-weight:700;color:#111827;margin-bottom:2px;letter-spacing:-.2px;">{title}</div>
        <div style="font-size:12px;color:#9ca3af;margin-bottom:14px;font-weight:400;">{subtitle}</div>
        {chart_html}
    </div>""",
        height=height,
    )


def render_kpi_cards():
    kpis = [
        {
            "title": "Précision du Modèle",
            "val": "94.2%",
            "delta": "+2.3%",
            "pos": True,
            "accent": "#2563eb",
            "bg": "#dbeafe",
            "icon": '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#2563eb" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>',
        },
        {
            "title": "Demandes Traitées",
            "val": "1 247",
            "delta": "+18.2%",
            "pos": True,
            "accent": "#0891b2",
            "bg": "#cffafe",
            "icon": '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#0891b2" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M23 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/></svg>',
        },
        {
            "title": "Taux d'Approbation",
            "val": "67.8%",
            "delta": "-3.1%",
            "pos": False,
            "accent": "#7c3aed",
            "bg": "#ede9fe",
            "icon": '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#7c3aed" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/></svg>',
        },
        {
            "title": "Défauts Détectés",
            "val": "42",
            "delta": "-12.5%",
            "pos": True,
            "accent": "#d97706",
            "bg": "#fef3c7",
            "icon": '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#d97706" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>',
        },
    ]
    cols = st.columns(4, gap="medium")
    for i, k in enumerate(kpis):
        dc = "#059669" if k["pos"] else "#dc2626"
        arrow = "↑" if k["pos"] else "↓"
        with cols[i]:
            st.markdown(
                f"""
            <div class="kpi-card" style="--kpi-accent:{k["accent"]};">
                <div style="flex:1;">
                    <div class="kpi-label">{k["title"]}</div>
                    <div class="kpi-value">{k["val"]}</div>
                    <div class="kpi-delta">
                        <span style="color:{dc};background:{"#f0fdf4" if k["pos"] else "#fef2f2"};
                                     padding:2px 7px;border-radius:20px;">
                            {arrow} {k["delta"]}
                        </span>
                        <span class="kpi-delta-label">vs mois dernier</span>
                    </div>
                </div>
                <div class="kpi-icon" style="background:{k["bg"]};">{k["icon"]}</div>
            </div>""",
                unsafe_allow_html=True,
            )


def render_performance():
    metrics = [
        ("94.2%", "Précision", "Prédictions correctes", "#2563eb"),
        ("91.7%", "Rappel", "Défauts détectés", "#0891b2"),
        ("92.9%", "F1-Score", "Équilibre Précision/Rappel", "#7c3aed"),
        ("0.96", "AUC-ROC", "Pouvoir discriminant", "#059669"),
    ]
    items_html = "".join(
        f"""
        <div class="metric-item">
            <div class="metric-val" style="color:{c};">{v}</div>
            <div class="metric-name">{n}</div>
            <div class="metric-desc">{d}</div>
        </div>"""
        for v, n, d, c in metrics
    )
    st.markdown(
        f"""
    <div class="perf-card">
        <div class="perf-title">
            <span style="background:linear-gradient(135deg,#6366f1,#3b82f6);-webkit-background-clip:text;
                         -webkit-text-fill-color:transparent;background-clip:text;">⚡</span>
            &nbsp;Performance du Modèle LightGBM
        </div>
        <div class="metric-grid">{items_html}</div>
    </div>""",
        unsafe_allow_html=True,
    )


# ════════════════════════════════════════════════════════════════════════════
# PAGES
# ════════════════════════════════════════════════════════════════════════════


# ── Landing ──────────────────────────────────────────────────────────────────
def show_landing():
    inject_landing_css()
    _, col, _ = st.columns([1, 2, 1])
    with col:
        st.markdown(
            """
        <div style="display:flex;flex-direction:column;align-items:center;padding:60px 0 40px;">
            <div class="landing-logo">
                <svg width="52" height="52" viewBox="0 0 24 24" fill="none"
                     stroke="white" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                    <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline>
                </svg>
            </div>
            <h1 class="landing-title">CreditAI</h1>
            <div class="landing-badge">✨ Propulsé par Machine Learning v2.1</div>
            <p class="landing-subtitle">Système Intelligent de Prédiction de Crédit</p>
        </div>""",
            unsafe_allow_html=True,
        )
        if st.button("🚀 Accéder au Dashboard", use_container_width=True):
            st.session_state.page = "dashboard"
            st.rerun()
        st.markdown(
            """
        <div class="landing-stats">
            <div class="landing-stat">
                <div class="landing-stat-value">94.2%</div>
                <div class="landing-stat-label">Précision</div>
            </div>
            <div class="landing-stat">
                <div class="landing-stat-value">1 247</div>
                <div class="landing-stat-label">Demandes traitées</div>
            </div>
            <div class="landing-stat">
                <div class="landing-stat-value">0.96</div>
                <div class="landing-stat-label">AUC-ROC</div>
            </div>
        </div>""",
            unsafe_allow_html=True,
        )


# ── Vue d'ensemble ────────────────────────────────────────────────────────────
def show_overview():
    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
    st.markdown(
        '<div class="section-label">Indicateurs clés</div>', unsafe_allow_html=True
    )
    render_kpi_cards()
    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    st.markdown(
        '<div class="section-label">Performance du modèle</div>', unsafe_allow_html=True
    )
    render_performance()
    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    st.markdown(
        '<div class="section-label">Analyses graphiques</div>', unsafe_allow_html=True
    )
    col1, col2 = st.columns(2, gap="medium")
    with col1:
        render_chart_card(
            "Distribution des Risques",
            "Répartition par niveau · Palette WCAG colorblind-safe",
            make_donut_chart(),
            height=370,
        )
    with col2:
        render_chart_card(
            "Tendances d'Approbation",
            "Évolution mensuelle du taux d'approbation (%)",
            make_line_chart(),
            height=390,
        )
    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    render_chart_card(
        "Distribution des Scores de Crédit",
        f"Nombre de demandes par tranche · Seuil d'acceptation : ≥ {SCORE_THRESH} / 1 000",
        make_bar_chart(),
        height=405,
    )


# ── Analyse Client ────────────────────────────────────────────────────────────
def show_client_analysis():
    """
    Page principale : score, probabilité, profil descriptif et comparaison population.
    Conforme WCAG 2.1 AA : palette colorblind-safe, labels textuels, descriptions accessibles.
    """
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    df = load_clients()
    if df.empty:
        st.error(
            "⚠️ Données clients non disponibles. Vérifiez `data/sample_clients.csv`."
        )
        return

    # ── Sélecteur client ──────────────────────────────────────────────────────
    n = min(len(df), 200)
    has_id = "SK_ID_CURR" in df.columns

    col_sel, col_spacer = st.columns([2, 3])
    with col_sel:
        if has_id:
            client_ids = df["SK_ID_CURR"].astype(int).tolist()[:n]
            selected_idx = st.selectbox(
                "🔍 Sélectionner un client par ID",
                range(n),
                format_func=lambda i: f"Client ID {client_ids[i]}",
                key="client_selector",
            )
        else:
            selected_idx = st.selectbox(
                "🔍 Sélectionner un client",
                range(n),
                format_func=lambda i: f"Client #{i + 1:03d}",
                key="client_selector",
            )

    client_row = df.iloc[selected_idx]

    # ── Résultat pré-calculé (CSV local, instantané) ──────────────────────────
    result = get_client_result(selected_idx)

    probability = result.get("probability")
    decision = result.get("decision")
    api_error = result.get("error")

    if api_error or probability is None:
        # Repli sur heuristique locale (démo)
        ext2 = float(client_row.get("EXT_SOURCE_2") or 0.5)
        ext3 = float(client_row.get("EXT_SOURCE_3") or 0.5)
        probability = float(np.clip(0.35 - 0.25 * ext2 - 0.15 * ext3, 0.02, 0.98))
        decision = 0 if probability < THRESHOLD else 1
        st.info(
            f"ℹ️ API temporairement indisponible ({api_error}). Score estimé localement."
        )

    score = round((1 - probability) * 1000)
    distance = probability - THRESHOLD
    risk_lbl, risk_expl, risk_color = interpret_risk(probability)

    # ══════════════════════════════════════════════════════════════════════
    # SECTION 1 – Score & Décision
    # ══════════════════════════════════════════════════════════════════════
    st.markdown(
        '<div class="section-label">Score de risque</div>', unsafe_allow_html=True
    )

    col_g, col_d = st.columns([1, 1.8], gap="medium")

    with col_g:
        st.plotly_chart(
            make_client_gauge(probability),
            use_container_width=True,
            config={"displayModeBar": False},
        )
        dist_sign = "+" if distance > 0 else ""
        st.metric(
            "Score crédit",
            f"{score} / 1 000",
            delta=f"{dist_sign}{distance * 100:.1f}% vs seuil ({THRESHOLD * 100:.1f}%)",
        )

    with col_d:
        # Bannière décision
        if decision == 0:
            bg, bdr, badge = "#eff6ff", C_ACCEPTED, "✅ CRÉDIT ACCORDÉ"
            badge_icon = "✅"
        else:
            bg, bdr, badge = "#fff7ed", C_REFUSED, "❌ CRÉDIT REFUSÉ"
            badge_icon = "❌"

        st.markdown(
            f"""
        <div role="status" aria-label="Décision de crédit : {badge}"
             style="background:{bg};border:2px solid {bdr};border-radius:14px;
                    padding:20px 24px;margin-bottom:16px;
                    box-shadow:0 4px 16px {bdr}22;">
            <div style="display:flex;align-items:center;gap:10px;margin-bottom:10px;">
                <span style="font-size:28px;">{badge_icon}</span>
                <span style="font-size:18px;font-weight:800;color:{bdr};letter-spacing:-.3px;">{badge.split(" ", 1)[1]}</span>
            </div>
            <div style="font-size:14px;color:#374151;line-height:1.8;">{risk_expl}</div>
        </div>""",
            unsafe_allow_html=True,
        )

        # Barre de probabilité (HTML pur, accessible)
        prob_pct = probability * 100
        seuil_pct = THRESHOLD * 100
        fill_clr = C_ACCEPTED if decision == 0 else C_REFUSED
        st.markdown(
            f"""
        <div style="margin:8px 0 20px 0;">
            <div style="font-size:13px;font-weight:600;color:#374151;margin-bottom:8px;">
                Probabilité de défaut :
                <span style="color:{fill_clr};font-size:15px;"> {prob_pct:.1f}%</span>
                &nbsp;·&nbsp; Seuil de décision : <strong>{seuil_pct:.1f}%</strong>
            </div>
            <div style="position:relative;background:#e5e7eb;border-radius:8px;height:22px;">
                <div style="width:{min(prob_pct, 100):.1f}%;background:{fill_clr};height:100%;
                            border-radius:8px;transition:width .5s;"></div>
                <div style="position:absolute;top:-8px;left:{seuil_pct:.1f}%;
                            width:3px;height:38px;background:#374151;border-radius:2px;
                            transform:translateX(-50%);"></div>
            </div>
            <div style="position:relative;height:22px;">
                <div style="position:absolute;left:{seuil_pct:.1f}%;transform:translateX(-50%);
                            font-size:11px;font-weight:600;color:#374151;white-space:nowrap;">
                    ↑ Seuil {seuil_pct:.1f}%
                </div>
            </div>
        </div>""",
            unsafe_allow_html=True,
        )

        m1, m2, m3 = st.columns(3)
        m1.metric("Score / 1 000", f"{score}")
        m2.metric("Niveau de risque", risk_lbl)
        m3.metric(
            "Distance seuil", f"{'+' if distance > 0 else ''}{distance * 100:.1f}%"
        )

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════
    # SECTION 2 – Profil descriptif du client
    # ══════════════════════════════════════════════════════════════════════
    st.markdown(
        '<div class="section-label">Profil descriptif du client</div>',
        unsafe_allow_html=True,
    )

    profile_items = [
        ("AGE_YEARS", "Âge", "👤", " ans"),
        ("EMPLOYMENT_YEARS", "Ancienneté emploi", "💼", " ans"),
        ("AMT_INCOME_TOTAL", "Revenus annuels", "💰", " €"),
        ("AMT_CREDIT", "Montant du crédit", "🏦", " €"),
        ("AMT_ANNUITY", "Annuité / mois", "📅", " €"),
        ("EXT_SOURCE_2", "Score externe 2", "📊", ""),
        ("EXT_SOURCE_3", "Score externe 3", "📈", ""),
        ("CNT_CHILDREN", "Enfants", "👶", ""),
        ("FLAG_OWN_REALTY", "Propriét. logement", "🏠", ""),
        ("FLAG_OWN_CAR", "Propriét. voiture", "🚗", ""),
    ]
    row1 = st.columns(5)
    row2 = st.columns(5)

    for i, (feat, lbl, icon, unit) in enumerate(profile_items):
        col = row1[i] if i < 5 else row2[i - 5]
        raw = client_row.get(feat)
        if raw is not None and not (isinstance(raw, float) and np.isnan(raw)):
            v = float(raw)
            if feat in {"FLAG_OWN_REALTY", "FLAG_OWN_CAR"}:
                display = "Oui ✅" if v == 1 else "Non ❌"
            elif feat in {"AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY"}:
                display = f"{v:,.0f} €"
            elif feat in {"AGE_YEARS", "EMPLOYMENT_YEARS"}:
                display = f"{v:.1f} ans"
            elif feat in {"EXT_SOURCE_2", "EXT_SOURCE_3"}:
                display = f"{v:.3f}"
            else:
                display = str(int(v))
        else:
            display = "N/A"
        with col:
            st.markdown(
                f"""
            <div class="profile-card">
                <div style="font-size:24px;margin-bottom:6px;">{icon}</div>
                <div style="font-size:11px;color:#6b7280;font-weight:500;
                             text-transform:uppercase;letter-spacing:.4px;margin-bottom:4px;">{lbl}</div>
                <div style="font-size:15px;font-weight:600;color:#111827;">{display}</div>
            </div>""",
                unsafe_allow_html=True,
            )

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════
    # SECTION 3 – Comparaison avec la population (WCAG)
    # ══════════════════════════════════════════════════════════════════════
    st.markdown(
        '<div class="section-label">Comparaison avec la population</div>',
        unsafe_allow_html=True,
    )

    cf_col, cg_col = st.columns(2)
    with cf_col:
        compare_feat = st.selectbox(
            "Variable à comparer",
            list(COMPARE_FEATURES.keys()),
            format_func=lambda k: COMPARE_FEATURES[k],
            key="compare_feature",
        )
    with cg_col:
        GROUP_OPTIONS = {
            "Tous les clients": None,
            "Même tranche d'âge (± 10 ans)": "age",
            "Même tranche de revenu (± 50 %)": "income",
        }
        grp_lbl = st.selectbox(
            "Groupe de comparaison", list(GROUP_OPTIONS.keys()), key="compare_group"
        )

    # Filtrage
    fdf = df.copy()
    if GROUP_OPTIONS[grp_lbl] == "age" and "AGE_YEARS" in df.columns:
        age_v = float(client_row.get("AGE_YEARS") or 40)
        fdf = df[abs(df["AGE_YEARS"] - age_v) <= 10]
    elif GROUP_OPTIONS[grp_lbl] == "income" and "AMT_INCOME_TOTAL" in df.columns:
        inc_v = float(client_row.get("AMT_INCOME_TOTAL") or 50000)
        fdf = df[
            (df["AMT_INCOME_TOTAL"] >= inc_v * 0.5)
            & (df["AMT_INCOME_TOTAL"] <= inc_v * 1.5)
        ]

    client_feat_val = client_row.get(compare_feat)
    if (
        client_feat_val is not None
        and not (isinstance(client_feat_val, float) and np.isnan(client_feat_val))
        and compare_feat in fdf.columns
    ):
        cfv = float(client_feat_val)
        fig_cmp = make_comparison_chart(
            fdf,
            compare_feat,
            cfv,
            group_label=grp_lbl,
            feature_label=COMPARE_FEATURES[compare_feat],
        )
        st.plotly_chart(
            fig_cmp, use_container_width=True, config={"displayModeBar": False}
        )

        # Description textuelle accessible (WCAG 1.1.1 Non-text Content)
        vals = fdf[compare_feat].dropna()
        percentile = (vals < cfv).mean() * 100
        median_val = vals.median()
        position = "supérieure" if cfv > median_val else "inférieure ou égale"
        st.markdown(
            f"""
        <div role="note" aria-label="Description textuelle du graphique"
             style="background:#f8fafc;border-left:4px solid {C_PRIMARY};border-radius:0 8px 8px 0;
                    padding:12px 16px;font-size:14px;color:#374151;margin-top:4px;">
            <strong>📖 Description accessible :</strong>
            Ce client se situe au <strong>{percentile:.0f}e percentile</strong>
            pour «&nbsp;{COMPARE_FEATURES[compare_feat]}&nbsp;»
            dans le groupe «&nbsp;{grp_lbl}&nbsp;» (N&nbsp;=&nbsp;{len(vals):,} clients).
            Sa valeur ({format_value(compare_feat, cfv)}) est {position}
            à la médiane du groupe ({format_value(compare_feat, median_val)}).
            La ligne <span style="color:{C_REFUSED};font-weight:600;">vermillon</span> indique
            la position du client,
            la ligne <span style="color:{C_NEUTRAL};font-weight:600;">grise pointillée</span>
            indique la médiane.
        </div>""",
            unsafe_allow_html=True,
        )
    else:
        st.info("Données non disponibles pour cette variable / ce groupe.")


# ── Nouvelle Prédiction ───────────────────────────────────────────────────────
def show_prediction():
    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

    # ── Section : Rechercher un client existant ──────────────────────────────
    df_all = load_clients()
    has_ids = not df_all.empty and "SK_ID_CURR" in df_all.columns

    if has_ids:
        st.markdown(
            '<div class="section-label">Charger un client existant</div>',
            unsafe_allow_html=True,
        )
        lc1, lc2, lc3 = st.columns([3, 1, 3], gap="small")

        with lc1:
            sk_input = st.text_input(
                "SK_ID",
                placeholder="Entrez un SK_ID client  (ex: 100002)",
                key="pred_sk_id_input",
                label_visibility="collapsed",
            )
        with lc2:
            load_clicked = st.button(
                "Charger", key="btn_load_pred", use_container_width=True
            )
        with lc3:
            if st.session_state.pred_loaded_sk_id:
                st.markdown(
                    f"""
                <div style="background:#eff6ff;border:1px solid #bfdbfe;border-radius:8px;
                            padding:9px 14px;font-size:13px;color:#1d4ed8;font-weight:600;
                            display:flex;align-items:center;gap:8px;">
                    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#1d4ed8"
                         stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/>
                        <polyline points="22 4 12 14.01 9 11.01"/>
                    </svg>
                    Client&nbsp;<strong>SK_ID&nbsp;{st.session_state.pred_loaded_sk_id}</strong>&nbsp;chargé
                </div>""",
                    unsafe_allow_html=True,
                )

        if load_clicked:
            if not sk_input.strip():
                st.warning("Entrez un SK_ID client avant de cliquer sur Charger.")
            else:
                try:
                    sk_id_int = int(float(sk_input.strip()))
                    mask = df_all["SK_ID_CURR"] == sk_id_int
                    if mask.any():
                        pos_idx = int(mask.to_numpy().argmax())
                        row = df_all.iloc[pos_idx]

                        def _si(v, d=0):
                            return (
                                d
                                if (
                                    v is None
                                    or (isinstance(v, float) and np.isnan(float(v)))
                                )
                                else int(float(v))
                            )

                        st.session_state["age"] = max(18, _si(row.get("AGE_YEARS"), 35))
                        st.session_state["years"] = max(
                            0, _si(row.get("EMPLOYMENT_YEARS"), 0)
                        )
                        st.session_state["income"] = max(
                            0, _si(row.get("AMT_INCOME_TOTAL"), 0)
                        )
                        st.session_state["loan"] = max(0, _si(row.get("AMT_CREDIT"), 0))
                        st.session_state["debt"] = max(
                            0, _si(row.get("AMT_ANNUITY"), 0)
                        )
                        st.session_state["children"] = max(
                            0, _si(row.get("CNT_CHILDREN"), 0)
                        )

                        ext2 = row.get("EXT_SOURCE_2")
                        ext2 = (
                            0.5
                            if (
                                ext2 is None
                                or (isinstance(ext2, float) and np.isnan(ext2))
                            )
                            else float(ext2)
                        )
                        if ext2 >= 0.7:
                            st.session_state["credit"] = "Excellent"
                        elif ext2 >= 0.5:
                            st.session_state["credit"] = "Bon"
                        elif ext2 >= 0.3:
                            st.session_state["credit"] = "Moyen"
                        else:
                            st.session_state["credit"] = "Faible"

                        realty = row.get("FLAG_OWN_REALTY")
                        st.session_state["prop"] = (
                            "Propriétaire" if realty == 1 else "Locataire"
                        )

                        st.session_state.prediction_result = get_client_result(pos_idx)
                        st.session_state.pred_loaded_sk_id = sk_id_int
                        st.rerun()
                    else:
                        st.error(
                            f"Aucun client avec SK_ID {sk_id_int} dans la base de données."
                        )
                        st.session_state.pred_loaded_sk_id = None
                except (ValueError, TypeError):
                    st.error(
                        "SK_ID invalide. Saisissez un identifiant numérique (ex: 100002)."
                    )

        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    # ── Formulaire (deux colonnes) ────────────────────────────────────────────
    col1, col2 = st.columns(2, gap="medium")

    with col1:
        st.markdown(
            '<div class="section-label">Informations du demandeur</div>',
            unsafe_allow_html=True,
        )
        ca, cb = st.columns(2)
        age = ca.number_input(
            "Âge",
            min_value=18,
            max_value=100,
            value=None,
            placeholder="Ex: 35",
            key="age",
        )
        years_emp = cb.number_input(
            "Années d'emploi",
            min_value=0,
            max_value=50,
            value=None,
            placeholder="Ex: 5",
            key="years",
        )

        income = st.number_input(
            "Revenu annuel (€)",
            min_value=0,
            value=None,
            placeholder="Ex: 45 000",
            step=1000,
            key="income",
        )
        loan = st.number_input(
            "Montant du prêt (€)",
            min_value=0,
            value=None,
            placeholder="Ex: 25 000",
            step=1000,
            key="loan",
        )
        debt = st.number_input(
            "Dettes existantes (€)",
            min_value=0,
            value=None,
            placeholder="Ex: 5 000",
            step=500,
            key="debt",
        )

        credit_h = st.selectbox(
            "Historique de crédit",
            ["", "Excellent", "Bon", "Moyen", "Faible"],
            key="credit",
        )
        ownership = st.selectbox(
            "Statut de propriété",
            ["", "Propriétaire", "Locataire", "Autre"],
            key="prop",
        )
        children = st.number_input(
            "Nombre d'enfants",
            min_value=0,
            max_value=20,
            value=None,
            placeholder="Ex: 2",
            key="children",
        )

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        ready = all([age, income, loan])
        if not ready:
            st.markdown(
                '<div style="font-size:12px;color:#9ca3af;margin-bottom:8px;">⚠️ Remplissez au minimum : Âge, Revenu et Montant du prêt</div>',
                unsafe_allow_html=True,
            )
        with st.container():
            st.markdown('<div class="predict-wrap">', unsafe_allow_html=True)
            if st.button(
                "Calculer la Prédiction", use_container_width=True, disabled=not ready
            ):
                ext2 = (
                    0.6
                    if credit_h in ["Excellent", "Bon"]
                    else 0.35
                    if credit_h == "Moyen"
                    else 0.2
                )
                income_v = income or 45000
                loan_v = loan or 25000
                ratio = min((loan_v / income_v) / 3, 1.0)
                emp_f = min((years_emp or 2) / 10, 1.0)
                prob = float(
                    np.clip(0.45 - 0.2 * ext2 - 0.1 * emp_f + 0.15 * ratio, 0.02, 0.97)
                )
                sc = round((1 - prob) * 1000)
                dec = 0 if prob < THRESHOLD else 1
                st.session_state.prediction_result = {
                    "probability": prob,
                    "decision": dec,
                    "score": sc,
                }
                st.session_state.pred_loaded_sk_id = None  # clear loaded client badge
            st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown(
            '<div class="section-label">Résultat de l\'analyse</div>',
            unsafe_allow_html=True,
        )
        res = st.session_state.prediction_result
        if res is None:
            st.markdown(
                """
            <div style="background:white;border:1px solid #e5e7eb;border-radius:14px;
                        text-align:center;padding:60px 24px;color:#9ca3af;
                        box-shadow:0 1px 3px rgba(0,0,0,.06);">
                <div style="font-size:52px;margin-bottom:16px;">📋</div>
                <div style="font-size:15px;font-weight:600;color:#374151;margin-bottom:8px;">En attente d'analyse</div>
                <div style="font-size:13px;">Remplissez le formulaire et lancez l'analyse pour obtenir un score de crédit</div>
            </div>""",
                unsafe_allow_html=True,
            )
        else:
            prob = res["probability"]
            sc = res["score"]
            dec = res["decision"]
            rl, re, rc = interpret_risk(prob)

            st.plotly_chart(
                make_score_gauge(sc),
                use_container_width=True,
                config={"displayModeBar": False},
            )

            if dec == 0:
                bg, bdr, icon_d = "#eff6ff", C_ACCEPTED, "✅"
                label = "CRÉDIT ACCORDÉ"
            else:
                bg, bdr, icon_d = "#fff7ed", C_REFUSED, "❌"
                label = "CRÉDIT REFUSÉ"

            st.markdown(
                f"""
            <div style="background:{bg};border:2px solid {bdr};border-radius:14px;
                        padding:18px 20px;margin-top:4px;box-shadow:0 4px 16px {bdr}22;">
                <div style="display:flex;align-items:center;gap:10px;margin-bottom:10px;">
                    <span style="font-size:26px;">{icon_d}</span>
                    <span style="font-size:17px;font-weight:800;color:{bdr};letter-spacing:-.2px;">{label}</span>
                </div>
                <div style="font-size:13px;color:#374151;line-height:1.75;">{re}</div>
            </div>""",
                unsafe_allow_html=True,
            )

            st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
            m1, m2 = st.columns(2)
            m1.metric("Score crédit", f"{sc} / 1 000")
            dist = prob - THRESHOLD
            m2.metric("Écart vs seuil", f"{'+' if dist > 0 else ''}{dist * 100:.1f}%")

        st.markdown("</div>", unsafe_allow_html=True)


# ── Historique ────────────────────────────────────────────────────────────────
def show_history():
    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
    st.markdown(
        '<div class="section-label">Demandes récentes</div>', unsafe_allow_html=True
    )
    c1, c2, c3 = st.columns([3, 2, 1], gap="small")
    search = c1.text_input(
        "",
        placeholder="🔍 Rechercher par client ou ID…",
        key="hist_search",
        label_visibility="collapsed",
    )
    filtre = c2.selectbox(
        "",
        ["Tous les statuts", "Approuvé", "Refusé"],
        key="hist_filter",
        label_visibility="collapsed",
    )
    with c3:
        st.button("📥 Exporter", use_container_width=True, key="export_btn")

    df = HISTORY_DATA.copy()
    if search:
        mask = df["Client"].str.contains(search, case=False) | df["ID"].str.contains(
            search, case=False
        )
        df = df[mask]
    if filtre != "Tous les statuts":
        df = df[df["Décision"] == filtre]

    rows_html = ""
    for _, r in df.iterrows():
        dec_color = C_ACCEPTED if r["Décision"] == "Approuvé" else C_REFUSED
        dec_bg = "#eff6ff" if r["Décision"] == "Approuvé" else "#fff7ed"
        dec_icon = "✅" if r["Décision"] == "Approuvé" else "❌"
        score_c = C_ACCEPTED if r["Score"] >= SCORE_THRESH else C_REFUSED
        rows_html += f"""
        <tr style="border-bottom:1px solid #f3f4f6;font-size:14px;">
            <td style="padding:12px 16px;color:#6b7280;">{r["ID"]}</td>
            <td style="padding:12px 16px;color:#374151;">{r["Date"]}</td>
            <td style="padding:12px 16px;font-weight:500;color:#111827;">{r["Client"]}</td>
            <td style="padding:12px 16px;color:#374151;">{r["Montant"]}</td>
            <td style="padding:12px 16px;color:#374151;">{r["Durée"]}</td>
            <td style="padding:12px 16px;">
                <span style="background:{dec_bg};color:{dec_color};border:1px solid {dec_color};
                             padding:3px 10px;border-radius:20px;font-size:12px;font-weight:600;">
                    {dec_icon} {r["Décision"]}
                </span>
            </td>
            <td style="padding:12px 16px;font-weight:700;color:{score_c};">{r["Score"]}</td>
        </tr>"""

    st.components.v1.html(
        f"""
    <div style="background:white;border:1px solid #e5e7eb;border-radius:12px;overflow:hidden;box-shadow:0 1px 3px rgba(0,0,0,.1);">
        <table style="width:100%;border-collapse:collapse;">
            <thead>
                <tr style="background:#f9fafb;border-bottom:1px solid #e5e7eb;">
                    <th style="padding:12px 16px;text-align:left;font-size:12px;color:#6b7280;font-weight:600;text-transform:uppercase;letter-spacing:.5px;">ID</th>
                    <th style="padding:12px 16px;text-align:left;font-size:12px;color:#6b7280;font-weight:600;text-transform:uppercase;letter-spacing:.5px;">DATE</th>
                    <th style="padding:12px 16px;text-align:left;font-size:12px;color:#6b7280;font-weight:600;text-transform:uppercase;letter-spacing:.5px;">CLIENT</th>
                    <th style="padding:12px 16px;text-align:left;font-size:12px;color:#6b7280;font-weight:600;text-transform:uppercase;letter-spacing:.5px;">MONTANT</th>
                    <th style="padding:12px 16px;text-align:left;font-size:12px;color:#6b7280;font-weight:600;text-transform:uppercase;letter-spacing:.5px;">DURÉE</th>
                    <th style="padding:12px 16px;text-align:left;font-size:12px;color:#6b7280;font-weight:600;text-transform:uppercase;letter-spacing:.5px;">DÉCISION</th>
                    <th style="padding:12px 16px;text-align:left;font-size:12px;color:#6b7280;font-weight:600;text-transform:uppercase;letter-spacing:.5px;">SCORE</th>
                </tr>
            </thead>
            <tbody>{rows_html}</tbody>
        </table>
        <div style="padding:12px 16px;font-size:13px;color:#6b7280;border-top:1px solid #f3f4f6;display:flex;justify-content:space-between;align-items:center;">
            <span>Affichage <strong>{len(df)}</strong> sur <strong>{len(HISTORY_DATA)}</strong> résultats</span>
            <span style="display:flex;gap:8px;">
                <span style="padding:4px 12px;border:1px solid #e5e7eb;border-radius:6px;font-size:12px;cursor:pointer;">← Précédent</span>
                <span style="padding:4px 10px;background:#2563eb;color:white;border-radius:6px;font-size:12px;">1</span>
                <span style="padding:4px 12px;border:1px solid #e5e7eb;border-radius:6px;font-size:12px;cursor:pointer;">Suivant →</span>
            </span>
        </div>
    </div>""",
        height=380,
    )


# ════════════════════════════════════════════════════════════════════════════
# DASHBOARD (routeur + sidebar)
# ════════════════════════════════════════════════════════════════════════════
def show_dashboard():
    view = st.session_state.active_view
    inject_dashboard_css(view)

    with st.sidebar:
        # ── Logo ──
        st.markdown(
            """
        <div style="padding:20px 16px 8px;border-bottom:1px solid rgba(255,255,255,.06);margin-bottom:8px;">
            <div style="display:flex;align-items:center;gap:11px;">
                <div style="width:34px;height:34px;
                            background:linear-gradient(135deg,#3b82f6,#2563eb);
                            border-radius:9px;display:flex;align-items:center;justify-content:center;
                            box-shadow:0 2px 8px rgba(37,99,235,.4);">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="white"
                         stroke-width="2.8" stroke-linecap="round" stroke-linejoin="round">
                        <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline>
                    </svg>
                </div>
                <div>
                    <div style="font-size:15px;font-weight:800;color:#f1f5f9;letter-spacing:-.3px;">CreditAI</div>
                    <div style="font-size:10px;color:#475569;font-weight:500;letter-spacing:.5px;text-transform:uppercase;">Dashboard v2.1</div>
                </div>
            </div>
        </div>""",
            unsafe_allow_html=True,
        )

        # ── Modèle actif ──
        st.markdown(
            f"""
        <div style="padding:10px 16px;margin:8px 0 4px;background:rgba(255,255,255,.04);
                    border-radius:8px;font-size:11px;color:#475569;border:1px solid rgba(255,255,255,.06);">
            <div style="color:#64748b;font-weight:600;letter-spacing:.5px;
                        text-transform:uppercase;margin-bottom:7px;font-size:10px;">Modèle actif</div>
            <div style="display:flex;flex-direction:column;gap:3px;">
                <div style="color:#94a3b8;">
                    <span style="color:#475569;">⚙</span>&nbsp;{CFG.get('model_version', 'N/A')}
                </div>
                <div style="color:#94a3b8;">
                    <span style="color:#475569;">⊘</span>&nbsp;Seuil&nbsp;:&nbsp;<strong style="color:#93c5fd;">{THRESHOLD:.4f}</strong>
                </div>
                <div style="color:#94a3b8;">
                    <span style="color:#475569;">◈</span>&nbsp;AUC&nbsp;:&nbsp;<strong style="color:#93c5fd;">{CFG.get('auc_val', 'N/A')}</strong>
                </div>
                <div style="color:#94a3b8;">
                    <span style="color:#475569;">◷</span>&nbsp;{CFG.get('date_trained', 'N/A')}
                </div>
            </div>
        </div>""",
            unsafe_allow_html=True,
        )

        st.markdown(
            '<p style="font-size:10px;color:#475569;font-weight:700;letter-spacing:1.2px;padding:12px 16px 6px;text-transform:uppercase;">Navigation</p>',
            unsafe_allow_html=True,
        )

        if st.button("Vue d'ensemble", key="btn_overview", use_container_width=True):
            st.session_state.active_view = "overview"
            st.rerun()
        if st.button("Analyse Client", key="btn_client", use_container_width=True):
            st.session_state.active_view = "client"
            st.rerun()
        if st.button(
            "Nouvelle Prédiction", key="btn_prediction", use_container_width=True
        ):
            st.session_state.active_view = "prediction"
            st.rerun()
        if st.button("Historique", key="btn_history", use_container_width=True):
            st.session_state.active_view = "history"
            st.rerun()

        st.markdown("<div style='flex:1'></div>", unsafe_allow_html=True)
        st.markdown("---")
        if st.button("Paramètres", key="btn_settings", use_container_width=True):
            pass
        if st.button("Déconnexion", key="btn_logout", use_container_width=True):
            st.session_state.page = "landing"
            st.rerun()

    # ── Header ──
    page_meta = {
        "overview": ("Vue d'ensemble", "Analyse et performance du modèle de crédit"),
        "client": (
            "Analyse Client",
            "Score, probabilité et comparaison pour chaque dossier",
        ),
        "prediction": (
            "Nouvelle Prédiction",
            "Soumettez un dossier et obtenez un score instantané",
        ),
        "history": (
            "Historique des Demandes",
            "Consultez et filtrez l'ensemble des demandes traitées",
        ),
    }
    title, subtitle = page_meta.get(view, ("Dashboard", ""))
    page_icons = {"overview": "📊", "client": "👤", "prediction": "➕", "history": "📋"}
    page_icon = page_icons.get(view, "💳")
    today = "28 fév. 2026"
    st.markdown(
        f"""
    <div class="dash-header">
        <div style="display:flex;align-items:center;gap:14px;">
            <div style="width:42px;height:42px;background:linear-gradient(135deg,#eff6ff,#dbeafe);
                        border:1px solid #bfdbfe;border-radius:12px;display:flex;
                        align-items:center;justify-content:center;font-size:20px;">{page_icon}</div>
            <div>
                <div class="dash-header-title">{title}</div>
                <div class="dash-header-subtitle">{subtitle}</div>
            </div>
        </div>
        <div style="display:flex;align-items:center;gap:16px;">
            <div style="text-align:right;">
                <div style="font-size:12px;font-weight:600;color:#374151;">Administrateur</div>
                <div style="font-size:11px;color:#9ca3af;">{today}</div>
            </div>
            <div style="width:38px;height:38px;background:linear-gradient(135deg,#3b82f6,#2563eb);
                        border-radius:50%;display:flex;align-items:center;justify-content:center;
                        color:white;font-weight:700;font-size:15px;box-shadow:0 2px 8px rgba(37,99,235,.35);">A</div>
        </div>
    </div>""",
        unsafe_allow_html=True,
    )

    # ── Routage ──
    if view == "overview":
        show_overview()
    elif view == "client":
        show_client_analysis()
    elif view == "prediction":
        show_prediction()
    elif view == "history":
        show_history()


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════
if st.session_state.page == "landing":
    show_landing()
else:
    show_dashboard()
