"""
Module : app.py
Auteur : William BLENON & Véronèse Nikina ZINSOU
Date   : 2026
Projet : Maintenance Prédictive Industrielle – M2 Data Engineering EFREI

Description :
    Dashboard décisionnel interactif développé avec Streamlit.

    Cet outil est destiné au responsable maintenance, pas au Data Scientist.
    Il ne reproduit PAS les visuels d'EDA du notebook — il s'agit d'un
    outil opérationnel permettant de :

        1. Vue d'ensemble  : KPIs temps réel, état du parc machines.
        2. Exploration     : Distribution des capteurs, corrélations.
        3. Comparaison     : Performances des 4 modèles côte à côte.
        4. Interprétabilité: Top features SHAP, importance des capteurs.
        5. Simulation      : Saisie manuelle → prédiction en temps réel.

Usage :
    $ streamlit run dashboard/app.py

Prérequis :
    - Avoir exécuté main.py pour générer les artefacts dans models/
    - Le fichier CSV doit être présent dans data/
"""

import os
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

# Ajout du répertoire racine au path pour les imports relatifs
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.data_loader import DataLoader
from src.data.data_preprocessor import DataPreprocessor
from src.models.gradient_boosting_model import GradientBoostingModel
from src.models.logistic_model import LogisticModel
from src.models.mlp_model import MLPModel
from src.models.random_forest_model import RandomForestModel

# ─────────────────────────────────────────────────────────────────
# Configuration Streamlit
# ─────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Maintenance Prédictive – Dashboard",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─────────────────────────────────────────────────────────────────
# THÈME VISUEL PREMIUM – STREAMLIT DASHBOARD
# À placer juste après st.set_page_config(...)
# ─────────────────────────────────────────────────────────────────

# Palette globale
COLORS = {
    "bg": "#071018",
    "surface": "#0f172a",
    "surface_light": "#162033",

    "primary": "#14b8a6",
    "secondary": "#38bdf8",

    "success": "#34d399",
    "warning": "#fbbf24",
    "danger": "#fb7185",

    "text": "#e2e8f0",
    "muted": "#94a3b8",

    "border": "rgba(255,255,255,0.06)",
}

# Template Plotly global
PLOTLY_TEMPLATE = "plotly_dark"

# CSS complet
st.markdown(f"""
<style>

/* =========================================================
   IMPORT FONTS
========================================================= */

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

/* =========================================================
   GLOBAL
========================================================= */

html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
    background-color: {COLORS["bg"]};
    color: {COLORS["text"]};
}}

.stApp {{
    background:
        radial-gradient(circle at top left,
            rgba(20,184,166,0.08),
            transparent 25%),

        radial-gradient(circle at bottom right,
            rgba(56,189,248,0.08),
            transparent 25%),

        {COLORS["bg"]};
}}

/* =========================================================
   ANIMATIONS
========================================================= */

@keyframes fadeIn {{
    from {{
        opacity: 0;
        transform: translateY(8px);
    }}

    to {{
        opacity: 1;
        transform: translateY(0px);
    }}
}}

.fade-in {{
    animation: fadeIn 0.4s ease-in-out;
}}

/* =========================================================
   SIDEBAR
========================================================= */

[data-testid="stSidebar"] {{
    background:
        linear-gradient(
            180deg,
            rgba(13,23,33,0.96),
            rgba(7,16,24,0.98)
        );

    border-right: 1px solid {COLORS["border"]};

    backdrop-filter: blur(12px);
}}

[data-testid="stSidebar"] * {{
    color: {COLORS["text"]};
}}

/* =========================================================
   HEADER
========================================================= */

.main-header {{

    background:
        linear-gradient(
            135deg,
            rgba(15,23,42,0.95),
            rgba(17,34,51,0.90)
        );

    border: 1px solid {COLORS["border"]};

    backdrop-filter: blur(14px);

    padding: 2.3rem;

    border-radius: 24px;

    margin-bottom: 2rem;

    animation: fadeIn 0.4s ease-in-out;

    box-shadow:
        0 10px 30px rgba(0,0,0,0.35),
        0 0 0 1px rgba(255,255,255,0.03);
}}

.main-header h1 {{

    color: #f8fafc;

    font-size: 2.4rem;

    font-weight: 700;

    margin-bottom: 0.5rem;

    font-family: 'JetBrains Mono', monospace;

    letter-spacing: -1px;
}}

.main-header p {{
    color: {COLORS["muted"]};
    font-size: 0.95rem;
    margin: 0;
}}

/* =========================================================
   SECTION TITLES
========================================================= */

.section-title {{

    font-size: 1.1rem;

    font-weight: 600;

    color: #f8fafc;

    margin-bottom: 1rem;

    padding-bottom: 0.55rem;

    border-bottom: 1px solid rgba(255,255,255,0.08);

    font-family: 'JetBrains Mono', monospace;
}}

/* =========================================================
   KPI CARDS
========================================================= */

.kpi-card {{

    background: rgba(15, 23, 42, 0.72);

    border: 1px solid {COLORS["border"]};

    backdrop-filter: blur(12px);

    border-radius: 18px;

    padding: 1.4rem 1rem;

    text-align: center;

    transition:
        transform 0.25s ease,
        border-color 0.25s ease,
        box-shadow 0.25s ease;

    animation: fadeIn 0.4s ease-in-out;

    box-shadow:
        0 8px 24px rgba(0,0,0,0.28);
}}

.kpi-card:hover {{

    transform: translateY(-4px);

    border-color: rgba(20,184,166,0.45);

    box-shadow:
        0 12px 30px rgba(0,0,0,0.38),
        0 0 18px rgba(20,184,166,0.12);
}}

.kpi-value {{

    font-size: 2.3rem;

    font-weight: 700;

    font-family: 'JetBrains Mono', monospace;

    color: {COLORS["primary"]};
}}

.kpi-label {{

    color: {COLORS["muted"]};

    font-size: 0.74rem;

    text-transform: uppercase;

    letter-spacing: 1.5px;

    margin-top: 0.45rem;
}}

.kpi-alert {{
    color: {COLORS["danger"]} !important;
}}

.kpi-warning {{
    color: {COLORS["warning"]} !important;
}}

.kpi-ok {{
    color: {COLORS["success"]} !important;
}}

/* =========================================================
   BADGES
========================================================= */

.badge-danger {{

    background: rgba(251,113,133,0.12);

    color: {COLORS["danger"]};

    border: 1px solid rgba(251,113,133,0.3);

    padding: 0.25rem 0.7rem;

    border-radius: 999px;

    font-size: 0.72rem;

    font-weight: 600;
}}

.badge-ok {{

    background: rgba(52,211,153,0.12);

    color: {COLORS["success"]};

    border: 1px solid rgba(52,211,153,0.3);

    padding: 0.25rem 0.7rem;

    border-radius: 999px;

    font-size: 0.72rem;

    font-weight: 600;
}}

/* =========================================================
   PREDICTION BOX
========================================================= */

.prediction-box {{

    border-radius: 22px;

    padding: 2rem;

    text-align: center;

    margin-top: 1rem;

    backdrop-filter: blur(12px);

    border: 1px solid rgba(255,255,255,0.08);

    animation: fadeIn 0.4s ease-in-out;

    box-shadow:
        0 10px 30px rgba(0,0,0,0.35);
}}

.prediction-danger {{
    background: rgba(190, 24, 93, 0.12);
    border-color: rgba(244, 63, 94, 0.35);
}}

.prediction-safe {{
    background: rgba(5, 150, 105, 0.12);
    border-color: rgba(16, 185, 129, 0.35);
}}

.prediction-title {{

    font-size: 1.5rem;

    font-weight: 700;

    color: #f8fafc;

    margin-bottom: 0.8rem;

    font-family: 'JetBrains Mono', monospace;
}}

.prediction-proba {{

    font-size: 3.5rem;

    font-weight: 700;

    margin: 1rem 0;

    font-family: 'JetBrains Mono', monospace;
}}

/* =========================================================
   STREAMLIT METRICS
========================================================= */

[data-testid="stMetric"] {{

    background: rgba(15,23,42,0.72);

    border: 1px solid {COLORS["border"]};

    border-radius: 18px;

    padding: 1rem;

    box-shadow:
        0 8px 24px rgba(0,0,0,0.25);

    transition: 0.25s ease;

    animation: fadeIn 0.4s ease-in-out;
}}

[data-testid="stMetric"]:hover {{
    transform: translateY(-3px);
}}

/* =========================================================
   BUTTONS
========================================================= */

.stButton > button {{

    background:
        linear-gradient(
            135deg,
            {COLORS["secondary"]},
            {COLORS["primary"]}
        );

    color: white;

    border: none;

    border-radius: 14px;

    padding: 0.65rem 1.25rem;

    font-weight: 600;

    transition: all 0.25s ease;

    box-shadow:
        0 8px 18px rgba(20,184,166,0.25);
}}

.stButton > button:hover {{

    transform: translateY(-2px);

    box-shadow:
        0 10px 24px rgba(20,184,166,0.35);

    filter: brightness(1.05);
}}

/* =========================================================
   SELECTBOX / INPUTS / SLIDERS
========================================================= */

.stSelectbox div[data-baseweb="select"] > div,
.stSlider,
.stTextInput > div > div > input,
.stNumberInput input {{

    background: rgba(15,23,42,0.72) !important;

    border: 1px solid rgba(255,255,255,0.05) !important;

    border-radius: 12px !important;

    color: {COLORS["text"]} !important;
}}

/* =========================================================
   DATAFRAME
========================================================= */

[data-testid="stDataFrame"] {{

    border-radius: 18px;

    overflow: hidden;

    border: 1px solid rgba(255,255,255,0.04);
}}

/* =========================================================
   ALERTS
========================================================= */

.stAlert {{
    border-radius: 16px;
}}

/* =========================================================
   TABS
========================================================= */

.stTabs [data-baseweb="tab-list"] {{
    gap: 10px;
}}

.stTabs [data-baseweb="tab"] {{

    background: rgba(15,23,42,0.72);

    border-radius: 12px;

    padding: 0.5rem 1rem;
}}

/* =========================================================
   SCROLLBAR
========================================================= */

::-webkit-scrollbar {{
    width: 10px;
}}

::-webkit-scrollbar-track {{
    background: {COLORS["bg"]};
}}

::-webkit-scrollbar-thumb {{
    background: #1e293b;
    border-radius: 10px;
}}

::-webkit-scrollbar-thumb:hover {{
    background: #334155;
}}

/* =========================================================
   PLOTLY CONTAINER
========================================================= */

.js-plotly-plot {{

    border-radius: 18px;

    overflow: hidden;

    background: rgba(15,23,42,0.45);

    border: 1px solid rgba(255,255,255,0.04);

    padding: 0.5rem;
}}

</style>
""", unsafe_allow_html=True)
# ─────────────────────────────────────────────────────────────────
# Styles CSS personnalisés
# ─────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    /* Police et thème général */
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;600;700&family=IBM+Plex+Mono:wght@400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
    }

    /* En-tête principal */
    .main-header {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        padding: 2rem 2.5rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        border-left: 5px solid #00d4aa;
    }
    .main-header h1 {
        color: #ffffff;
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
        font-family: 'IBM Plex Mono', monospace;
        letter-spacing: -0.5px;
    }
    .main-header p {
        color: #a8c8d8;
        margin: 0.5rem 0 0 0;
        font-size: 0.95rem;
    }

    /* Cartes KPI */
    .kpi-card {
        background: #1a2332;
        border: 1px solid #2d3f55;
        border-radius: 10px;
        padding: 1.2rem 1.5rem;
        text-align: center;
        transition: border-color 0.2s;
    }
    .kpi-card:hover { border-color: #00d4aa; }
    .kpi-value {
        font-size: 2.2rem;
        font-weight: 700;
        font-family: 'IBM Plex Mono', monospace;
        color: #00d4aa;
    }
    .kpi-label {
        font-size: 0.8rem;
        color: #7a9ab5;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 0.3rem;
    }
    .kpi-alert { color: #ff6b6b !important; }
    .kpi-warning { color: #ffd93d !important; }
    .kpi-ok { color: #6bcb77 !important; }

    /* Badges de statut */
    .badge-danger {
        background: rgba(255, 107, 107, 0.15);
        color: #ffff;
        border: 1px solid #ff6b6b;
        padding: 0.2rem 0.7rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .badge-ok {
        background: rgba(107, 203, 119, 0.15);
        color: #6bcb77;
        border: 1px solid #6bcb77;
        padding: 0.2rem 0.7rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
    }

    /* Section titles */
    .section-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #e0eaf4;
        border-bottom: 2px solid #00d4aa;
        padding-bottom: 0.4rem;
        margin-bottom: 1.2rem;
        font-family: 'IBM Plex Mono', monospace;
    }

    /* Prédiction résultat */
    .prediction-box {
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }
    .prediction-danger {
        background: rgba(255, 107, 107, 0.1);
        border: 2px solid #ff6b6b;
    }
    .prediction-safe {
        background: rgba(107, 203, 119, 0.1);
        border: 2px solid #6bcb77;
    }
    .prediction-title {
        font-size: 1.5rem;
        font-weight: 700;
        font-family: 'IBM Plex Mono', monospace;
    }
    .prediction-proba {
        font-size: 2.8rem;
        font-weight: 700;
        font-family: 'IBM Plex Mono', monospace;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #0d1b2a;
        border-right: 1px solid #1e3048;
    }

    /* Metrics Streamlit natifs */
    [data-testid="stMetric"] {
        background: #1a2332;
        border: 1px solid #2d3f55;
        border-radius: 8px;
        padding: 0.8rem 1rem;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# Chargement des données et modèles (mis en cache)
# ─────────────────────────────────────────────────────────────────

def apply_dark_theme(fig):
    """
    Applique le thème graphique premium à toutes les figures Plotly.
    """

    fig.update_layout(
        template=PLOTLY_TEMPLATE,

        paper_bgcolor="rgba(15,23,42,0.65)",
        plot_bgcolor="rgba(15,23,42,0.35)",

        font=dict(
            family="Inter",
            color=COLORS["text"],
            size=13,
        ),

        title_font=dict(
            family="JetBrains Mono",
            size=18,
            color="#f8fafc",
        ),

        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(color=COLORS["muted"]),
        ),

        margin=dict(
            t=40,
            b=40,
            l=20,
            r=20,
        ),
    )

    fig.update_xaxes(
        gridcolor="rgba(255,255,255,0.05)",
        zerolinecolor="rgba(255,255,255,0.05)",
    )

    fig.update_yaxes(
        gridcolor="rgba(255,255,255,0.05)",
        zerolinecolor="rgba(255,255,255,0.05)",
    )

    return fig

@st.cache_data
def load_data() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]:
    """
    Charge et prépare les données depuis le CSV.
    Mis en cache par Streamlit pour éviter les rechargements inutiles.

    Returns:
        tuple: (df_raw, X_train, X_test, y_train, y_test, feature_names)
    """
    data_path = os.path.join(os.path.dirname(__file__), "..", "data",
                             "industrial_machine_maintenance.csv")
    loader = DataLoader(data_path)
    df = loader.load()

    preprocessor = DataPreprocessor(target="failure_within_24h", test_size=0.2, random_state=42)
    X_train, X_test, y_train, y_test = preprocessor.fit_transform(df)
    feature_names = preprocessor.feature_names

    return df, X_train, X_test, y_train, y_test, feature_names


@st.cache_resource
def load_and_train_models(X_train, y_train):
    """
    Entraîne les 4 modèles ou les charge depuis le disque si disponibles.
    Mis en cache pour éviter les réentraînements à chaque interaction.

    Returns:
        list[BaseModel]: Les 4 modèles entraînés.
    """
    models = [
        LogisticModel(),
        RandomForestModel(),
        GradientBoostingModel(),
        MLPModel(),
    ]
    for model in models:
        model_path = os.path.join(
            os.path.dirname(__file__), "..", "models",
            f"{model.name.replace(' ', '_').replace('(', '').replace(')', '').lower()}.pkl"
        )
        if os.path.exists(model_path):
            model.load(model_path)
        else:
            model.train(X_train, y_train)

    return models


# ─────────────────────────────────────────────────────────────────
# En-tête principal
# ─────────────────────────────────────────────────────────────────

st.markdown("""
<div class="main-header">
    <h1>⚙️ PREDICTIVE MAINTENANCE SYSTEM</h1>
    <p>Système Intelligent de Détection de Pannes Industrielles · Classification Binaire · failure_within_24h</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# Sidebar – Navigation
# ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚙️ Navigation")
    page = st.radio(
        "Navigation",
        label_visibility="collapsed",
        options=[
            "📊 Vue d'ensemble",
            "🔍 Exploration des données",
            "🤖 Comparaison des modèles",
            "🧠 Interprétabilité",
            "🎯 Simulation & Prédiction",
        ],
    )

    st.markdown("---")
    st.markdown("### ⚙️ Paramètres")
    selected_model_name = st.selectbox(
        "Modèle actif",
        ["Logistic Regression", "Random Forest",
         "Gradient Boosting (XGBoost)", "MLP (Deep Learning)"],
        index=1,
    )
    threshold = st.slider(
        "Seuil de décision",
        min_value=0.1, max_value=0.9, value=0.5, step=0.05,
        help="Ajustez le seuil pour équilibrer Precision et Recall. "
             "Un seuil bas détecte plus de pannes (moins de faux négatifs)."
    )

    st.markdown("---")
    st.markdown(
        "<small style='color:#5a7a95;'>EFREI M2 Data Engineering · 2025<br>"
        "Maintenance Prédictive Industrielle</small>",
        unsafe_allow_html=True
    )

# ─────────────────────────────────────────────────────────────────
# Chargement des ressources
# ─────────────────────────────────────────────────────────────────

with st.spinner("Chargement des données et des modèles..."):
    try:
        df, X_train, X_test, y_train, y_test, feature_names = load_data()
        models = load_and_train_models(X_train, y_train)
        active_model = next(m for m in models if m.name == selected_model_name)
        DATA_LOADED = True
    except Exception as e:
        DATA_LOADED = False
        st.error(
            f"❌ Impossible de charger les données.\n\n"
            f"Vérifiez que le fichier CSV est bien présent dans `data/`.\n\n"
            f"Erreur : `{e}`"
        )
        st.stop()


# ─────────────────────────────────────────────────────────────────
# PAGE 1 – Vue d'ensemble
# ─────────────────────────────────────────────────────────────────

if page == "📊 Vue d'ensemble":

    st.markdown('<p class="section-title">KPIs du Parc Machine</p>', unsafe_allow_html=True)

    # Calcul des KPIs
    n_total = len(df)
    n_panne = int(df["failure_within_24h"].sum())
    pct_panne = round(n_panne / n_total * 100, 1)

    y_pred = (active_model.predict_proba(X_test)[:, 1] >= threshold).astype(int)
    recall = round(recall_score(y_test, y_pred, zero_division=0) * 100, 1)
    f1 = round(f1_score(y_test, y_pred, zero_division=0) * 100, 1)
    auc = round(roc_auc_score(y_test, active_model.predict_proba(X_test)[:, 1]) * 100, 1)

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{n_total:,}</div>
            <div class="kpi-label">Enregistrements</div>
        </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value kpi-alert">{n_panne:,}</div>
            <div class="kpi-label">Pannes détectées</div>
        </div>""", unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value kpi-warning">{pct_panne}%</div>
            <div class="kpi-label">Taux de panne</div>
        </div>""", unsafe_allow_html=True)

    with col4:
        color = "kpi-ok" if recall >= 70 else "kpi-alert"
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value {color}">{recall}%</div>
            <div class="kpi-label">Recall (seuil={threshold})</div>
        </div>""", unsafe_allow_html=True)

    with col5:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value kpi-ok">{auc}%</div>
            <div class="kpi-label">ROC-AUC</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown('<p class="section-title">Distribution des pannes</p>', unsafe_allow_html=True)
        dist_data = df["failure_within_24h"].value_counts().reset_index()
        dist_data.columns = ["Classe", "Effectif"]
        dist_data["Label"] = dist_data["Classe"].map({0: "Pas de panne", 1: "Panne < 24h"})

        fig_pie = px.pie(
            dist_data,
            values="Effectif",
            names="Label",
            color_discrete_sequence=["#00d4aa", "#ff6b6b"],
            hole=0.5,
        )
        fig_pie.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e0eaf4"),
            legend=dict(font=dict(size=12)),
            margin=dict(t=20, b=20),
        )
        st.plotly_chart(fig_pie, width='stretch')

    with col_right:
        st.markdown('<p class="section-title">Probabilités de panne – Distribution</p>',
                    unsafe_allow_html=True)
        probas = active_model.predict_proba(X_test)[:, 1]

        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=probas[y_test == 0],
            name="Pas de panne (réel)",
            marker_color="#00d4aa",
            opacity=0.7,
            nbinsx=40,
        ))
        fig_hist.add_trace(go.Histogram(
            x=probas[y_test == 1],
            name="Panne (réel)",
            marker_color="#ff6b6b",
            opacity=0.7,
            nbinsx=40,
        ))
        fig_hist.add_vline(
            x=threshold,
            line_dash="dash",
            line_color="#ffd93d",
            annotation_text=f"Seuil = {threshold}",
            annotation_font_color="#ffd93d",
        )
        fig_hist.update_layout(
            barmode="overlay",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e0eaf4"),
            xaxis_title="Probabilité de panne prédite",
            yaxis_title="Nombre d'observations",
            legend=dict(font=dict(size=11)),
            margin=dict(t=20, b=40),
        )
        st.plotly_chart(fig_hist, width='stretch')

    # Tableau synthèse modèle actif
    st.markdown('<p class="section-title">Synthèse du modèle actif</p>', unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    from sklearn.metrics import accuracy_score, precision_score
    acc = round(accuracy_score(y_test, y_pred) * 100, 1)
    prec = round(precision_score(y_test, y_pred, zero_division=0) * 100, 1)

    m1.metric("Accuracy", f"{acc}%")
    m2.metric("Precision", f"{prec}%")
    m3.metric("Recall", f"{recall}%", help="Pannes réelles correctement détectées")
    m4.metric("F1-Score", f"{f1}%")


# ─────────────────────────────────────────────────────────────────
# PAGE 2 – Exploration des données
# ─────────────────────────────────────────────────────────────────

elif page == "🔍 Exploration des données":

    st.markdown('<p class="section-title">Distribution des variables capteurs</p>',
                unsafe_allow_html=True)

    numeric_cols = ["vibration_rms", "temperature_motor", "current_phase_avg",
                    "pressure_level", "rpm", "hours_since_maintenance", "ambient_temp", "rul_hours"]
    numeric_cols = [c for c in numeric_cols if c in df.columns]

    selected_feature = st.selectbox("Sélectionner un capteur", numeric_cols)

    col1, col2 = st.columns([2, 1])

    with col1:
        fig_box = go.Figure()
        for label, color, name in [(0, "#00d4aa", "Pas de panne"), (1, "#ff6b6b", "Panne < 24h")]:
            fig_box.add_trace(go.Box(
                y=df[df["failure_within_24h"] == label][selected_feature],
                name=name,
                marker_color=color,
                boxmean="sd",
            ))
        fig_box.update_layout(
            title=f"Distribution de '{selected_feature}' selon la classe de panne",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e0eaf4"),
            yaxis_title=selected_feature,
            margin=dict(t=50, b=30),
        )
        st.plotly_chart(fig_box, width='stretch')

    with col2:
        st.markdown("**Statistiques descriptives**")
        stats = df.groupby("failure_within_24h")[selected_feature].agg(
            ["mean", "std", "min", "max"]
        ).round(2)
        stats.index = ["Pas de panne", "Panne"]
        st.dataframe(stats, width='stretch')

    st.markdown("---")
    st.markdown('<p class="section-title">Matrice de corrélation</p>', unsafe_allow_html=True)

    corr_cols = ["vibration_rms", "temperature_motor", "pressure_level",
                  "rpm", "rul_hours", "failure_within_24h"]
    corr_cols = [c for c in corr_cols if c in df.columns]
    corr_matrix = df[corr_cols].corr()

    fig_corr = px.imshow(
        corr_matrix,
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        text_auto=".2f",
        aspect="auto",
    )
    fig_corr.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e0eaf4"),
        margin=dict(t=20, b=20),
    )
    st.plotly_chart(fig_corr, width='stretch')

    st.markdown("---")
    st.markdown('<p class="section-title">Scatter plot – Deux capteurs</p>',
                unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        x_feat = st.selectbox("Axe X", numeric_cols, index=0)
    with c2:
        y_feat = st.selectbox("Axe Y", numeric_cols, index=1)

    df_sample = df.sample(min(3000, len(df)), random_state=42)
    fig_scatter = px.scatter(
        df_sample,
        x=x_feat,
        y=y_feat,
        color=df_sample["failure_within_24h"].map({0: "Pas de panne", 1: "Panne < 24h"}),
        color_discrete_map={"Pas de panne": "#00d4aa", "Panne < 24h": "#ff6b6b"},
        opacity=0.5,
        title=f"{x_feat} vs {y_feat}",
    )
    fig_scatter.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e0eaf4"),
        legend_title="Classe",
        margin=dict(t=50, b=30),
    )
    st.plotly_chart(fig_scatter, width='stretch')


# ─────────────────────────────────────────────────────────────────
# PAGE 3 – Comparaison des modèles
# ─────────────────────────────────────────────────────────────────

elif page == "🤖 Comparaison des modèles":

    st.markdown('<p class="section-title">Tableau comparatif des performances</p>',
                unsafe_allow_html=True)

    # Construction du tableau de métriques pour tous les modèles
    rows = []
    for model in models:
        y_pred_m = (model.predict_proba(X_test)[:, 1] >= threshold).astype(int)
        y_proba_m = model.predict_proba(X_test)[:, 1]
        from sklearn.metrics import accuracy_score
        rows.append({
            "Modèle": model.name,
            "Accuracy": round(accuracy_score(y_test, y_pred_m) * 100, 2),
            "Precision": round(precision_score(y_test, y_pred_m, zero_division=0) * 100, 2),
            "Recall": round(recall_score(y_test, y_pred_m, zero_division=0) * 100, 2),
            "F1-Score": round(f1_score(y_test, y_pred_m, zero_division=0) * 100, 2),
            "ROC-AUC": round(roc_auc_score(y_test, y_proba_m) * 100, 2),
        })

    df_results = pd.DataFrame(rows).sort_values("F1-Score", ascending=False)

    # Mise en forme du tableau
    st.dataframe(
        df_results.style
        .highlight_max(subset=["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"],
                       color="#1a3d2e")
        .format("{:.2f}%", subset=["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]),
        width='stretch',
        hide_index=True,
    )

    st.markdown("---")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown('<p class="section-title">Courbes ROC</p>', unsafe_allow_html=True)
        fig_roc = go.Figure()
        colors = ["#00d4aa", "#ff6b6b", "#ffd93d", "#a78bfa"]

        for model, color in zip(models, colors):
            y_proba_m = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_proba_m)
            auc_val = round(roc_auc_score(y_test, y_proba_m), 3)
            fig_roc.add_trace(go.Scatter(
                x=fpr, y=tpr,
                mode="lines",
                name=f"{model.name} (AUC={auc_val})",
                line=dict(color=color, width=2),
            ))

        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode="lines",
            name="Aléatoire",
            line=dict(color="#5a7a95", dash="dash"),
        ))
        fig_roc.update_layout(
            xaxis_title="Taux de Faux Positifs",
            yaxis_title="Recall (TPR)",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e0eaf4"),
            legend=dict(font=dict(size=10)),
            margin=dict(t=20, b=40),
        )
        st.plotly_chart(fig_roc, width='stretch')

    with col2:
        st.markdown('<p class="section-title">Matrice de confusion</p>',
                    unsafe_allow_html=True)

        cm_model_name = st.selectbox(
            "Modèle",
            [m.name for m in models],
            key="cm_select",
        )
        cm_model = next(m for m in models if m.name == cm_model_name)
        y_pred_cm = (cm_model.predict_proba(X_test)[:, 1] >= threshold).astype(int)
        cm = confusion_matrix(y_test, y_pred_cm)

        fig_cm = px.imshow(
            cm,
            text_auto=True,
            color_continuous_scale="Blues",
            x=["Prédit : Pas de panne", "Prédit : Panne"],
            y=["Réel : Pas de panne", "Réel : Panne"],
            aspect="auto",
        )
        fig_cm.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e0eaf4"),
            margin=dict(t=20, b=40),
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_cm, width='stretch')

        # Analyse rapide de la matrice
        tn, fp, fn, tp = cm.ravel()
        st.markdown(
            f"**Analyse :** {fn} fausses alarmes manquées (FN) · "
            f"{fp} fausses alertes (FP) · **Recall = {round(tp/(tp+fn)*100, 1)}%**"
        )

    # Radar chart comparatif
    st.markdown("---")
    st.markdown('<p class="section-title">Radar – Profil des modèles</p>',
                unsafe_allow_html=True)

    categories = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    fig_radar = go.Figure()

    for idx, color in enumerate(colors[:len(df_results)]):
        row = df_results.iloc[idx]
        values = [row[c] for c in categories]
        values += [values[0]]  # fermeture du polygone

        fig_radar.add_trace(go.Scatterpolar(
            r=values,
            theta=categories + [categories[0]],
            fill="toself",
            name=row['Modèle'],
            line=dict(color=color),
            opacity=0.6,
        ))

    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100]),
            bgcolor="rgba(0,0,0,0)",
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e0eaf4"),
        legend=dict(font=dict(size=10)),
        margin=dict(t=30, b=30),
    )
    st.plotly_chart(fig_radar, width='stretch')


# ─────────────────────────────────────────────────────────────────
# PAGE 4 – Interprétabilité
# ─────────────────────────────────────────────────────────────────

elif page == "🧠 Interprétabilité":

    st.markdown('<p class="section-title">Importance des variables capteurs</p>',
                unsafe_allow_html=True)

    st.info(
        "💡 **Lecture :** Plus l'importance est élevée, plus le capteur influence "
        "la décision du modèle. Cette information permet au responsable maintenance "
        "d'orienter les efforts de surveillance sur les variables critiques.",
        icon="💡"
    )

    interp_model_name = st.selectbox(
        "Modèle à analyser",
        [m.name for m in models],
        key="interp_model",
    )
    interp_model = next(m for m in models if m.name == interp_model_name)

    # Feature importance native (arbres)
    if hasattr(interp_model.model, "feature_importances_"):
        st.markdown('<p class="section-title">Feature Importance Native (Gini)</p>',
                    unsafe_allow_html=True)

        importances = interp_model.model.feature_importances_
        df_imp = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances,
        }).sort_values("Importance", ascending=True).tail(15)

        fig_imp = go.Figure(go.Bar(
            x=df_imp["Importance"],
            y=df_imp["Feature"],
            orientation="h",
            marker=dict(
                color=df_imp["Importance"],
                colorscale="Teal",
                showscale=False,
            ),
        ))
        fig_imp.update_layout(
            xaxis_title="Importance (réduction d'impureté Gini)",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e0eaf4"),
            margin=dict(t=20, b=40, l=180),
            height=450,
        )
        st.plotly_chart(fig_imp, width='stretch')

    else:
        st.warning(
            f"⚠️ Feature Importance native non disponible pour `{interp_model_name}`. "
            f"Utilisez la Permutation Importance ci-dessous."
        )

    st.markdown("---")
    st.markdown(
        '<p class="section-title">Permutation Importance (tous modèles)</p>',
        unsafe_allow_html=True
    )
    st.caption(
        "La permutation importance mesure la chute de performance "
        "quand les valeurs d'une feature sont mélangées aléatoirement. "
        "Elle est plus fiable que l'importance Gini pour les variables à haute cardinalité."
    )

    if st.button("🔄 Calculer la Permutation Importance", key="perm_btn"):
        with st.spinner("Calcul en cours (peut prendre quelques secondes)..."):
            from sklearn.inspection import permutation_importance as perm_imp
            result = perm_imp(
                interp_model.model,
                X_test, y_test,
                n_repeats=5,
                scoring="f1",
                random_state=42,
                n_jobs=-1,
            )
            df_perm = pd.DataFrame({
                "Feature": feature_names,
                "Importance": result.importances_mean,
                "Std": result.importances_std,
            }).sort_values("Importance", ascending=True).tail(15)

            fig_perm = go.Figure(go.Bar(
                x=df_perm["Importance"],
                y=df_perm["Feature"],
                orientation="h",
                error_x=dict(type="data", array=df_perm["Std"], visible=True),
                marker_color="#ffd93d",
            ))
            fig_perm.update_layout(
                xaxis_title="Chute de F1-Score lors de la permutation",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e0eaf4"),
                margin=dict(t=20, b=40, l=180),
                height=450,
            )
            st.plotly_chart(fig_perm, width='stretch')


# ─────────────────────────────────────────────────────────────────
# PAGE 5 – Simulation & Prédiction
# ─────────────────────────────────────────────────────────────────

elif page == "🎯 Simulation & Prédiction":

    st.markdown('<p class="section-title">Simuler un scénario machine</p>',
                unsafe_allow_html=True)
    st.markdown(
        "Saisissez les valeurs des capteurs pour obtenir une prédiction "
        "en temps réel. Modèle actif : **" + selected_model_name + "**"
    )

    # Récupération des plages de valeurs depuis les données réelles
    sim_cols = ["vibration_rms", "temperature_motor", "rpm", "pressure_level"]
    sim_cols = [c for c in sim_cols if c in df.columns]
    stats = df[sim_cols].describe()

    col1, col2 = st.columns(2)

    with col1:
        vibration = st.slider(
            "🔧 Vibration RMS",
            min_value=float(stats.loc["min", "vibration_rms"]),
            max_value=float(stats.loc["max", "vibration_rms"]),
            value=float(stats.loc["mean", "vibration_rms"]),
            step=0.01,
            help="Amplitude de vibration mesurée par l'accéléromètre (g)"
        )
        temperature = st.slider(
            "🌡️ Température moteur (°C)",
            min_value=float(stats.loc["min", "temperature_motor"]),
            max_value=float(stats.loc["max", "temperature_motor"]),
            value=float(stats.loc["mean", "temperature_motor"]),
            step=0.5,
        )

    with col2:
        rpm = st.slider(
            "⚙️ Vitesse de rotation (RPM)",
            min_value=float(stats.loc["min", "rpm"]),
            max_value=float(stats.loc["max", "rpm"]),
            value=float(stats.loc["mean", "rpm"]),
            step=10.0,
        )
        pressure = st.slider(
            "🔩 Pression (bar)",
            min_value=float(stats.loc["min", "pressure_level"]),
            max_value=float(stats.loc["max", "pressure_level"]),
            value=float(stats.loc["mean", "pressure_level"]),
            step=0.1,
        )

    operating_mode = st.selectbox(
        "🏭 Mode de fonctionnement",
        options=df["operating_mode"].unique().tolist() if "operating_mode" in df.columns else ["normal", "degraded", "maintenance"],
    )

    st.markdown("---")

    if st.button("🚀 Lancer la prédiction", type="primary"):

        # Construction du DataFrame de saisie
        input_data = {
            "vibration_rms": [vibration],
            "temperature_motor": [temperature],
            "rpm": [rpm],
            "pressure_level": [pressure],
            "operating_mode": [operating_mode],
        }

        # Ajout des colonnes manquantes avec leurs valeurs moyennes
        for col in df.columns:
            if col not in input_data and col not in ["failure_within_24h", "failure_type",
                                                      "rul_hours", "machine_id", "timestamp"]:
                if df[col].dtype in ["int64", "float64"]:
                    input_data[col] = [df[col].mean()]
                else:
                    input_data[col] = [df[col].mode()[0]]

        input_df = pd.DataFrame(input_data)

        try:
            # Preprocessing via le pipeline déjà ajusté
            data_path = os.path.join(os.path.dirname(__file__), "..", "data",
                                     "industrial_machine_maintenance.csv")
            loader_tmp = DataLoader(data_path)
            df_tmp = loader_tmp.load()
            prep_tmp = DataPreprocessor(target="failure_within_24h")
            prep_tmp.fit_transform(df_tmp)
            X_input = prep_tmp.transform(input_df)

            proba = active_model.predict_proba(X_input)[0][1]
            is_failure = proba >= threshold

            # Affichage du résultat
            box_class = "prediction-danger" if is_failure else "prediction-safe"
            icon = "🔴" if is_failure else "🟢"
            verdict = "RISQUE DE PANNE ÉLEVÉ" if is_failure else "MACHINE OPÉRATIONNELLE"
            proba_color = "#ff6b6b" if is_failure else "#6bcb77"

            st.markdown(f"""
            <div class="prediction-box {box_class}">
                <div class="prediction-title">{icon} {verdict}</div>
                <div class="prediction-proba" style="color:{proba_color}">
                    {proba:.1%}
                </div>
                <div style="color:#a8c8d8; font-size:0.85rem;">
                    Probabilité de panne dans les 24h · Seuil = {threshold}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Jauge de risque
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=proba * 100,
                number={"suffix": "%", "font": {"size": 36, "color": "#e0eaf4"}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#5a7a95"},
                    "bar": {"color": "#ff6b6b" if is_failure else "#00d4aa"},
                    "bgcolor": "#1a2332",
                    "bordercolor": "#2d3f55",
                    "steps": [
                        {"range": [0, 30], "color": "#1a3d2e"},
                        {"range": [30, 60], "color": "#3d3a1a"},
                        {"range": [60, 100], "color": "#3d1a1a"},
                    ],
                    "threshold": {
                        "line": {"color": "#ffd93d", "width": 3},
                        "thickness": 0.8,
                        "value": threshold * 100,
                    },
                },
                title={"text": "Score de risque", "font": {"color": "#a8c8d8"}},
            ))
            fig_gauge.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e0eaf4"),
                height=280,
                margin=dict(t=40, b=20),
            )
            st.plotly_chart(fig_gauge, width='stretch')

            if is_failure:
                st.error(
                    "⚠️ **Action recommandée :** Planifier une inspection immédiate. "
                    "Vérifiez en priorité les niveaux de vibration et la température moteur."
                )
            else:
                st.success(
                    "✅ **Machine opérationnelle.** Aucune intervention immédiate requise. "
                    "Prochaine vérification planifiée selon le calendrier standard."
                )

        except Exception as e:
            st.error(f"Erreur lors de la prédiction : `{e}`")