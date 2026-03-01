import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
import os
from pathlib import Path

# 🔧 CONFIGURATION — À MODIFIER AVEC TA VRAIE URL
API_URL = "https://projet7-mlops-implementation-modele.onrender.com/predict"

# 🎨 Style global
st.set_page_config(
    page_title="Simulateur Crédit - Projet 7", page_icon="🏦", layout="centered"
)

# 🏦 Titre principal
st.markdown(
    """
    <div style="text-align: center; padding: 10px; background-color: #f0f2f6; border-radius: 10px; margin-bottom: 20px;">
        <h1 style="color: #1e3a8a; font-size: 2.2em;">🏦 Simulateur de Décision de Crédit</h1>
        <p style="color: #4b5563; font-size: 1.1em;">
            Sélectionnez un client pour obtenir la décision basée sur le modèle en production.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# 📂 Charger les exemples de clients - CORRECTION ICI
@st.cache_data
def load_sample_clients():
    """
    Charge sample_clients.csv en cherchant dans plusieurs emplacements possibles
    """
    # Liste des emplacements possibles
    possible_paths = [
        "sample_clients.csv",  # Même dossier que le script
        "./sample_clients.csv",  # Explicite même dossier
        "../sample_clients.csv",  # Dossier parent
        "streamlit_app/sample_clients.csv",  # Si lancé depuis racine
        str(Path(__file__).parent / "sample_clients.csv"),  # Dossier du script
    ]

    # Essayer chaque chemin
    for path in possible_paths:
        if os.path.exists(path):
            st.success(f"✅ Fichier trouvé : {path}")
            return pd.read_csv(path)

    # Si aucun chemin ne fonctionne
    st.error(f"❌ sample_clients.csv introuvable dans les emplacements suivants :")
    for path in possible_paths:
        st.write(f"  - {os.path.abspath(path)}")

    st.info("📁 Répertoire de travail actuel : " + os.getcwd())
    st.info("📄 Fichiers dans le dossier actuel : " + str(os.listdir(".")))

    # Retourner un DataFrame vide pour éviter le crash
    return pd.DataFrame()


try:
    df_clients = load_sample_clients()

    if df_clients.empty:
        st.warning(
            "⚠️ Aucun client exemple disponible. Veuillez placer sample_clients.csv dans le bon dossier."
        )
        st.stop()

    client_names = [f"Client {i + 1}" for i in range(len(df_clients))]

    # 📋 Menu déroulant
    selected_name = st.selectbox(
        "👤 Choisissez un client exemple :", client_names, index=0
    )
    idx = client_names.index(selected_name)
    selected_client = df_clients.iloc[idx].to_dict()

    # ▶️ Bouton de prédiction
    if st.button("🔍 Obtenir la décision", type="primary", use_container_width=True):
        with st.spinner("Appel à l'API de scoring en production..."):
            try:
                response = requests.post(
                    API_URL, json={"data": selected_client}, timeout=10
                )

                if response.status_code == 200:
                    result = response.json()
                    proba = result["probability"]
                    decision = result["decision"]
                    threshold = result["threshold"]

                    # 🎯 Affichage principal — grand et clair
                    st.markdown("---")
                    st.subheader("📊 Résultat de la simulation")

                    # Couleurs métier
                    color = "red" if decision == 1 else "green"
                    status_text = "🔴 **REFUSÉ**" if decision == 1 else "🟢 **ACCEPTÉ**"
                    risk_level = "Élevé" if proba >= threshold else "Faible"

                    # Carte de résultat
                    st.markdown(
                        f"""
                        <div style="text-align: center; padding: 20px; background-color: #ffffff; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); margin: 10px 0;">
                            <h2 style="color: {color}; margin: 0;">{status_text}</h2>
                            <p style="font-size: 1.1em; color: #4b5563; margin: 10px 0;">
                                Probabilité de défaut : <strong>{proba:.2%}</strong><br>
                                Seuil métier : <strong>{threshold:.2%}</strong><br>
                                Niveau de risque : <strong>{risk_level}</strong>
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    # 📈 Visualisation : barre de probabilité vs seuil
                    st.subheader("📈 Visualisation du risque")
                    fig = go.Figure()
                    fig.add_trace(
                        go.Bar(
                            x=[proba],
                            y=["Probabilité"],
                            orientation="h",
                            marker_color=color,
                            width=0.5,
                        )
                    )
                    fig.add_vline(
                        x=threshold,
                        line_dash="dash",
                        line_color="orange",
                        annotation_text=f"Seuil ({threshold:.0%})",
                        annotation_position="top right",
                    )
                    fig.update_layout(
                        xaxis=dict(range=[0, 1], tickformat=".0%"),
                        height=150,
                        margin=dict(l=0, r=0, t=30, b=0),
                        showlegend=False,
                    )
                    st.plotly_chart(fig, use_container_width=True)

                else:
                    st.error(f"❌ Erreur API ({response.status_code}): {response.text}")

            except requests.exceptions.Timeout:
                st.error(
                    "⏰ Délai d'attente dépassé. L'API est peut-être en train de démarrer (Render). Réessayez dans 30 secondes."
                )
            except requests.exceptions.RequestException as e:
                st.error(f"🔌 Erreur de connexion : {str(e)}")

except Exception as e:
    st.exception("Une erreur inattendue s'est produite :")
    st.error(str(e))


# 📝 Section de debug (à retirer en production)
with st.expander("🔧 Informations de debug"):
    st.write("**Répertoire de travail :**", os.getcwd())
    st.write("**Fichiers dans le dossier :**", os.listdir("."))
    st.write(
        "**Chemin du script :**", __file__ if "__file__" in dir() else "Non disponible"
    )
