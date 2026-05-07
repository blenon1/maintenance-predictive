# Système Intelligent de Maintenance Prédictive Industrielle

> Projet Data Science M2 – EFREI · Data Engineering & AI · 2026  
> Épreuve certifiante RNCP36739 – Bloc 4
> Realisé par : William BLENON & Véronèse Nikina ZINSOU
> Encadrante : Sarah Malaeb - EFREI

---

## Description

Plateforme complète de **maintenance prédictive industrielle** construite en
approche POO (Programmation Orientée Objet). Le système détecte les risques
de panne dans les 24 prochaines heures à partir de données capteurs
(vibration, température, RPM, pression, mode opératoire).

**Tâche choisie :** Classification binaire - `failure_within_24h`

---

## Architecture du projet

```
predictive_maintenance/
│
├── data/                          # Dataset CSV (à placer ici)
│   └── industrial_machine_maintenance.csv
│
├── models/                        # Modèles sérialisés (générés par main.py)
│
├── src/
│   ├── data/
│   │   ├── data_loader.py         # Chargement & inspection du dataset
│   │   └── data_preprocessor.py   # Pipeline sklearn (sans data leakage)
│   │
│   ├── models/
│   │   ├── base_model.py          # Classe abstraite (interface commune)
│   │   ├── logistic_model.py      # Régression Logistique (baseline)
│   │   ├── random_forest_model.py # Random Forest
│   │   ├── gradient_boosting_model.py  # XGBoost
│   │   └── mlp_model.py           # MLP – Deep Learning (obligatoire)
│   │
│   ├── evaluation/
│   │   └── evaluator.py           # Comparaison multi-modèles
│   │
│   └── explainability/
│       └── explainer.py           # Feature Importance + SHAP
│
├── dashboard/
│   └── app.py                     # Dashboard Streamlit (5 pages)
│
├── api/
│   ├── main.py                    # API REST FastAPI
│   └── test_api.py                # Suite de tests
│
├── notebooks/
│   └── EDA.ipynb                  # Analyse exploratoire
│
├── main.py                        # Pipeline principal
├── requirements.txt
└── README.md
```

---

## Installation

```bash
# 1. Cloner le dépôt
git clone https://github.com/votre-repo/predictive-maintenance.git
cd predictive-maintenance

# 2. Créer un environnement virtuel
python -m venv venv
source venv/bin/activate       # Linux/macOS
venv\Scripts\activate          # Windows

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Placer le dataset
# Télécharger depuis : https://www.kaggle.com/datasets/tatheerabbas/industrial-machine-predictive-maintenance
# Placer dans : data/industrial_machine_maintenance.csv
```

---

## Utilisation

### Pipeline complet (entraînement + évaluation)

```bash
python main.py
```

Ce script exécute dans l'ordre :
1. Chargement et validation des données
2. Préparation (split stratifié 80/20, StandardScaler, OneHotEncoder)
3. Entraînement des 4 modèles
4. Évaluation comparative (Accuracy, Precision, Recall, F1, ROC-AUC)
5. Sélection du meilleur modèle (critère : Recall)
6. Interprétabilité (Permutation Importance + SHAP)

### Dashboard décisionnel

```bash
streamlit run dashboard/app.py
```

Accessible sur : `http://localhost:8501`

### API REST

```bash
# Démarrer le serveur
uvicorn api.main:app --reload --port 8000

# Documentation interactive
http://localhost:8000/docs

# Lancer les tests
python api/test_api.py
```

---

## API REST – Endpoints

| Méthode | Endpoint         | Description                              |
|---------|------------------|------------------------------------------|
| GET     | `/health`        | État du service et du modèle chargé      |
| GET     | `/model-info`    | Métadonnées du modèle actif              |
| POST    | `/predict`       | Prédiction unitaire (1 machine)          |
| POST    | `/predict/batch` | Prédiction batch (jusqu'à 100 machines)  |

### Exemple de requête `/predict`

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "vibration_rms": 8.5,
    "temperature_motor": 145.0,
    "rpm": 4200.0,
    "pressure_level": 12.5,
    "operating_mode": "degraded",
    "threshold": 0.4
  }'
```

### Exemple de réponse

```json
{
  "prediction": 1,
  "probability": 0.8732,
  "risk_level": "HIGH",
  "recommendation": "RISQUE ÉLEVÉ DE PANNE. Intervention immédiate recommandée...",
  "model_name": "Random Forest",
  "threshold_used": 0.4,
  "timestamp": "2025-03-14T10:23:45.123456"
}
```

---

## 🤖 Modèles implémentés

| Modèle                    | Type          | Rôle                          |
|---------------------------|---------------|-------------------------------|
| Logistic Regression       | ML classique  | Baseline (référence)          |
| Random Forest             | ML classique  | Ensemble (bagging)            |
| Gradient Boosting (XGBoost) | ML classique | Ensemble (boosting)           |
| MLP                       | Deep Learning | Obligatoire (interactions profondes) |

---

## 📊 Métriques d'évaluation

Métriques prioritaires pour la classification de pannes industrielles :

- **Recall** : prioritaire — minimise les pannes non détectées (faux négatifs coûteux)
- **F1-Score** : compromis Precision/Recall
- **ROC-AUC** : robuste au déséquilibre de classes

---

## 🧠 Interprétabilité

Trois niveaux d'explicabilité implémentés :

1. **Feature Importance native** (Gini) — modèles arbres uniquement
2. **Permutation Importance** — tous modèles, recommandée
3. **SHAP** — explication locale et globale (modèle final)

---

## 👥 Auteurs

- William BLENON
- Véronèse Nikina ZINSOU

**Encadrante :** Sarah Malaeb – EFREI, Data Engineering & AI