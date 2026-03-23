"""
Module : main.py
Auteur : William BLENON & Véronèse Nikina ZINSOU
Date   : 2026
Projet : Maintenance Prédictive Industrielle – M2 Data Engineering EFREI

Description :
    Point d'entrée principal du pipeline de maintenance prédictive.

    Orchestre l'ensemble des étapes dans l'ordre :
        1. Chargement et inspection des données (DataLoader),
        2. Préparation et découpage train/test (DataPreprocessor),
        3. Entraînement des 4 modèles (Logistic, RF, GBM, MLP),
        4. Évaluation comparative (Evaluator),
        5. Sélection et sauvegarde du meilleur modèle.

Usage :
    $ python main.py
"""

import os
import matplotlib
matplotlib.use('Agg')  # backend non-interactif (pas d'écran requis)
import matplotlib.pyplot as plt
from src.data.data_loader import DataLoader
from src.data.data_preprocessor import DataPreprocessor
from src.evaluation.evaluator import Evaluator
from src.explainability.explainer import Explainer
from src.models.gradient_boosting_model import GradientBoostingModel
from src.models.logistic_model import LogisticModel
from src.models.mlp_model import MLPModel
from src.models.random_forest_model import RandomForestModel

# ─── Configuration ────────────────────────────────────────────────────────────

DATA_PATH = "data/industrial_machine_maintenance.csv"
TARGET = "failure_within_24h"
MODEL_SAVE_DIR = "models"


def main() -> None:
    """
    Exécute le pipeline complet de maintenance prédictive.

    Étapes :
        1. Chargement des données,
        2. Validation des colonnes,
        3. Analyse de la distribution de la cible,
        4. Préparation (preprocessing + split),
        5. Entraînement des modèles,
        6. Évaluation comparative,
        7. Sauvegarde du meilleur modèle.
    """

    print("=" * 60)
    print(" SYSTÈME DE MAINTENANCE PRÉDICTIVE INDUSTRIELLE")
    print("=" * 60)

    # ── Étape 1 : Chargement ──────────────────────────────────────
    print("\n[1/5] Chargement des données...")
    loader = DataLoader(DATA_PATH)
    df = loader.load()

    info = loader.get_info()
    print(f"  → Mémoire utilisée : {info['memory_usage_mb']} Mo")
    print(f"  → Doublons : {info['duplicates']}")

    loader.validate_columns()

    dist = loader.get_target_distribution(TARGET)
    print(f"  → Distribution cible : {dist['percentages']}")

    # ── Étape 2 : Préparation ─────────────────────────────────────
    print("\n[2/5] Préparation des données...")
    preprocessor = DataPreprocessor(target=TARGET, test_size=0.2, random_state=42)
    X_train, X_test, y_train, y_test = preprocessor.fit_transform(df)

    # ── Étape 3 : Entraînement ────────────────────────────────────
    print("\n[3/5] Entraînement des modèles...")
    models = [
        LogisticModel(),             # Baseline
        RandomForestModel(),         # Ensemble (bagging)
        GradientBoostingModel(),     # Ensemble (boosting)
        MLPModel(),                  # Deep Learning (obligatoire)
    ]

    for model in models:
        model.train(X_train, y_train)

    # ── Étape 4 : Évaluation ──────────────────────────────────────
    print("\n[4/5] Évaluation comparative...")
    evaluator = Evaluator(models=models, X_test=X_test, y_test=y_test)
    results = evaluator.evaluate_all()

    print("\n── Tableau comparatif des performances ──")
    print(results.to_string(index=False))

    evaluator.plot_roc_curves()
    plt.savefig(f"{MODEL_SAVE_DIR}/roc_curves.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  → Courbes ROC sauvegardées : models/roc_curves.png")

    evaluator.plot_metrics_comparison()
    plt.savefig(f"{MODEL_SAVE_DIR}/metrics_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  → Comparaison métriques sauvegardée : models/metrics_comparison.png")

    # ── Étape 5 : Sélection et sauvegarde ────────────────────────
    print("\n[5/5] Sélection du meilleur modèle...")

    # Création du dossier models/ s'il n'existe pas
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

    # Recall prioritaire : minimiser les pannes non détectées
    best_model = evaluator.get_best_model(metric="recall")

    # Nom de fichier safe : supprime parenthèses et espaces
    model_filename = (
        best_model.name
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .lower()
        + ".pkl"
    )
    best_model.save(f"{MODEL_SAVE_DIR}/{model_filename}")
    print(f"  → Modèle sauvegardé : {MODEL_SAVE_DIR}/{model_filename}")

    # ── Étape 6 : Interprétabilité ────────────────────────────────
    print("\n[6/6] Interprétabilité du modèle final...")
    explainer = Explainer(
        model=best_model,
        X_train=X_train,
        X_test=X_test,
        feature_names=preprocessor.feature_names,
        y_test=y_test,
    )

    # Permutation Importance
    explainer.plot_permutation_importance(top_n=12, scoring="recall")
    plt.savefig(f"{MODEL_SAVE_DIR}/permutation_importance.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  → Permutation importance sauvegardée.")

    # Feature Importance native (si modèle arbre)
    try:
        explainer.plot_native_importance(top_n=12)
        plt.savefig(f"{MODEL_SAVE_DIR}/feature_importance.png", dpi=150, bbox_inches='tight')
        plt.close()
        print("  → Feature importance native sauvegardée.")
    except AttributeError:
        print(f"  → Feature importance native non disponible pour {best_model.name}.")

    # SHAP (niveau avancé — modèle final)
    explainer.compute_shap_values(max_samples=200)

    explainer.plot_shap_summary()
    plt.savefig(f"{MODEL_SAVE_DIR}/shap_summary.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  → SHAP summary sauvegardé.")

    explainer.plot_shap_bar(top_n=12)
    plt.savefig(f"{MODEL_SAVE_DIR}/shap_bar.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  → SHAP bar sauvegardé.")

    explainer.plot_shap_waterfall(sample_index=0)
    plt.savefig(f"{MODEL_SAVE_DIR}/shap_waterfall.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  → SHAP waterfall sauvegardé.")

    # Top features exportées (utile pour le dashboard)
    top_features = explainer.get_top_features(top_n=10)
    print("\n── Top features SHAP ──")
    print(top_features.to_string(index=False))

    print("\n" + "=" * 60)
    print(f" Pipeline terminé. Modèle retenu : {best_model.name}")
    print("=" * 60)


if __name__ == "__main__":
    main()