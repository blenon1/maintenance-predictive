"""
Module : evaluator.py
Auteur : William BLENON & Véronèse Nikina ZINSOU
Date   : 2026
Projet : Maintenance Prédictive Industrielle – M2 Data Engineering EFREI

Description :
    Ce module contient la classe Evaluator, responsable de l'évaluation
    comparative de tous les modèles sur le jeu de test.

    Elle génère :
        - Un tableau comparatif des métriques (Accuracy, Precision, Recall, F1, ROC-AUC),
        - Les matrices de confusion,
        - Les courbes ROC.

    Métriques prioritaires pour la classification de pannes industrielles :
        - Recall (sensibilité) : minimise les faux négatifs (pannes non détectées).
          Un faux négatif = panne manquée = coût élevé.
        - F1-Score : compromis entre Precision et Recall.
        - ROC-AUC : robuste au déséquilibre des classes.

Dépendances :
    - scikit-learn
    - pandas
    - matplotlib
    - seaborn
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.models.base_model import BaseModel


class Evaluator:
    """
    Classe responsable de l'évaluation et de la comparaison des modèles.

    Calcule les métriques de classification pour chaque modèle sur le jeu
    de test, produit des visualisations (matrices de confusion, courbes ROC)
    et identifie le meilleur modèle selon le critère choisi.

    Attributes:
        models (list[BaseModel]): Liste des modèles entraînés à comparer.
        X_test (np.ndarray): Features du jeu de test.
        y_test (np.ndarray): Labels réels du jeu de test.
        results (pd.DataFrame | None): Tableau comparatif des métriques,
                                       disponible après evaluate_all().

    Example:
        >>> evaluator = Evaluator(models=[lr, rf, gb, mlp], X_test=X_test, y_test=y_test)
        >>> results = evaluator.evaluate_all()
        >>> evaluator.plot_roc_curves()
        >>> best = evaluator.get_best_model(metric="f1")
    """

    def __init__(
        self,
        models: list[BaseModel],
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> None:
        """
        Initialise l'Evaluator avec les modèles et les données de test.

        Args:
            models (list[BaseModel]): Liste des modèles entraînés à évaluer.
            X_test (np.ndarray): Features du jeu de test (déjà transformées).
            y_test (np.ndarray): Labels réels du jeu de test.

        Raises:
            ValueError: Si la liste de modèles est vide.
            TypeError: Si un élément de models n'est pas une instance de BaseModel.
        """
        if not models:
            raise ValueError("La liste de modèles ne peut pas être vide.")

        for m in models:
            if not isinstance(m, BaseModel):
                raise TypeError(
                    f"Tous les modèles doivent hériter de BaseModel. "
                    f"Reçu : {type(m).__name__}"
                )

        self.models: list[BaseModel] = models
        self.X_test: np.ndarray = X_test
        self.y_test: np.ndarray = y_test
        self.results: pd.DataFrame | None = None

    def evaluate_all(self) -> pd.DataFrame:
        """
        Évalue tous les modèles sur le jeu de test et produit un tableau comparatif.

        Métriques calculées pour chaque modèle :
            - Accuracy  : taux de bonnes prédictions global (à nuancer si déséquilibre).
            - Precision : parmi les pannes prédites, combien sont réelles ?
            - Recall    : parmi les pannes réelles, combien sont détectées ? (prioritaire)
            - F1-Score  : moyenne harmonique de Precision et Recall.
            - ROC-AUC   : aire sous la courbe ROC (robuste au déséquilibre).

        Returns:
            pd.DataFrame: Tableau comparatif des métriques, trié par F1-Score décroissant.

        Example:
            >>> results = evaluator.evaluate_all()
            >>> print(results.to_string())
        """
        rows = []

        for model in self.models:
            y_pred = model.predict(self.X_test)
            y_proba = model.predict_proba(self.X_test)[:, 1]

            row = {
                "Modèle": model.name,
                "Accuracy": round(accuracy_score(self.y_test, y_pred), 4),
                "Precision": round(
                    precision_score(self.y_test, y_pred, zero_division=0), 4
                ),
                "Recall": round(
                    recall_score(self.y_test, y_pred, zero_division=0), 4
                ),
                "F1-Score": round(
                    f1_score(self.y_test, y_pred, zero_division=0), 4
                ),
                "ROC-AUC": round(roc_auc_score(self.y_test, y_proba), 4),
            }
            rows.append(row)
            print(
                f"[Evaluator] {model.name} → "
                f"Recall={row['Recall']} | F1={row['F1-Score']} | AUC={row['ROC-AUC']}"
            )

        self.results = pd.DataFrame(rows).sort_values("F1-Score", ascending=False)
        self.results.reset_index(drop=True, inplace=True)

        return self.results

    def plot_confusion_matrix(self, model: BaseModel) -> None:
        """
        Affiche la matrice de confusion pour un modèle donné.

        La matrice de confusion permet d'analyser en détail les types d'erreurs :
            - Faux Négatifs (FN) : pannes non détectées → coût élevé en industrie.
            - Faux Positifs (FP) : fausses alarmes → maintenance inutile.

        Args:
            model (BaseModel): Le modèle dont on veut afficher la matrice.

        Example:
            >>> evaluator.plot_confusion_matrix(rf_model)
        """
        y_pred = model.predict(self.X_test)
        cm = confusion_matrix(self.y_test, y_pred)

        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Pas de panne", "Panne"],
            yticklabels=["Pas de panne", "Panne"],
            ax=ax,
        )
        ax.set_xlabel("Prédiction")
        ax.set_ylabel("Réalité")
        ax.set_title(f"Matrice de confusion – {model.name}")
        plt.tight_layout()

    def plot_roc_curves(self) -> None:
        """
        Affiche les courbes ROC de tous les modèles sur un même graphique.

        La courbe ROC représente le compromis entre le taux de vrais positifs (Recall)
        et le taux de faux positifs en fonction du seuil de décision.
        Une AUC proche de 1 indique un excellent modèle.

        Example:
            >>> evaluator.plot_roc_curves()
        """
        fig, ax = plt.subplots(figsize=(8, 6))

        for model in self.models:
            y_proba = model.predict_proba(self.X_test)[:, 1]
            fpr, tpr, _ = roc_curve(self.y_test, y_proba)
            auc = round(roc_auc_score(self.y_test, y_proba), 3)
            ax.plot(fpr, tpr, label=f"{model.name} (AUC = {auc})")

        # Ligne de référence : modèle aléatoire (AUC = 0.5)
        ax.plot([0, 1], [0, 1], "k--", label="Aléatoire (AUC = 0.5)")

        ax.set_xlabel("Taux de Faux Positifs (FPR)")
        ax.set_ylabel("Taux de Vrais Positifs (Recall / TPR)")
        ax.set_title("Comparaison des courbes ROC – Tous les modèles")
        ax.legend(loc="lower right")
        ax.grid(alpha=0.3)
        plt.tight_layout()

    def plot_metrics_comparison(self) -> None:
        """
        Affiche un graphique en barres comparant les métriques de tous les modèles.

        Nécessite d'avoir appelé evaluate_all() au préalable.

        Raises:
            RuntimeError: Si evaluate_all() n'a pas encore été appelé.

        Example:
            >>> evaluator.evaluate_all()
            >>> evaluator.plot_metrics_comparison()
        """
        self._check_evaluated()

        metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
        df_plot = self.results.set_index("Modèle")[metrics]

        ax = df_plot.plot(kind="bar", figsize=(12, 6), colormap="Set2", edgecolor="white")
        ax.set_title("Comparaison des métriques – Tous les modèles")
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1.1)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
        ax.legend(loc="lower right")
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()

    def get_best_model(self, metric: str = "f1") -> BaseModel:
        """
        Retourne le modèle le plus performant selon la métrique choisie.

        Dans un contexte de maintenance prédictive, le Recall est souvent
        la métrique prioritaire (minimiser les pannes non détectées).
        Le F1-Score offre un bon compromis général.

        Args:
            metric (str): Métrique de sélection. Valeurs possibles :
                          "accuracy", "precision", "recall", "f1", "auc".
                          Par défaut : "f1".

        Returns:
            BaseModel: Le modèle ayant le meilleur score sur la métrique choisie.

        Raises:
            RuntimeError: Si evaluate_all() n'a pas encore été appelé.
            ValueError: Si la métrique spécifiée n'est pas reconnue.

        Example:
            >>> best = evaluator.get_best_model(metric="recall")
            >>> print(f"Meilleur modèle : {best.name}")
        """
        self._check_evaluated()

        metric_map = {
            "accuracy": "Accuracy",
            "precision": "Precision",
            "recall": "Recall",
            "f1": "F1-Score",
            "auc": "ROC-AUC",
        }

        if metric not in metric_map:
            raise ValueError(
                f"Métrique '{metric}' non reconnue. "
                f"Valeurs acceptées : {list(metric_map.keys())}"
            )

        col = metric_map[metric]
        best_name = self.results.loc[self.results[col].idxmax(), "Modèle"]
        best_model = next(m for m in self.models if m.name == best_name)

        print(
            f"[Evaluator] ✅ Meilleur modèle ({metric}) : {best_model.name} "
            f"({col} = {self.results.loc[self.results['Modèle'] == best_name, col].values[0]})"
        )

        return best_model

    def _check_evaluated(self) -> None:
        """
        Méthode privée : vérifie que evaluate_all() a bien été appelé.

        Raises:
            RuntimeError: Si self.results est None.
        """
        if self.results is None:
            raise RuntimeError(
                "Aucun résultat disponible. Appelez d'abord evaluate_all()."
            )

    def __repr__(self) -> str:
        """
        Représentation officielle de l'objet Evaluator.

        Returns:
            str: Représentation lisible listant les modèles évalués.
        """
        model_names = [m.name for m in self.models]
        return (
            f"Evaluator("
            f"models={model_names}, "
            f"n_test_samples={len(self.y_test)}, "
            f"evaluated={self.results is not None}"
            f")"
        )