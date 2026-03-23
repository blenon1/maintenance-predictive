"""
Module : random_forest_model.py
Auteur : William BLENON & Véronèse Nikina ZINSOU
Date   : 2026
Encadrante : Sarah Malaeb – EFREI
Projet : Maintenance Prédictive Industrielle – M2 Data Engineering EFREI

Description :
    Modèle Random Forest pour la classification binaire de pannes industrielles.

    Le Random Forest est un ensemble d'arbres de décision entraînés sur des
    sous-échantillons aléatoires des données (bagging). Il capture les relations
    non linéaires et offre une feature importance native, précieuse pour
    l'interprétabilité en contexte industriel.

    Avantages :
        - Robuste au bruit et aux outliers,
        - Feature importance native,
        - Peu sensible à la mise à l'échelle des features.

    Limites :
        - Moins interprétable qu'une régression logistique,
        - Temps d'inférence plus long que les modèles linéaires.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from src.models.base_model import BaseModel


class RandomForestModel(BaseModel):
    """
    Modèle Random Forest pour la classification binaire de pannes.

    Offre une feature importance native utilisée dans la phase d'interprétabilité.

    Example:
        >>> model = RandomForestModel(params={"n_estimators": 200})
        >>> model.train(X_train, y_train)
        >>> importance = model.get_feature_importance()
    """

    DEFAULT_PARAMS = {
        "n_estimators": 100,         # Nombre d'arbres dans la forêt
        "max_depth": None,           # Profondeur maximale (None = illimitée)
        "min_samples_split": 2,
        "class_weight": "balanced",  # Compense le déséquilibre des classes
        "random_state": 42,
        "n_jobs": -1,                # Utilise tous les cœurs disponibles
    }

    def __init__(self, params: dict | None = None) -> None:
        merged_params = {**self.DEFAULT_PARAMS, **(params or {})}
        super().__init__(name="Random Forest", params=merged_params)
        self.model = RandomForestClassifier(**self.params)

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Entraîne le Random Forest sur les données d'entraînement.

        Args:
            X_train (np.ndarray): Features d'entraînement.
            y_train (np.ndarray): Labels d'entraînement.
        """
        print(f"[{self.name}] Entraînement en cours...")
        self.model.fit(X_train, y_train)
        self.is_trained = True
        print(f"[{self.name}] ✅ Entraînement terminé.")

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_trained()
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._check_trained()
        return self.model.predict_proba(X)

    def get_feature_importance(self) -> np.ndarray:
        """
        Retourne l'importance native des features (basée sur la réduction d'impureté Gini).

        Utilisée dans la phase d'interprétabilité pour identifier les capteurs
        industriels les plus influents dans la détection de pannes.

        Returns:
            np.ndarray: Tableau d'importances normalisées (somme = 1).

        Raises:
            RuntimeError: Si le modèle n'a pas encore été entraîné.

        Example:
            >>> importance = model.get_feature_importance()
            >>> # Combiner avec feature_names pour une visualisation claire
        """
        self._check_trained()
        return self.model.feature_importances_