"""
Module : logistic_model.py
Auteur : William BLENON & Véronèse Nikina ZINSOU
Date   : 2026
Encadrante : Sarah Malaeb - EFREI
Projet : Maintenance Prédictive Industrielle – M2 Data Engineering EFREI

Description :
    Modèle de référence (baseline) : Régression Logistique.

    La régression logistique est le premier modèle à implémenter dans toute
    démarche rigoureuse. Elle sert de baseline : tout modèle plus complexe
    doit surpasser cette référence pour justifier sa complexité additionnelle.

    Avantages :
        - Rapide à entraîner,
        - Interprétable (coefficients = importance des features),
        - Robuste sur des données linéairement séparables.

    Limites :
        - Ne capture pas les relations non linéaires,
        - Sensible à la multicolinéarité.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression

from src.models.base_model import BaseModel


class LogisticModel(BaseModel):
    """
    Modèle de régression logistique pour la classification binaire.

    Sert de modèle de référence (baseline). Tout autre modèle doit être
    comparé à cette référence pour justifier sa complexité additionnelle.

    Example:
        >>> model = LogisticModel()
        >>> model.train(X_train, y_train)
        >>> preds = model.predict(X_test)
        >>> probas = model.predict_proba(X_test)
    """

    DEFAULT_PARAMS = {
        "C": 1.0,                    # Inverse de la régularisation (plus C est petit, plus forte)
        "max_iter": 1000,            # Nombre max d'itérations pour la convergence
        "class_weight": "balanced",  # Compense le déséquilibre des classes
        "random_state": 42,
        "solver": "lbfgs",
    }

    def __init__(self, params: dict | None = None) -> None:
        """
        Initialise le modèle de régression logistique.

        Args:
            params (dict | None): Hyperparamètres personnalisés. Les clés non
                                  spécifiées prennent les valeurs de DEFAULT_PARAMS.
        """
        merged_params = {**self.DEFAULT_PARAMS, **(params or {})}
        super().__init__(name="Logistic Regression", params=merged_params)
        self.model = LogisticRegression(**self.params)

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Entraîne la régression logistique sur les données d'entraînement.

        Args:
            X_train (np.ndarray): Features d'entraînement.
            y_train (np.ndarray): Labels d'entraînement (0 ou 1).
        """
        print(f"[{self.name}] Entraînement en cours...")
        self.model.fit(X_train, y_train)
        self.is_trained = True
        print(f"[{self.name}] ✅ Entraînement terminé.")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Prédit la classe (0 = pas de panne, 1 = panne dans 24h).

        Args:
            X (np.ndarray): Features à prédire.

        Returns:
            np.ndarray: Classes prédites.
        """
        self._check_trained()
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Retourne les probabilités de panne pour chaque observation.

        Args:
            X (np.ndarray): Features à prédire.

        Returns:
            np.ndarray: Probabilités de forme (n_samples, 2).
                        Colonne 1 = probabilité de panne.
        """
        self._check_trained()
        return self.model.predict_proba(X)