"""
Module : gradient_boosting_model.py
Auteur : William BLENON & Véronèse Nikina ZINSOU
Date   : 2026
Encadrante : Sarah Malaeb - EFREI
Projet : Maintenance Prédictive Industrielle – M2 Data Engineering EFREI

Description :
    Modèle Gradient Boosting (via XGBoost) pour la classification binaire.

    Contrairement au Random Forest (bagging), le Gradient Boosting construit
    les arbres séquentiellement : chaque arbre corrige les erreurs du précédent.
    Il offre généralement de meilleures performances prédictives mais est plus
    sensible à l'overfitting et nécessite un tuning plus fin.

    Avantages :
        - Souvent le meilleur modèle tabulaire en compétition,
        - Feature importance native,
        - Gestion native du déséquilibre (scale_pos_weight).

    Limites :
        - Plus lent à entraîner,
        - Plus de risque d'overfitting si mal paramétré.
"""

import numpy as np
from xgboost import XGBClassifier

from src.models.base_model import BaseModel


class GradientBoostingModel(BaseModel):
    """
    Modèle Gradient Boosting (XGBoost) pour la classification binaire de pannes.

    Example:
        >>> model = GradientBoostingModel()
        >>> model.train(X_train, y_train)
        >>> probas = model.predict_proba(X_test)
    """

    DEFAULT_PARAMS = {
        "n_estimators": 100,
        "learning_rate": 0.1,     # Pas d'apprentissage (shrinkage)
        "max_depth": 6,           # Profondeur maximale de chaque arbre
        "subsample": 0.8,         # Fraction de données utilisée par arbre (réduction overfitting)
        "colsample_bytree": 0.8,  # Fraction de features utilisée par arbre
        "use_label_encoder": False,
        "eval_metric": "logloss",
        "random_state": 42,
        "n_jobs": -1,
    }

    def __init__(self, params: dict | None = None) -> None:
        merged_params = {**self.DEFAULT_PARAMS, **(params or {})}
        super().__init__(name="Gradient Boosting (XGBoost)", params=merged_params)
        self.model = XGBClassifier(**self.params)

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Entraîne le modèle XGBoost sur les données d'entraînement.

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
        Retourne l'importance native des features (gain moyen par feature).

        Returns:
            np.ndarray: Importances des features.
        """
        self._check_trained()
        return self.model.feature_importances_