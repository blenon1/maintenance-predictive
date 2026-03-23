"""
Module : mlp_model.py
Auteur : William BLENON & Véronèse Nikina ZINSOU
Date   : 2026
Encadrante : Sarah Malaeb – EFREI
Projet : Maintenance Prédictive Industrielle – M2 Data Engineering EFREI

Description :
    Modèle Deep Learning (MLP - Multi Layer Perceptron) pour la classification
    binaire de pannes industrielles.

    Le MLP est le modèle Deep Learning obligatoire du projet. Il capture des
    interactions non linéaires complexes entre les variables capteurs, que les
    modèles classiques ne peuvent modéliser directement.

    Attention : le Deep Learning n'est pas toujours supérieur. Il doit être
    justifié par un gain réel par rapport aux modèles classiques, en tenant
    compte du compromis biais/variance et du risque d'overfitting.

    Architecture retenue :
        Input → Dense(128, ReLU) → Dropout(0.3) → Dense(64, ReLU)
             → Dropout(0.2) → Dense(1, Sigmoid)

    Avantages :
        - Capture des interactions non linéaires profondes,
        - Flexible (architecture adaptable).

    Limites :
        - Boîte noire (moins interprétable),
        - Nécessite plus de données et de temps d'entraînement,
        - Risque d'overfitting si mal régularisé.
"""

import numpy as np
from sklearn.neural_network import MLPClassifier

from src.models.base_model import BaseModel


class MLPModel(BaseModel):
    """
    Modèle MLP (Multi Layer Perceptron) pour la classification binaire de pannes.

    Implémente le modèle Deep Learning obligatoire du projet via sklearn.
    Pour des architectures plus complexes (LSTM, CNN), TensorFlow/Keras
    serait préférable, mais sklearn offre ici un bon compromis pédagogique.

    Architecture par défaut :
        - Couche 1 : 128 neurones, activation ReLU
        - Couche 2 : 64 neurones, activation ReLU
        - Sortie : 1 neurone, activation logistique (sigmoid)

    Example:
        >>> model = MLPModel()
        >>> model.train(X_train, y_train)
        >>> probas = model.predict_proba(X_test)
    """

    DEFAULT_PARAMS = {
        "hidden_layer_sizes": (128, 64),  # Architecture : 2 couches cachées
        "activation": "relu",             # Fonction d'activation (ReLU évite le vanishing gradient)
        "solver": "adam",                 # Optimiseur adaptatif (recommandé pour MLP)
        "alpha": 0.001,                   # Régularisation L2 (évite l'overfitting)
        "batch_size": 64,                 # Taille des mini-batches
        "learning_rate": "adaptive",      # Réduit le taux si la loss stagne
        "max_iter": 200,                  # Nombre d'époques maximum
        "early_stopping": True,           # Arrêt anticipé si pas d'amélioration
        "validation_fraction": 0.1,       # 10% du train pour la validation
        "n_iter_no_change": 15,           # Patience avant arrêt anticipé
        "random_state": 42,
        "verbose": False,
    }

    def __init__(self, params: dict | None = None) -> None:
        """
        Initialise le modèle MLP avec les hyperparamètres spécifiés.

        Args:
            params (dict | None): Hyperparamètres personnalisés. Les paramètres
                                  non spécifiés prennent les valeurs DEFAULT_PARAMS.

        Example:
            >>> model = MLPModel(params={"hidden_layer_sizes": (256, 128, 64)})
        """
        merged_params = {**self.DEFAULT_PARAMS, **(params or {})}
        super().__init__(name="MLP (Deep Learning)", params=merged_params)
        self.model = MLPClassifier(**self.params)

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Entraîne le réseau de neurones sur les données d'entraînement.

        L'early stopping est activé par défaut pour éviter l'overfitting :
        l'entraînement s'arrête automatiquement si la loss de validation
        ne s'améliore pas pendant n_iter_no_change époques.

        Args:
            X_train (np.ndarray): Features d'entraînement (standardisation requise —
                                  assurée par DataPreprocessor).
            y_train (np.ndarray): Labels d'entraînement.

        Note:
            Le MLP sklearn est sensible à l'échelle des features. Il est donc
            impératif que DataPreprocessor ait appliqué StandardScaler avant
            l'entraînement.
        """
        print(f"[{self.name}] Entraînement en cours...")
        print(
            f"  → Architecture : {self.params['hidden_layer_sizes']}\n"
            f"  → Early stopping : {self.params['early_stopping']}\n"
            f"  → Max epochs : {self.params['max_iter']}"
        )

        self.model.fit(X_train, y_train)
        self.is_trained = True

        # Rapport post-entraînement
        n_iter = self.model.n_iter_
        loss = round(self.model.loss_, 4)
        print(
            f"[{self.name}] ✅ Entraînement terminé.\n"
            f"  → Epochs réalisées : {n_iter} | Loss finale : {loss}"
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Prédit la classe de panne pour chaque observation.

        Args:
            X (np.ndarray): Features à prédire.

        Returns:
            np.ndarray: Classes prédites (0 ou 1).
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
        """
        self._check_trained()
        return self.model.predict_proba(X)

    def get_loss_curve(self) -> list:
        """
        Retourne la courbe de loss au fil des époques d'entraînement.

        Utile pour analyser la convergence du modèle et détecter
        un éventuel overfitting (divergence train vs validation).

        Returns:
            list: Valeurs de loss par époque.

        Raises:
            RuntimeError: Si le modèle n'a pas encore été entraîné.

        Example:
            >>> loss_curve = model.get_loss_curve()
            >>> plt.plot(loss_curve)
            >>> plt.title("Convergence du MLP")
        """
        self._check_trained()
        return self.model.loss_curve_