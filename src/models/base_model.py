"""
Module : base_model.py
Auteur : William BLENON & Véronèse Nikina ZINSOU
Date   : 2026
Encadrante : Sarah Malaeb - EFREI
Projet : Maintenance Prédictive Industrielle – M2 Data Engineering EFREI

Description :
    Ce module définit la classe abstraite BaseModel, socle commun à tous
    les modèles de Machine Learning et Deep Learning du projet.

    Grâce à l'héritage, chaque modèle concret (LogisticModel, RandomForestModel,
    MLPModel, etc.) partage la même interface : train(), predict(), save(), load().
    Cela garantit une comparaison homogène dans l'Evaluator et une intégration
    uniforme dans le dashboard et l'API.

    Principe SOLID appliqué :
        - S (Single Responsibility) : chaque modèle gère uniquement sa logique ML.
        - O (Open/Closed) : on étend sans modifier la classe de base.
        - L (Liskov Substitution) : tout modèle fils peut remplacer BaseModel.

Dépendances :
    - abc (bibliothèque standard Python)
    - numpy
    - joblib
"""

import os
from abc import ABC, abstractmethod

import joblib
import numpy as np


class BaseModel(ABC):
    """
    Classe abstraite définissant l'interface commune à tous les modèles du projet.

    Toute classe modèle concrète (LogisticModel, RandomForestModel, MLPModel, etc.)
    doit hériter de BaseModel et implémenter les méthodes abstraites :
        - train()
        - predict()
        - predict_proba()

    Les méthodes save() et load() sont fournies avec une implémentation par défaut
    basée sur joblib, utilisable par tous les modèles sauf ceux nécessitant
    une sérialisation spécifique (ex. modèles Keras).

    Attributes:
        name (str): Nom lisible du modèle (ex. "Random Forest").
        model: Instance du modèle sklearn ou équivalent, initialisée dans les
               classes filles. Vaut None avant l'appel à train().
        is_trained (bool): Indique si le modèle a été entraîné.
        params (dict): Hyperparamètres utilisés pour l'instanciation du modèle.

    Example:
        >>> class MyModel(BaseModel):
        ...     def train(self, X_train, y_train): ...
        ...     def predict(self, X): ...
        ...     def predict_proba(self, X): ...
        >>> m = MyModel(name="My Model")
        >>> m.train(X_train, y_train)
        >>> preds = m.predict(X_test)
    """

    def __init__(self, name: str, params: dict | None = None) -> None:
        """
        Initialise le modèle de base avec un nom et des hyperparamètres optionnels.

        Args:
            name (str): Nom lisible du modèle (utilisé dans les logs et les comparatifs).
            params (dict | None): Dictionnaire des hyperparamètres. Si None, utilise
                                  les valeurs par défaut définies dans la classe fille.

        Raises:
            TypeError: Si name n'est pas une chaîne de caractères.

        Example:
            >>> model = RandomForestModel(
            ...     name="Random Forest",
            ...     params={"n_estimators": 200, "max_depth": 10}
            ... )
        """
        if not isinstance(name, str):
            raise TypeError(
                f"'name' doit être une chaîne de caractères. Reçu : {type(name).__name__}"
            )

        self.name: str = name
        self.params: dict = params if params is not None else {}
        self.model = None         # Initialisé dans les classes filles
        self.is_trained: bool = False

    @abstractmethod
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Entraîne le modèle sur les données d'entraînement.

        Cette méthode doit obligatoirement être implémentée dans chaque
        classe fille. Elle doit mettre à jour self.model et passer
        self.is_trained à True en fin d'entraînement.

        Args:
            X_train (np.ndarray): Features d'entraînement (déjà transformées
                                  par DataPreprocessor).
            y_train (np.ndarray): Labels d'entraînement.

        Example:
            >>> model.train(X_train, y_train)
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Génère les prédictions de classe pour un jeu de données.

        Args:
            X (np.ndarray): Features à prédire (déjà transformées).

        Returns:
            np.ndarray: Tableau des classes prédites (0 ou 1 pour la classification binaire).

        Example:
            >>> predictions = model.predict(X_test)
            >>> print(predictions[:5])
            [0 1 0 0 1]
        """
        pass

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Génère les probabilités de classe pour un jeu de données.

        Utilisées pour les métriques ROC-AUC et l'ajustement du seuil de décision.
        Indispensables dans un contexte industriel où les faux négatifs
        (panne non détectée) ont un coût élevé.

        Args:
            X (np.ndarray): Features à prédire (déjà transformées).

        Returns:
            np.ndarray: Tableau 2D de probabilités de forme (n_samples, n_classes).
                        La colonne 1 correspond à la probabilité de panne.

        Example:
            >>> probas = model.predict_proba(X_test)
            >>> print(probas[:3, 1])  # Probabilité de panne
            [0.82, 0.13, 0.67]
        """
        pass

    def save(self, path: str) -> None:
        """
        Sérialise le modèle entraîné sur le disque via joblib.

        joblib est préféré à pickle pour les modèles sklearn car il gère
        mieux les objets numpy et offre une meilleure compression.

        Args:
            path (str): Chemin complet du fichier de sauvegarde (ex. "models/rf.pkl").

        Raises:
            RuntimeError: Si le modèle n'a pas encore été entraîné.
            OSError: Si le répertoire cible n'existe pas et ne peut être créé.

        Example:
            >>> model.save("models/random_forest.pkl")
        """
        self._check_trained()

        # Création du répertoire si nécessaire
        os.makedirs(os.path.dirname(path), exist_ok=True)

        joblib.dump(self.model, path)
        print(f"[{self.name}] Modèle sauvegardé → {path}")

    def load(self, path: str) -> None:
        """
        Charge un modèle préalablement sérialisé depuis le disque.

        Utilisée par l'API et le dashboard pour charger le modèle final
        sans le réentraîner.

        Args:
            path (str): Chemin vers le fichier .pkl du modèle sauvegardé.

        Raises:
            FileNotFoundError: Si le fichier spécifié n'existe pas.

        Example:
            >>> model.load("models/random_forest.pkl")
            >>> predictions = model.predict(X_test)
        """
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Fichier modèle introuvable : '{path}'"
            )

        self.model = joblib.load(path)
        self.is_trained = True
        print(f"[{self.name}] Modèle chargé depuis → {path}")

    def _check_trained(self) -> None:
        """
        Méthode privée : vérifie que le modèle a bien été entraîné.

        Appelée en début de predict(), predict_proba() et save() pour éviter
        des erreurs silencieuses liées à un modèle non initialisé.

        Raises:
            RuntimeError: Si is_trained est False.
        """
        if not self.is_trained:
            raise RuntimeError(
                f"[{self.name}] Le modèle n'a pas encore été entraîné. "
                f"Appelez d'abord train() avant toute prédiction ou sauvegarde."
            )

    def __repr__(self) -> str:
        """
        Représentation officielle de l'objet BaseModel.

        Returns:
            str: Représentation lisible indiquant le nom et l'état d'entraînement.

        Example:
            >>> print(repr(model))
            BaseModel(name='Random Forest', trained=True)
        """
        return (
            f"{self.__class__.__name__}("
            f"name='{self.name}', "
            f"trained={self.is_trained}"
            f")"
        )