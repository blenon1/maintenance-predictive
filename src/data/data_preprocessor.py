"""
Module : data_preprocessor.py
Auteur : William BLENON & Véronèse Nikina ZINSOU
Date   : 2026
Projet : Maintenance Prédictive Industrielle – M2 Data Engineering EFREI

Description :
    Ce module contient la classe DataPreprocessor, responsable du nettoyage,
    de l'encodage, de la normalisation et du découpage train/test des données.

    Point critique : toutes les transformations (scaling, encodage) sont
    apprises UNIQUEMENT sur le train set, puis appliquées au test set.
    Cela garantit l'absence de data leakage.

Dépendances :
    - pandas
    - numpy
    - scikit-learn
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


class DataPreprocessor:
    """
    Classe responsable de la préparation complète des données pour la modélisation.

    Elle encapsule l'ensemble du pipeline de préparation :
        1. Suppression des colonnes non pertinentes,
        2. Traitement des valeurs manquantes,
        3. Encodage des variables catégorielles,
        4. Normalisation des variables numériques,
        5. Découpage stratifié train/test.

    Le pipeline sklearn est ajusté (fit) uniquement sur les données d'entraînement
    puis appliqué (transform) sur le test set, évitant tout data leakage.

    Attributes:
        target (str): Nom de la colonne cible (variable à prédire).
        test_size (float): Proportion du jeu de test (entre 0 et 1).
        random_state (int): Graine aléatoire pour la reproductibilité.
        pipeline (Pipeline | None): Pipeline sklearn ajusté après fit_transform().
        feature_names (list | None): Noms des features après transformation.
        label_encoder (LabelEncoder | None): Encodeur de la variable cible si catégorielle.

    Example:
        >>> preprocessor = DataPreprocessor(target="failure_within_24h")
        >>> X_train, X_test, y_train, y_test = preprocessor.fit_transform(df)
        >>> X_new = preprocessor.transform(new_df)
    """

    # Colonnes à exclure du dataset (identifiants, cibles alternatives, fuites de données)
    # Colonnes a exclure — mis a jour apres inspection reelle du dataset
    # (24 042 obs. x 15 variables, dont 3 cibles alternatives et 2 identifiants)
    COLUMNS_TO_DROP = [
        "timestamp",              # Horodatage brut — non utilise comme feature numerique
        "machine_id",             # Identifiant unique — sans valeur predictive
        "failure_type",           # Cible alternative : classification multi-classe
        "rul_hours",              # Cible alternative : regression RUL
        "estimated_repair_cost",  # Cible alternative : regression cout de reparation
    ]

    def __init__(
        self,
        target: str = "failure_within_24h",
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> None:
        """
        Initialise le DataPreprocessor avec les paramètres du pipeline.

        Args:
            target (str): Nom de la colonne cible. Par défaut : "failure_within_24h".
            test_size (float): Proportion du jeu de test. Par défaut : 0.2 (20%).
            random_state (int): Graine aléatoire pour la reproductibilité. Par défaut : 42.

        Raises:
            ValueError: Si test_size n'est pas dans l'intervalle ]0, 1[.
            TypeError: Si target n'est pas une chaîne de caractères.

        Example:
            >>> preprocessor = DataPreprocessor(
            ...     target="failure_within_24h",
            ...     test_size=0.2,
            ...     random_state=42
            ... )
        """
        if not isinstance(target, str):
            raise TypeError(
                f"'target' doit être une chaîne de caractères. Reçu : {type(target).__name__}"
            )

        if not (0 < test_size < 1):
            raise ValueError(
                f"'test_size' doit être compris entre 0 et 1 (exclus). Reçu : {test_size}"
            )

        self.target: str = target
        self.test_size: float = test_size
        self.random_state: int = random_state

        # Attributs initialisés après fit_transform()
        self.pipeline: Pipeline | None = None
        self.feature_names: list | None = None
        self.label_encoder: LabelEncoder | None = None

    def fit_transform(
        self, df: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Applique le pipeline complet de préparation et découpe les données en train/test.

        Étapes réalisées dans l'ordre :
            1. Suppression des colonnes non pertinentes,
            2. Séparation features / cible,
            3. Découpage stratifié train/test,
            4. Construction et ajustement du pipeline sur le train set uniquement,
            5. Transformation du test set avec le pipeline ajusté.

        Args:
            df (pd.DataFrame): Le DataFrame brut chargé par DataLoader.

        Returns:
            tuple: (X_train, X_test, y_train, y_test) où :
                - X_train (np.ndarray): Features d'entraînement transformées.
                - X_test (np.ndarray): Features de test transformées.
                - y_train (np.ndarray): Labels d'entraînement.
                - y_test (np.ndarray): Labels de test.

        Raises:
            KeyError: Si la colonne cible n'est pas présente dans le DataFrame.
            ValueError: Si le DataFrame est vide après nettoyage.

        Example:
            >>> preprocessor = DataPreprocessor(target="failure_within_24h")
            >>> X_train, X_test, y_train, y_test = preprocessor.fit_transform(df)
            >>> print(X_train.shape)
            (19233, 12)
        """
        if self.target not in df.columns:
            raise KeyError(
                f"La colonne cible '{self.target}' est absente du DataFrame.\n"
                f"Colonnes disponibles : {list(df.columns)}"
            )

        # --- Étape 1 : Copie défensive et suppression des colonnes non pertinentes ---
        df_clean = df.copy()
        cols_to_drop = [c for c in self.COLUMNS_TO_DROP if c in df_clean.columns]
        df_clean.drop(columns=cols_to_drop, inplace=True)

        # --- Étape 2 : Séparation features / cible ---
        X = df_clean.drop(columns=[self.target])
        y = df_clean[self.target]

        if X.empty:
            raise ValueError("Le DataFrame est vide après suppression des colonnes.")

        # Encodage de la cible si elle est catégorielle (cas classification multi-classe)
        if y.dtype == "object":
            self.label_encoder = LabelEncoder()
            y = self.label_encoder.fit_transform(y)
        else:
            y = y.values

        # --- Étape 3 : Découpage stratifié train/test ---
        # Stratifié pour conserver la proportion des classes (important si déséquilibre)
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y,
        )

        print(
            f"[DataPreprocessor] Découpage train/test effectué :\n"
            f"  → Train : {X_train.shape[0]} lignes | Test : {X_test.shape[0]} lignes"
        )

        # --- Étape 4 : Construction du pipeline et ajustement sur le TRAIN uniquement ---
        self.pipeline = self._build_pipeline(X_train)
        self.pipeline.fit(X_train)

        # Récupération des noms de features après transformation
        self.feature_names = self._get_feature_names(X_train)

        # --- Étape 5 : Transformation du train et du test ---
        X_train_transformed = self.pipeline.transform(X_train)
        X_test_transformed = self.pipeline.transform(X_test)

        print(
            f"[DataPreprocessor] Pipeline ajusté et transformations appliquées.\n"
            f"  → Dimensions X_train : {X_train_transformed.shape}"
        )

        return X_train_transformed, X_test_transformed, y_train, y_test

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Applique le pipeline déjà ajusté à de nouvelles données (inférence).

        Utilisée par le dashboard ou l'API pour transformer les données saisies
        par l'utilisateur avant de les soumettre au modèle.

        Args:
            df (pd.DataFrame): Nouvelles données à transformer (même format que le dataset).

        Returns:
            np.ndarray: Données transformées prêtes pour la prédiction.

        Raises:
            RuntimeError: Si fit_transform() n'a pas encore été appelé.

        Example:
            >>> new_data = pd.DataFrame({...})
            >>> X_new = preprocessor.transform(new_data)
            >>> prediction = model.predict(X_new)
        """
        self._check_fitted()

        # Suppression des colonnes non pertinentes (si présentes)
        df_clean = df.copy()
        cols_to_drop = [
            c for c in self.COLUMNS_TO_DROP + [self.target]
            if c in df_clean.columns
        ]
        df_clean.drop(columns=cols_to_drop, inplace=True)

        return self.pipeline.transform(df_clean)

    def _build_pipeline(self, X: pd.DataFrame) -> Pipeline:
        """
        Méthode privée : construit le pipeline sklearn adapté aux types de colonnes.

        Détecte automatiquement les colonnes numériques et catégorielles,
        puis construit un ColumnTransformer qui applique :
            - StandardScaler sur les colonnes numériques,
            - OneHotEncoder sur les colonnes catégorielles.

        Args:
            X (pd.DataFrame): DataFrame des features (train set) utilisé pour
                               identifier les types de colonnes.

        Returns:
            Pipeline: Pipeline sklearn prêt à être ajusté.
        """
        # Identification automatique des types de colonnes
        numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
        categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

        print(
            f"[DataPreprocessor] Colonnes détectées :\n"
            f"  → Numériques ({len(numeric_cols)}) : {numeric_cols}\n"
            f"  → Catégorielles ({len(categorical_cols)}) : {categorical_cols}"
        )

        # Transformations numériques :
        #   1. Imputation par médiane (robuste aux outliers)
        #   2. Standardisation (moyenne=0, écart-type=1)
        # L'EDA a révélé des NaN sur : vibration_rms (4.16%),
        # pressure_level (3.84%), temperature_motor (3.47%),
        # current_phase_avg (3.04%), rpm (2.22%) — tous < 5%
        numeric_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler()),
        ])

        # Transformations catégorielles :
        #   1. Imputation par la valeur la plus fréquente (mode)
        #   2. Encodage one-hot
        # handle_unknown='ignore' : évite les erreurs si une valeur inconnue
        # apparaît au moment de l'inférence (dashboard / API)
        categorical_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])

        # Assemblage du ColumnTransformer
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_cols),
                ("cat", categorical_transformer, categorical_cols),
            ],
            remainder="drop",  # Supprime les colonnes non spécifiées
        )

        return Pipeline(steps=[("preprocessor", preprocessor)])

    def _get_feature_names(self, X: pd.DataFrame) -> list:
        """
        Méthode privée : récupère les noms des features après transformation.

        Nécessaire pour l'interprétabilité (feature importance, SHAP).

        Args:
            X (pd.DataFrame): DataFrame des features d'entraînement.

        Returns:
            list: Liste des noms de features après transformation (numériques
                  + colonnes one-hot encodées).
        """
        column_transformer = self.pipeline.named_steps["preprocessor"]

        feature_names = []
        for name, transformer, cols in column_transformer.transformers_:
            if name == "num":
                feature_names.extend(cols)
            elif name == "cat":
                ohe_names = transformer.get_feature_names_out(cols).tolist()
                feature_names.extend(ohe_names)

        return feature_names

    def _check_fitted(self) -> None:
        """
        Méthode privée : vérifie que le pipeline a bien été ajusté avant toute transformation.

        Raises:
            RuntimeError: Si fit_transform() n'a pas encore été appelé.
        """
        if self.pipeline is None:
            raise RuntimeError(
                "Le pipeline n'a pas encore été ajusté. "
                "Appelez d'abord fit_transform() avant d'utiliser transform()."
            )

    def __repr__(self) -> str:
        """
        Représentation officielle de l'objet DataPreprocessor.

        Returns:
            str: Représentation lisible indiquant la cible et l'état du pipeline.

        Example:
            >>> preprocessor = DataPreprocessor(target="failure_within_24h")
            >>> print(repr(preprocessor))
            DataPreprocessor(target='failure_within_24h', test_size=0.2, fitted=False)
        """
        fitted = self.pipeline is not None
        return (
            f"DataPreprocessor("
            f"target='{self.target}', "
            f"test_size={self.test_size}, "
            f"fitted={fitted}"
            f")"
        )