"""
Module : data_loader.py
Auteur : William BLENON & Véronèse Nikina ZINSOU
Date   : 2026
Projet : Maintenance Prédictive Industrielle – M2 Data Engineering EFREI

Description :
    Ce module contient la classe DataLoader, responsable du chargement
    et de l'inspection initiale du dataset industriel.
    Il constitue la première brique du pipeline de données.

Dépendances :
    - pandas
    - os
"""

import os
import pandas as pd


class DataLoader:
    """
    Classe responsable du chargement et de l'inspection du dataset industriel.

    Cette classe encapsule toute la logique d'acquisition des données brutes
    à partir d'un fichier CSV. Elle vérifie l'existence du fichier, charge
    les données en mémoire et expose des méthodes d'inspection utiles pour
    l'analyse exploratoire (EDA).

    Attributes:
        filepath (str): Chemin absolu ou relatif vers le fichier CSV.
        df (pd.DataFrame | None): DataFrame chargé en mémoire après appel à load().
                                  Vaut None avant le premier chargement.

    Example:
        >>> loader = DataLoader("data/industrial_machine_maintenance.csv")
        >>> df = loader.load()
        >>> info = loader.get_info()
        >>> print(info["shape"])
        (24042, 15)
    """

    # Colonnes attendues dans le dataset industriel
    # Colonnes attendues — validees apres inspection reelle du dataset
    EXPECTED_COLUMNS = [
        "timestamp",
        "machine_id",
        "machine_type",
        "vibration_rms",
        "temperature_motor",
        "current_phase_avg",
        "pressure_level",
        "rpm",
        "operating_mode",
        "hours_since_maintenance",
        "ambient_temp",
        "rul_hours",
        "failure_within_24h",
        "failure_type",
        "estimated_repair_cost",
    ]

    def __init__(self, filepath: str) -> None:
        """
        Initialise le DataLoader avec le chemin vers le fichier de données.

        Args:
            filepath (str): Chemin vers le fichier CSV du dataset industriel.

        Raises:
            TypeError: Si filepath n'est pas une chaîne de caractères.

        Example:
            >>> loader = DataLoader("data/industrial_machine_maintenance.csv")
        """
        if not isinstance(filepath, str):
            raise TypeError(
                f"Le chemin doit être une chaîne de caractères. "
                f"Reçu : {type(filepath).__name__}"
            )

        self.filepath: str = filepath
        self.df: pd.DataFrame | None = None  # Chargé uniquement après appel à load()

    def load(self) -> pd.DataFrame:
        """
        Charge le dataset CSV en mémoire sous forme de DataFrame pandas.

        Cette méthode vérifie d'abord l'existence du fichier, puis charge
        les données. Le résultat est stocké dans l'attribut self.df pour
        permettre des accès ultérieurs sans rechargement.

        Returns:
            pd.DataFrame: Le dataset complet chargé en mémoire.

        Raises:
            FileNotFoundError: Si le fichier spécifié dans filepath n'existe pas.
            ValueError: Si le fichier est vide ou ne contient aucune ligne de données.
            Exception: Pour toute autre erreur de lecture (encodage, format, etc.).

        Example:
            >>> loader = DataLoader("data/industrial_machine_maintenance.csv")
            >>> df = loader.load()
            >>> print(df.shape)
            (24042, 15)
        """
        # Vérification de l'existence du fichier
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(
                f"Le fichier de données est introuvable : '{self.filepath}'\n"
                f"Vérifiez le chemin et réessayez."
            )

        try:
            self.df = pd.read_csv(self.filepath)
        except Exception as e:
            raise Exception(
                f"Erreur lors de la lecture du fichier '{self.filepath}' : {e}"
            )

        # Vérification que le fichier n'est pas vide
        if self.df.empty:
            raise ValueError(
                f"Le fichier '{self.filepath}' est vide ou ne contient aucune donnée."
            )

        print(f"[DataLoader] Dataset chargé avec succès.")
        print(f"  → {self.df.shape[0]} enregistrements | {self.df.shape[1]} variables")

        return self.df

    def get_info(self) -> dict:
        """
        Retourne un dictionnaire synthétique d'informations sur le dataset chargé.

        Fournit une vue d'ensemble utile pour l'EDA : dimensions, types de données,
        valeurs manquantes, doublons et colonnes disponibles.

        Returns:
            dict: Dictionnaire contenant les clés suivantes :
                - "shape" (tuple): Dimensions (lignes, colonnes) du dataset.
                - "columns" (list): Liste des noms de colonnes.
                - "dtypes" (dict): Types de données par colonne.
                - "missing_values" (dict): Nombre de valeurs manquantes par colonne.
                - "missing_percent" (dict): Pourcentage de valeurs manquantes par colonne.
                - "duplicates" (int): Nombre de lignes dupliquées.
                - "memory_usage_mb" (float): Utilisation mémoire approximative en Mo.

        Raises:
            RuntimeError: Si load() n'a pas encore été appelé.

        Example:
            >>> loader = DataLoader("data/industrial_machine_maintenance.csv")
            >>> loader.load()
            >>> info = loader.get_info()
            >>> print(info["missing_values"])
            {'vibration_rms': 0, 'temperature_motor': 12, ...}
        """
        self._check_loaded()

        n_rows = self.df.shape[0]

        # Calcul des valeurs manquantes en nombre et en pourcentage
        missing_counts = self.df.isnull().sum().to_dict()
        missing_percent = {
            col: round((count / n_rows) * 100, 2)
            for col, count in missing_counts.items()
        }

        return {
            "shape": self.df.shape,
            "columns": list(self.df.columns),
            "dtypes": self.df.dtypes.astype(str).to_dict(),
            "missing_values": missing_counts,
            "missing_percent": missing_percent,
            "duplicates": int(self.df.duplicated().sum()),
            "memory_usage_mb": round(
                self.df.memory_usage(deep=True).sum() / (1024 ** 2), 2
            ),
        }

    def get_target_distribution(self, target_col: str = "failure_within_24h") -> dict:
        """
        Analyse la distribution de la variable cible pour détecter un déséquilibre de classes.

        Dans un contexte industriel, les pannes sont des événements rares.
        Un fort déséquilibre (ex. 95% classe 0 / 5% classe 1) nécessite des
        stratégies adaptées (stratified split, class_weight, seuil de décision).

        Args:
            target_col (str): Nom de la colonne cible à analyser.
                              Par défaut : "failure_within_24h".

        Returns:
            dict: Dictionnaire contenant :
                - "counts" (dict): Effectifs par classe.
                - "percentages" (dict): Pourcentages par classe (arrondis à 2 décimales).
                - "is_imbalanced" (bool): True si la classe minoritaire < 20%.

        Raises:
            RuntimeError: Si load() n'a pas encore été appelé.
            KeyError: Si target_col n'existe pas dans le dataset.

        Example:
            >>> loader.get_target_distribution("failure_within_24h")
            {'counts': {0: 22000, 1: 2042}, 'percentages': {0: 91.5, 1: 8.5},
             'is_imbalanced': True}
        """
        self._check_loaded()

        if target_col not in self.df.columns:
            raise KeyError(
                f"La colonne '{target_col}' n'existe pas dans le dataset.\n"
                f"Colonnes disponibles : {list(self.df.columns)}"
            )

        counts = self.df[target_col].value_counts().to_dict()
        total = sum(counts.values())
        percentages = {cls: round((n / total) * 100, 2) for cls, n in counts.items()}

        # Détection du déséquilibre : classe minoritaire < 20%
        min_percent = min(percentages.values())
        is_imbalanced = min_percent < 20.0

        if is_imbalanced:
            print(
                f"[DataLoader] ⚠️  Déséquilibre détecté sur '{target_col}' : "
                f"classe minoritaire = {min_percent}%.\n"
                f"  → Pensez à utiliser : stratified split, class_weight='balanced', "
                f"ajustement du seuil de décision."
            )

        return {
            "counts": counts,
            "percentages": percentages,
            "is_imbalanced": is_imbalanced,
        }

    def validate_columns(self) -> dict:
        """
        Vérifie la présence des colonnes clés attendues dans le dataset.

        Compare les colonnes du dataset chargé avec la liste des colonnes
        attendues (EXPECTED_COLUMNS). Utile pour détecter rapidement un
        problème de format ou de version du fichier source.

        Returns:
            dict: Dictionnaire contenant :
                - "present" (list): Colonnes attendues trouvées dans le dataset.
                - "missing" (list): Colonnes attendues absentes du dataset.
                - "extra" (list): Colonnes présentes dans le dataset mais non attendues.
                - "is_valid" (bool): True si toutes les colonnes attendues sont présentes.

        Raises:
            RuntimeError: Si load() n'a pas encore été appelé.

        Example:
            >>> result = loader.validate_columns()
            >>> if not result["is_valid"]:
            ...     print("Colonnes manquantes :", result["missing"])
        """
        self._check_loaded()

        actual_columns = set(self.df.columns)
        expected_columns = set(self.EXPECTED_COLUMNS)

        present = list(expected_columns & actual_columns)
        missing = list(expected_columns - actual_columns)
        extra = list(actual_columns - expected_columns)

        is_valid = len(missing) == 0

        if not is_valid:
            print(f"[DataLoader] ⚠️  Colonnes manquantes : {missing}")
        else:
            print(f"[DataLoader] ✅ Toutes les colonnes attendues sont présentes.")

        return {
            "present": present,
            "missing": missing,
            "extra": extra,
            "is_valid": is_valid,
        }

    def _check_loaded(self) -> None:
        """
        Méthode privée : vérifie que le dataset a bien été chargé avant toute opération.

        Cette méthode est appelée en début de chaque méthode publique nécessitant
        l'accès à self.df. Elle évite les erreurs AttributeError silencieuses.

        Raises:
            RuntimeError: Si self.df est None, c'est-à-dire si load() n'a pas
                          encore été appelé.
        """
        if self.df is None:
            raise RuntimeError(
                "Le dataset n'est pas encore chargé. "
                "Appelez d'abord la méthode load() avant toute autre opération."
            )

    def __repr__(self) -> str:
        """
        Représentation officielle de l'objet DataLoader.

        Returns:
            str: Représentation lisible indiquant le chemin et l'état du chargement.

        Example:
            >>> loader = DataLoader("data/industrial_machine_maintenance.csv")
            >>> print(repr(loader))
            DataLoader(filepath='data/industrial_machine_maintenance.csv', loaded=False)
        """
        loaded = self.df is not None
        return (
            f"DataLoader("
            f"filepath='{self.filepath}', "
            f"loaded={loaded}"
            f")"
        )