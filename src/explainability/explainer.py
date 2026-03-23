"""
Module : explainer.py
Auteur : William BLENON & Véronèse Nikina ZINSOU
Date   : 2026
Projet : Maintenance Prédictive Industrielle – M2 Data Engineering EFREI

Description :
    Ce module contient la classe Explainer, responsable de l'interprétabilité
    des modèles de Machine Learning et Deep Learning.

    Dans un contexte industriel, un modèle performant ne suffit pas. Un
    responsable maintenance doit pouvoir répondre à : "Pourquoi le modèle
    prédit une panne ? Quels capteurs ont déclenché l'alerte ?"

    Trois niveaux d'explicabilité sont implémentés :

        1. Feature Importance native (modèles à base d'arbres uniquement)
           → Vision globale rapide, basée sur la réduction d'impureté Gini.

        2. Permutation Importance (tous modèles, recommandée)
           → Agnostique au modèle. Mesure la chute de performance quand
             une variable est permutée aléatoirement.

        3. SHAP – SHapley Additive exPlanations (niveau avancé)
           → Explication locale (pourquoi CETTE prédiction ?) et globale.
           → Basée sur la théorie des jeux coopératifs.
           → Référence en entreprise pour l'explicabilité IA.

    Quand appliquer ces techniques ?
        → Après l'entraînement, sur le modèle final sélectionné.
        → Sans modèle entraîné, il n'y a rien à expliquer.

Dépendances :
    - shap
    - scikit-learn
    - matplotlib
    - pandas
    - numpy
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.inspection import permutation_importance

from src.models.base_model import BaseModel


class Explainer:
    """
    Classe responsable de l'interprétabilité des modèles de maintenance prédictive.

    Fournit trois niveaux d'analyse :
        - Feature Importance native (arbres uniquement),
        - Permutation Importance (tous modèles),
        - SHAP values (local + global, modèle final).

    Attributes:
        model (BaseModel): Le modèle entraîné à expliquer.
        X_train (np.ndarray): Données d'entraînement (utilisées par SHAP pour
                               construire l'explainer).
        X_test (np.ndarray): Données de test (utilisées pour les explications locales).
        feature_names (list[str]): Noms des features après transformation.
        shap_values (np.ndarray | None): Valeurs SHAP calculées, disponibles
                                         après appel à compute_shap_values().

    Example:
        >>> explainer = Explainer(
        ...     model=best_model,
        ...     X_train=X_train,
        ...     X_test=X_test,
        ...     feature_names=preprocessor.feature_names
        ... )
        >>> explainer.plot_permutation_importance()
        >>> explainer.compute_shap_values()
        >>> explainer.plot_shap_summary()
        >>> explainer.plot_shap_waterfall(sample_index=0)
    """

    def __init__(
        self,
        model: BaseModel,
        X_train: np.ndarray,
        X_test: np.ndarray,
        feature_names: list[str],
        y_test: np.ndarray | None = None,
    ) -> None:
        """
        Initialise l'Explainer avec le modèle et les données nécessaires.

        Args:
            model (BaseModel): Modèle entraîné à expliquer. Doit avoir
                               is_trained=True.
            X_train (np.ndarray): Features d'entraînement (référence pour SHAP).
            X_test (np.ndarray): Features de test (pour les explications locales).
            feature_names (list[str]): Noms des features après preprocessing.
                                       Récupérables via preprocessor.feature_names.
            y_test (np.ndarray | None): Labels réels du test (requis pour
                                         Permutation Importance). Optionnel.

        Raises:
            RuntimeError: Si le modèle n'a pas encore été entraîné.
            ValueError: Si feature_names est vide ou incompatible avec X_train.

        Example:
            >>> explainer = Explainer(
            ...     model=rf_model,
            ...     X_train=X_train,
            ...     X_test=X_test,
            ...     feature_names=preprocessor.feature_names,
            ...     y_test=y_test
            ... )
        """
        if not model.is_trained:
            raise RuntimeError(
                f"Le modèle '{model.name}' n'a pas encore été entraîné. "
                f"Appelez train() avant d'instancier Explainer."
            )

        if not feature_names:
            raise ValueError("'feature_names' ne peut pas être vide.")

        if len(feature_names) != X_train.shape[1]:
            raise ValueError(
                f"Incohérence : {len(feature_names)} feature_names pour "
                f"{X_train.shape[1]} colonnes dans X_train."
            )

        self.model: BaseModel = model
        self.X_train: np.ndarray = X_train
        self.X_test: np.ndarray = X_test
        self.feature_names: list[str] = feature_names
        self.y_test: np.ndarray | None = y_test

        # Initialisé après compute_shap_values()
        self.shap_values: np.ndarray | None = None
        self._shap_explainer = None

        print(
            f"[Explainer] Initialisé pour le modèle : '{model.name}'\n"
            f"  → {len(feature_names)} features | "
            f"{X_train.shape[0]} samples train | "
            f"{X_test.shape[0]} samples test"
        )

    # ──────────────────────────────────────────────────────────────
    # 1. Feature Importance Native (arbres uniquement)
    # ──────────────────────────────────────────────────────────────

    def plot_native_importance(self, top_n: int = 15) -> None:
        """
        Affiche l'importance native des features (modèles à base d'arbres uniquement).

        Méthode applicable à : Random Forest, Gradient Boosting, XGBoost.
        Non applicable à : Régression Logistique, MLP.

        L'importance est calculée par réduction moyenne d'impureté (Gini) :
        plus une feature divise les données efficacement, plus elle est importante.

        Limitation connue : cette méthode favorise les variables à haute cardinalité
        (nombreuses valeurs uniques). La Permutation Importance est plus fiable.

        Args:
            top_n (int): Nombre de features à afficher. Par défaut : 15.

        Raises:
            AttributeError: Si le modèle sous-jacent ne dispose pas de
                            feature_importances_ (ex. LogisticRegression, MLP).

        Example:
            >>> explainer.plot_native_importance(top_n=10)
        """
        if not hasattr(self.model.model, "feature_importances_"):
            raise AttributeError(
                f"Le modèle '{self.model.name}' ne dispose pas de feature_importances_.\n"
                f"Utilisez plot_permutation_importance() à la place."
            )

        importances = self.model.model.feature_importances_

        # Tri décroissant et sélection du top_n
        df_imp = pd.DataFrame({
            "Feature": self.feature_names,
            "Importance": importances,
        }).sort_values("Importance", ascending=False).head(top_n)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(
            df_imp["Feature"][::-1],
            df_imp["Importance"][::-1],
            color="steelblue",
            edgecolor="white",
        )
        ax.set_xlabel("Importance (réduction d'impureté Gini)")
        ax.set_title(
            f"Feature Importance Native – {self.model.name}\n"
            f"(Top {top_n} features)"
        )
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()

        print(
            f"[Explainer] Top 3 features (importance native) :\n"
            + "\n".join(
                f"  {i+1}. {row['Feature']} : {row['Importance']:.4f}"
                for i, (_, row) in enumerate(df_imp.head(3).iterrows())
            )
        )

    # ──────────────────────────────────────────────────────────────
    # 2. Permutation Importance (tous modèles — recommandée)
    # ──────────────────────────────────────────────────────────────

    def plot_permutation_importance(
        self,
        top_n: int = 15,
        n_repeats: int = 10,
        scoring: str = "f1",
    ) -> pd.DataFrame:
        """
        Calcule et affiche la Permutation Importance sur le jeu de test.

        Principe :
            1. Mesurer la performance initiale du modèle (F1, ROC-AUC, etc.),
            2. Permuter aléatoirement les valeurs d'une feature,
            3. Recalculer la performance — la chute indique l'importance.

        Avantages par rapport à la Feature Importance native :
            - Agnostique au modèle (fonctionne avec MLP, Logistic Regression, etc.),
            - Moins biaisée par la cardinalité des variables,
            - Prend en compte la performance réelle sur le test set.

        Args:
            top_n (int): Nombre de features à afficher. Par défaut : 15.
            n_repeats (int): Nombre de permutations par feature (plus élevé = plus stable).
                             Par défaut : 10.
            scoring (str): Métrique d'évaluation. Par défaut : "f1".
                           Valeurs possibles : "f1", "roc_auc", "accuracy", "recall".

        Returns:
            pd.DataFrame: DataFrame avec les colonnes "Feature", "Importance_Mean",
                          "Importance_Std", trié par importance décroissante.

        Raises:
            RuntimeError: Si y_test n'a pas été fourni à l'initialisation.

        Example:
            >>> df_perm = explainer.plot_permutation_importance(top_n=10, scoring="recall")
            >>> print(df_perm.head())
        """
        if self.y_test is None:
            raise RuntimeError(
                "y_test est requis pour la Permutation Importance. "
                "Fournissez-le à l'initialisation de l'Explainer."
            )

        print(
            f"[Explainer] Calcul de la Permutation Importance "
            f"({n_repeats} répétitions, scoring='{scoring}')..."
        )

        result = permutation_importance(
            self.model.model,
            self.X_test,
            self.y_test,
            n_repeats=n_repeats,
            scoring=scoring,
            random_state=42,
            n_jobs=-1,
        )

        df_perm = pd.DataFrame({
            "Feature": self.feature_names,
            "Importance_Mean": result.importances_mean,
            "Importance_Std": result.importances_std,
        }).sort_values("Importance_Mean", ascending=False).head(top_n)

        # Visualisation avec barres d'erreur (écart-type)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(
            df_perm["Feature"][::-1],
            df_perm["Importance_Mean"][::-1],
            xerr=df_perm["Importance_Std"][::-1],
            color="darkorange",
            edgecolor="white",
            capsize=4,
        )
        ax.set_xlabel(f"Chute de {scoring} lors de la permutation")
        ax.set_title(
            f"Permutation Importance – {self.model.name}\n"
            f"(Top {top_n} features | {n_repeats} répétitions)"
        )
        ax.axvline(x=0, color="black", linewidth=0.8, linestyle="--")
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()

        print(
            f"[Explainer] Top 3 features (permutation) :\n"
            + "\n".join(
                f"  {i+1}. {row['Feature']} : {row['Importance_Mean']:.4f} "
                f"(±{row['Importance_Std']:.4f})"
                for i, (_, row) in enumerate(df_perm.head(3).iterrows())
            )
        )

        return df_perm

    # ──────────────────────────────────────────────────────────────
    # 3. SHAP – SHapley Additive exPlanations
    # ──────────────────────────────────────────────────────────────

    def compute_shap_values(self, max_samples: int = 200) -> None:
        """
        Calcule les valeurs SHAP pour le modèle sélectionné.

        SHAP est basé sur la théorie des jeux coopératifs de Shapley (1953).
        Chaque feature reçoit une contribution marginale à la prédiction,
        garantissant efficience, symétrie et consistance.

        Types d'explainers utilisés selon le modèle :
            - TreeExplainer  : pour Random Forest, XGBoost, LightGBM (rapide, exact).
            - KernelExplainer: pour MLP, Logistic Regression (approximation, plus lent).

        Args:
            max_samples (int): Nombre maximum de samples du test utilisés pour
                               le calcul SHAP (KernelExplainer uniquement, pour
                               limiter le temps de calcul). Par défaut : 200.

        Note:
            Cette méthode doit être appelée avant toute visualisation SHAP
            (plot_shap_summary, plot_shap_waterfall, plot_shap_dependence).

        Example:
            >>> explainer.compute_shap_values(max_samples=100)
        """
        print(f"[Explainer] Calcul des valeurs SHAP pour '{self.model.name}'...")

        model_type = type(self.model.model).__name__

        # Sélection de l'explainer adapté au type de modèle
        tree_based_models = [
            "RandomForestClassifier",
            "XGBClassifier",
            "GradientBoostingClassifier",
            "LGBMClassifier",
            "DecisionTreeClassifier",
            "ExtraTreesClassifier",
        ]

        if model_type in tree_based_models:
            # TreeExplainer : rapide et exact pour les modèles à base d'arbres
            print(f"  → Utilisation de TreeExplainer (modèle arbre détecté)")
            self._shap_explainer = shap.TreeExplainer(self.model.model)
            self.shap_values = self._shap_explainer.shap_values(self.X_test)

            # Pour la classification binaire, TreeExplainer retourne une liste [classe_0, classe_1]
            # On garde uniquement les SHAP values de la classe 1 (panne)
            if isinstance(self.shap_values, list):
                self.shap_values = self.shap_values[1]

        else:
            # KernelExplainer : agnostique au modèle, mais plus lent
            # On utilise un résumé du train set comme background (kmeans pour accélérer)
            print(
                f"  → Utilisation de KernelExplainer (modèle non-arbre détecté)\n"
                f"  → Calcul limité à {max_samples} samples du test (performance)"
            )
            background = shap.kmeans(self.X_train, k=50)
            self._shap_explainer = shap.KernelExplainer(
                self.model.model.predict_proba,
                background,
            )
            X_sample = self.X_test[:max_samples]
            # nsamples='auto' adapte le nombre de perturbations au nombre de features
            self.shap_values = self._shap_explainer.shap_values(
                X_sample, nsamples="auto"
            )
            # Garde uniquement la classe 1 (probabilité de panne)
            if isinstance(self.shap_values, list):
                self.shap_values = self.shap_values[1]

        print(
            f"[Explainer] ✅ SHAP values calculées.\n"
            f"  → Shape : {np.array(self.shap_values).shape}"
        )

    def plot_shap_summary(self) -> None:
        """
        Affiche le SHAP Summary Plot (importance globale + direction d'impact).

        Le Summary Plot combine :
            - L'importance globale des features (axe X = valeur SHAP moyenne),
            - La direction de l'impact (rouge = valeur élevée de la feature,
              bleu = valeur faible),
            - La densité des observations.

        Lecture industrielle :
            - "vibration_rms élevée → SHAP positif → augmente le risque de panne"
            - "temperature_motor faible → SHAP négatif → réduit le risque de panne"

        Raises:
            RuntimeError: Si compute_shap_values() n'a pas encore été appelé.

        Example:
            >>> explainer.compute_shap_values()
            >>> explainer.plot_shap_summary()
        """
        self._check_shap_computed()

        print("[Explainer] Génération du SHAP Summary Plot...")

        # Reconstruction du DataFrame pour SHAP (nécessite les noms de colonnes)
        X_display = pd.DataFrame(
            self.X_test[:len(self.shap_values)],
            columns=self.feature_names,
        )

        plt.figure(figsize=(10, 7))
        shap.summary_plot(
            self.shap_values,
            X_display,
            plot_type="dot",
            show=False,
        )
        plt.title(
            f"SHAP Summary Plot – {self.model.name}\n"
            f"(Rouge = valeur élevée | Bleu = valeur faible | Axe X = impact sur la prédiction)"
        )
        plt.tight_layout()

    def plot_shap_bar(self, top_n: int = 15) -> None:
        """
        Affiche le SHAP Bar Plot : importance globale moyenne des features.

        Représentation simplifiée du Summary Plot : montre la moyenne des
        valeurs SHAP absolues par feature, sans information directionnelle.
        Plus lisible pour une présentation à un public non technique.

        Args:
            top_n (int): Nombre de features à afficher. Par défaut : 15.

        Raises:
            RuntimeError: Si compute_shap_values() n'a pas encore été appelé.

        Example:
            >>> explainer.plot_shap_bar(top_n=10)
        """
        self._check_shap_computed()

        X_display = pd.DataFrame(
            self.X_test[:len(self.shap_values)],
            columns=self.feature_names,
        )

        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            self.shap_values,
            X_display,
            plot_type="bar",
            max_display=top_n,
            show=False,
        )
        plt.title(f"SHAP Feature Importance (globale) – {self.model.name}")
        plt.tight_layout()

    def plot_shap_waterfall(self, sample_index: int = 0) -> None:
        """
        Affiche le SHAP Waterfall Plot pour une observation spécifique (explication locale).

        Le Waterfall Plot répond à la question :
        "Pourquoi le modèle a-t-il prédit une panne pour CETTE machine précise ?"

        Lecture :
            - Barre rouge : feature qui augmente le risque de panne,
            - Barre bleue : feature qui réduit le risque de panne,
            - E[f(X)] : valeur de base (prédiction moyenne sur l'ensemble),
            - f(x) : prédiction finale pour cet échantillon.

        Args:
            sample_index (int): Indice de l'observation dans X_test à expliquer.
                                Par défaut : 0 (première observation).

        Raises:
            RuntimeError: Si compute_shap_values() n'a pas encore été appelé.
            IndexError: Si sample_index dépasse le nombre d'observations disponibles.

        Example:
            >>> # Expliquer pourquoi la machine #42 a une haute probabilité de panne
            >>> explainer.plot_shap_waterfall(sample_index=42)
        """
        self._check_shap_computed()

        n_available = len(self.shap_values)
        if sample_index >= n_available:
            raise IndexError(
                f"sample_index={sample_index} dépasse le nombre d'observations "
                f"disponibles ({n_available}). Choisissez un index entre 0 et {n_available - 1}."
            )

        print(
            f"[Explainer] SHAP Waterfall Plot – Observation #{sample_index} "
            f"(modèle : {self.model.name})"
        )

        # Construction de l'objet Explanation SHAP
        expected_value = (
            self._shap_explainer.expected_value[1]
            if isinstance(self._shap_explainer.expected_value, (list, np.ndarray))
            else self._shap_explainer.expected_value
        )

        explanation = shap.Explanation(
            values=self.shap_values[sample_index],
            base_values=expected_value,
            data=self.X_test[sample_index],
            feature_names=self.feature_names,
        )

        plt.figure(figsize=(10, 6))
        shap.plots.waterfall(explanation, show=False)
        plt.title(
            f"SHAP Waterfall – Observation #{sample_index} | {self.model.name}\n"
            f"(Rouge = augmente le risque | Bleu = réduit le risque)"
        )
        plt.tight_layout()

    def plot_shap_dependence(
        self,
        feature_name: str,
        interaction_feature: str = "auto",
    ) -> None:
        """
        Affiche le SHAP Dependence Plot pour une feature donnée.

        Montre la relation entre la valeur d'une feature et son impact SHAP,
        coloré par une feature d'interaction (détectée automatiquement ou
        spécifiée manuellement).

        Utile pour identifier des effets non linéaires ou des interactions
        entre capteurs industriels (ex. vibration × température).

        Args:
            feature_name (str): Nom de la feature principale à analyser.
                                Doit figurer dans feature_names.
            interaction_feature (str): Feature de coloration (interaction).
                                       "auto" = détection automatique par SHAP.
                                       Par défaut : "auto".

        Raises:
            RuntimeError: Si compute_shap_values() n'a pas encore été appelé.
            ValueError: Si feature_name n'est pas dans feature_names.

        Example:
            >>> explainer.plot_shap_dependence(
            ...     feature_name="vibration_rms",
            ...     interaction_feature="temperature_motor"
            ... )
        """
        self._check_shap_computed()

        if feature_name not in self.feature_names:
            raise ValueError(
                f"Feature '{feature_name}' introuvable.\n"
                f"Features disponibles : {self.feature_names}"
            )

        X_display = pd.DataFrame(
            self.X_test[:len(self.shap_values)],
            columns=self.feature_names,
        )

        plt.figure(figsize=(9, 6))
        shap.dependence_plot(
            feature_name,
            self.shap_values,
            X_display,
            interaction_index=interaction_feature,
            show=False,
        )
        plt.title(
            f"SHAP Dependence Plot – '{feature_name}' | {self.model.name}"
        )
        plt.tight_layout()

    def get_top_features(self, top_n: int = 10) -> pd.DataFrame:
        """
        Retourne un DataFrame des features les plus importantes selon SHAP.

        Utile pour le dashboard : afficher les capteurs industriels les plus
        déterminants dans la détection de pannes, sous forme de tableau.

        Args:
            top_n (int): Nombre de features à retourner. Par défaut : 10.

        Returns:
            pd.DataFrame: DataFrame avec les colonnes :
                - "Feature" : nom de la feature,
                - "SHAP_Mean_Abs" : importance SHAP moyenne absolue (arrondie à 4 décimales),
                - "Rank" : rang d'importance (1 = plus importante).

        Raises:
            RuntimeError: Si compute_shap_values() n'a pas encore été appelé.

        Example:
            >>> top_features = explainer.get_top_features(top_n=5)
            >>> print(top_features)
               Rank           Feature  SHAP_Mean_Abs
            0     1     vibration_rms         0.1823
            1     2  temperature_motor         0.1245
            ...
        """
        self._check_shap_computed()

        mean_abs_shap = np.abs(self.shap_values).mean(axis=0)

        df = pd.DataFrame({
            "Feature": self.feature_names,
            "SHAP_Mean_Abs": np.round(mean_abs_shap, 4),
        }).sort_values("SHAP_Mean_Abs", ascending=False).head(top_n)

        df.insert(0, "Rank", range(1, len(df) + 1))
        df.reset_index(drop=True, inplace=True)

        return df

    # ──────────────────────────────────────────────────────────────
    # Méthodes privées
    # ──────────────────────────────────────────────────────────────

    def _check_shap_computed(self) -> None:
        """
        Méthode privée : vérifie que les valeurs SHAP ont bien été calculées.

        Raises:
            RuntimeError: Si compute_shap_values() n'a pas encore été appelé.
        """
        if self.shap_values is None:
            raise RuntimeError(
                "Les valeurs SHAP n'ont pas encore été calculées. "
                "Appelez d'abord compute_shap_values()."
            )

    def __repr__(self) -> str:
        """
        Représentation officielle de l'objet Explainer.

        Returns:
            str: Représentation lisible indiquant le modèle et l'état SHAP.

        Example:
            >>> print(repr(explainer))
            Explainer(model='Random Forest', n_features=12, shap_computed=False)
        """
        return (
            f"Explainer("
            f"model='{self.model.name}', "
            f"n_features={len(self.feature_names)}, "
            f"shap_computed={self.shap_values is not None}"
            f")"
        )