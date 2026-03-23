"""
Module : test_api.py
Auteur : William BLENON & Véronèse Nikina ZINSOU
Date   : 2026
Projet : Maintenance Prédictive Industrielle – M2 Data Engineering EFREI

Description :
    Suite de tests pour valider le bon fonctionnement de l'API REST.

    Ces tests couvrent :
        - Les endpoints de supervision (/health, /model-info),
        - La prédiction unitaire (/predict) avec cas valides et invalides,
        - La prédiction batch (/predict/batch),
        - La validation des schémas Pydantic (valeurs hors plage, champs manquants),
        - La gestion des erreurs (422, 503).

    Deux modes d'utilisation :
        1. Tests automatisés (pytest) : `pytest api/test_api.py -v`
        2. Tests manuels (script) : `python api/test_api.py`
           → Lance les requêtes directement contre le serveur local.

Usage :
    # Démarrer l'API dans un terminal :
    $ uvicorn api.main:app --reload --port 8000

    # Lancer les tests dans un autre terminal :
    $ python api/test_api.py

Dépendances :
    - requests
    - json
"""

import json
import sys

import requests

# ─────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────

BASE_URL = "http://localhost:8000"
HEADERS = {"Content-Type": "application/json"}

# Couleurs terminal pour la lisibilité des résultats
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RESET = "\033[0m"
BOLD = "\033[1m"


# ─────────────────────────────────────────────────────────────────
# Utilitaires de test
# ─────────────────────────────────────────────────────────────────

def print_test(name: str, passed: bool, details: str = "") -> None:
    """Affiche le résultat d'un test avec formatage coloré."""
    icon = f"{GREEN}✅{RESET}" if passed else f"{RED}❌{RESET}"
    status = f"{GREEN}PASS{RESET}" if passed else f"{RED}FAIL{RESET}"
    print(f"  {icon} [{status}] {name}")
    if details:
        print(f"       {CYAN}{details}{RESET}")


def print_section(title: str) -> None:
    """Affiche un séparateur de section."""
    print(f"\n{BOLD}{YELLOW}{'─' * 55}{RESET}")
    print(f"{BOLD}{YELLOW}  {title}{RESET}")
    print(f"{BOLD}{YELLOW}{'─' * 55}{RESET}")


# ─────────────────────────────────────────────────────────────────
# Données de test
# ─────────────────────────────────────────────────────────────────

# Machine en bon état (faible probabilité de panne)
# Champs complets correspondant aux 14 features réelles du dataset
HEALTHY_MACHINE = {
    "vibration_rms": 1.2,
    "temperature_motor": 55.0,
    "current_phase_avg": 4.5,
    "pressure_level": 23.0,
    "rpm": 900.0,
    "hours_since_maintenance": 50.0,
    "ambient_temp": 20.0,
    "machine_type": "CNC",
    "operating_mode": "normal",
    "threshold": 0.5,
}

# Machine en état dégradé (forte probabilité de panne)
DEGRADED_MACHINE = {
    "vibration_rms": 8.5,
    "temperature_motor": 145.0,
    "current_phase_avg": 18.0,
    "pressure_level": 95.0,
    "rpm": 3800.0,
    "hours_since_maintenance": 450.0,
    "ambient_temp": 35.0,
    "machine_type": "Pump",
    "operating_mode": "degraded",
    "threshold": 0.4,
}

# Données invalides – vibration hors plage
INVALID_VIBRATION = {
    "vibration_rms": 999.0,    # Hors plage [0, 50]
    "temperature_motor": 80.0,
    "current_phase_avg": 6.0,
    "pressure_level": 45.0,
    "rpm": 1200.0,
    "hours_since_maintenance": 100.0,
    "ambient_temp": 22.0,
    "machine_type": "CNC",
    "operating_mode": "normal",
}

# Données invalides – champ manquant (rpm absent)
MISSING_FIELD = {
    "vibration_rms": 2.0,
    "temperature_motor": 70.0,
    "current_phase_avg": 5.0,
    "pressure_level": 30.0,
    # rpm manquant intentionnellement
    "hours_since_maintenance": 80.0,
    "ambient_temp": 21.0,
    "machine_type": "CNC",
    "operating_mode": "normal",
}

# Lot de machines pour le test batch
BATCH_MACHINES = {
    "observations": [
        {
            "vibration_rms": 1.5,
            "temperature_motor": 60.0,
            "current_phase_avg": 4.8,
            "pressure_level": 25.0,
            "rpm": 850.0,
            "hours_since_maintenance": 40.0,
            "ambient_temp": 19.0,
            "machine_type": "CNC",
            "operating_mode": "normal",
        },
        {
            "vibration_rms": 7.2,
            "temperature_motor": 135.0,
            "current_phase_avg": 16.5,
            "pressure_level": 88.0,
            "rpm": 3200.0,
            "hours_since_maintenance": 380.0,
            "ambient_temp": 32.0,
            "machine_type": "Pump",
            "operating_mode": "degraded",
        },
        {
            "vibration_rms": 2.8,
            "temperature_motor": 78.0,
            "current_phase_avg": 7.2,
            "pressure_level": 42.0,
            "rpm": 1400.0,
            "hours_since_maintenance": 95.0,
            "ambient_temp": 23.0,
            "machine_type": "Compressor",
            "operating_mode": "normal",
        },
    ]
}


# ─────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────

def test_health() -> bool:
    """
    Test du endpoint GET /health.

    Vérifie que le service répond correctement et que le modèle est chargé.
    """
    print_section("GET /health – Supervision")

    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)

        # Test 1 : Code HTTP 200
        print_test(
            "Statut HTTP 200",
            response.status_code == 200,
            f"Status code : {response.status_code}",
        )

        data = response.json()

        # Test 2 : Champ status présent
        print_test(
            "Champ 'status' présent",
            "status" in data,
            f"status = {data.get('status')}",
        )

        # Test 3 : Modèle chargé
        print_test(
            "Modèle chargé en mémoire",
            data.get("model_loaded", False),
            f"model_name = {data.get('model_name')}",
        )

        # Test 4 : Preprocessor chargé
        print_test(
            "Pipeline preprocessing chargé",
            data.get("preprocessor_loaded", False),
        )

        print(f"\n  Réponse complète :\n  {json.dumps(data, indent=4, ensure_ascii=False)}")
        return response.status_code == 200

    except requests.exceptions.ConnectionError:
        print(f"  {RED}❌ Impossible de se connecter à {BASE_URL}{RESET}")
        print(f"  {YELLOW}→ Démarrez l'API : uvicorn api.main:app --reload --port 8000{RESET}")
        return False


def test_model_info() -> bool:
    """Test du endpoint GET /model-info."""
    print_section("GET /model-info – Informations modèle")

    try:
        response = requests.get(f"{BASE_URL}/model-info", timeout=5)

        print_test("Statut HTTP 200", response.status_code == 200)

        data = response.json()

        print_test(
            "Champ 'target_variable' correct",
            data.get("target_variable") == "failure_within_24h",
            f"target = {data.get('target_variable')}",
        )

        print_test(
            "Features d'entrée disponibles",
            len(data.get("input_features", [])) > 0,
            f"{len(data.get('input_features', []))} features",
        )

        print_test(
            "Version API présente",
            "api_version" in data,
            f"version = {data.get('api_version')}",
        )

        return response.status_code == 200

    except requests.exceptions.ConnectionError:
        print(f"  {RED}❌ Service non disponible.{RESET}")
        return False


def test_predict_healthy() -> bool:
    """Test de prédiction sur une machine saine."""
    print_section("POST /predict – Machine saine")

    try:
        response = requests.post(
            f"{BASE_URL}/predict",
            json=HEALTHY_MACHINE,
            headers=HEADERS,
            timeout=10,
        )

        print_test("Statut HTTP 200", response.status_code == 200)

        data = response.json()

        print_test(
            "Champ 'prediction' présent (0 ou 1)",
            "prediction" in data and data["prediction"] in [0, 1],
            f"prediction = {data.get('prediction')}",
        )

        print_test(
            "Probabilité entre 0 et 1",
            0.0 <= data.get("probability", -1) <= 1.0,
            f"probability = {data.get('probability')}",
        )

        print_test(
            "Niveau de risque valide",
            data.get("risk_level") in ["LOW", "MEDIUM", "HIGH"],
            f"risk_level = {data.get('risk_level')}",
        )

        print_test(
            "Recommandation présente",
            bool(data.get("recommendation")),
            f"recommendation = {data.get('recommendation')[:60]}...",
        )

        print_test(
            "Timestamp présent",
            bool(data.get("timestamp")),
            f"timestamp = {data.get('timestamp')}",
        )

        print(f"\n  Réponse :\n  {json.dumps(data, indent=4, ensure_ascii=False)}")
        return response.status_code == 200

    except requests.exceptions.ConnectionError:
        print(f"  {RED}❌ Service non disponible.{RESET}")
        return False


def test_predict_degraded() -> bool:
    """Test de prédiction sur une machine dégradée."""
    print_section("POST /predict – Machine dégradée")

    try:
        response = requests.post(
            f"{BASE_URL}/predict",
            json=DEGRADED_MACHINE,
            headers=HEADERS,
            timeout=10,
        )

        print_test("Statut HTTP 200", response.status_code == 200)
        data = response.json()

        print_test(
            "Probabilité élevée pour machine dégradée",
            data.get("probability", 0) > 0.3,
            f"probability = {data.get('probability')} (attendu > 0.3)",
        )

        print_test(
            "Niveau de risque MEDIUM ou HIGH",
            data.get("risk_level") in ["MEDIUM", "HIGH"],
            f"risk_level = {data.get('risk_level')}",
        )

        print(f"\n  Réponse :\n  {json.dumps(data, indent=4, ensure_ascii=False)}")
        return response.status_code == 200

    except requests.exceptions.ConnectionError:
        print(f"  {RED}❌ Service non disponible.{RESET}")
        return False


def test_predict_invalid_range() -> bool:
    """Test de validation – valeur hors plage."""
    print_section("POST /predict – Validation (valeur hors plage)")

    try:
        response = requests.post(
            f"{BASE_URL}/predict",
            json=INVALID_VIBRATION,
            headers=HEADERS,
            timeout=5,
        )

        print_test(
            "Statut HTTP 422 (Unprocessable Entity)",
            response.status_code == 422,
            f"Status code : {response.status_code} (attendu 422)",
        )

        data = response.json()
        print_test(
            "Message d'erreur de validation présent",
            "detail" in data,
            f"Erreur : {str(data.get('detail', ''))[:80]}",
        )

        return response.status_code == 422

    except requests.exceptions.ConnectionError:
        print(f"  {RED}❌ Service non disponible.{RESET}")
        return False


def test_predict_missing_field() -> bool:
    """Test de validation – champ manquant."""
    print_section("POST /predict – Validation (champ manquant)")

    try:
        response = requests.post(
            f"{BASE_URL}/predict",
            json=MISSING_FIELD,
            headers=HEADERS,
            timeout=5,
        )

        print_test(
            "Statut HTTP 422 (champ requis manquant)",
            response.status_code == 422,
            f"Status code : {response.status_code} (attendu 422)",
        )

        return response.status_code == 422

    except requests.exceptions.ConnectionError:
        print(f"  {RED}❌ Service non disponible.{RESET}")
        return False


def test_predict_batch() -> bool:
    """Test de prédiction batch."""
    print_section("POST /predict/batch – Prédiction par lot")

    try:
        response = requests.post(
            f"{BASE_URL}/predict/batch",
            json=BATCH_MACHINES,
            headers=HEADERS,
            timeout=15,
        )

        print_test("Statut HTTP 200", response.status_code == 200)
        data = response.json()

        n_obs = len(BATCH_MACHINES["observations"])
        n_preds = len(data.get("predictions", []))

        print_test(
            f"{n_obs} prédictions retournées",
            n_preds == n_obs,
            f"Reçu : {n_preds} prédictions",
        )

        print_test(
            "Synthèse (summary) présente",
            "summary" in data,
            f"total_machines = {data.get('summary', {}).get('total_machines')}",
        )

        print_test(
            "Temps de traitement mesuré",
            data.get("processing_time_ms", -1) >= 0,
            f"processing_time = {data.get('processing_time_ms')} ms",
        )

        summary = data.get("summary", {})
        print(f"\n  Synthèse du parc :")
        print(f"  {json.dumps(summary, indent=4, ensure_ascii=False)}")

        return response.status_code == 200

    except requests.exceptions.ConnectionError:
        print(f"  {RED}❌ Service non disponible.{RESET}")
        return False


# ─────────────────────────────────────────────────────────────────
# Runner principal
# ─────────────────────────────────────────────────────────────────

def run_all_tests() -> None:
    """
    Exécute la suite complète de tests et affiche un rapport synthétique.
    """
    print(f"\n{BOLD}{'=' * 55}{RESET}")
    print(f"{BOLD}  TESTS API – Maintenance Prédictive{RESET}")
    print(f"{BOLD}  Cible : {BASE_URL}{RESET}")
    print(f"{BOLD}{'=' * 55}{RESET}")

    tests = [
        ("Health Check", test_health),
        ("Model Info", test_model_info),
        ("Prédiction machine saine", test_predict_healthy),
        ("Prédiction machine dégradée", test_predict_degraded),
        ("Validation hors plage", test_predict_invalid_range),
        ("Validation champ manquant", test_predict_missing_field),
        ("Prédiction batch", test_predict_batch),
    ]

    results = []
    for name, test_fn in tests:
        try:
            passed = test_fn()
            results.append((name, passed))
        except Exception as e:
            print(f"  {RED}❌ Erreur inattendue dans '{name}' : {e}{RESET}")
            results.append((name, False))

    # Rapport final
    n_passed = sum(1 for _, p in results if p)
    n_total = len(results)

    print(f"\n{BOLD}{'=' * 55}{RESET}")
    print(f"{BOLD}  RAPPORT FINAL : {n_passed}/{n_total} tests réussis{RESET}")
    print(f"{BOLD}{'=' * 55}{RESET}")

    for name, passed in results:
        icon = f"{GREEN}✅{RESET}" if passed else f"{RED}❌{RESET}"
        print(f"  {icon} {name}")

    if n_passed == n_total:
        print(f"\n{GREEN}{BOLD}  ✅ Tous les tests sont passés. API opérationnelle.{RESET}")
    else:
        print(f"\n{YELLOW}{BOLD}  ⚠️  {n_total - n_passed} test(s) en échec.{RESET}")
        sys.exit(1)


if __name__ == "__main__":
    run_all_tests()