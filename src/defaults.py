# src/defaults.py
import json, os

def load_defaults(path="results/defaults.json"):
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return {}

def get_defaults_for(p, alpha, path="results/defaults.json"):
    D = load_defaults(path)
    sp, sa = str(int(p)), f"{float(alpha):.2f}"
    try:
        return D[sp][sa]
    except Exception:
        return None