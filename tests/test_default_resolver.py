import json, os, tempfile
from src.defaults import load_defaults, get_defaults_for

def test_defaults_resolver_basic():
    d = {
        "250": {"0.05": {"B": 200, "coarse_grid": None, "winsorize": 5, "var_method": "cai_liu"}}
    }
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "defaults.json")
        with open(path, "w") as f:
            json.dump(d, f)
        D = load_defaults(path)
        x = get_defaults_for(250, 0.05, path)
        assert D["250"]["0.05"]["B"] == 200
        assert x["winsorize"] == 5