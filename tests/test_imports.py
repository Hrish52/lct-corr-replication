# tests/test_imports.py
def test_imports():
    import numpy, scipy, pandas, numba, joblib, dask, matplotlib, networkx, tqdm  # noqa: F401
    import src  # noqa: F401
