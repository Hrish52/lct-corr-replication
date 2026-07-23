# tests/test_imports.py
def test_imports():
    """Verify the actual runtime dependencies import cleanly."""
    import numpy, scipy, pandas, matplotlib, joblib  # noqa: F401
    import src  # noqa: F401