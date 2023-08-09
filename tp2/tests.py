from pathlib import Path


def load_tests(loader, tests, pattern):
    here = Path(__file__)
    names = [f"tp2.{p.stem}" for p in here.parent.glob("*.py") if p != here]
    tests.addTests(loader.loadTestsFromNames(names))
    return tests
