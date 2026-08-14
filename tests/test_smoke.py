import importlib

MODULES = [
    "pipeline",
    "retrieval",
    "blockchain",
    "storage",
    "generation",
    "scripts",
]


def test_all_packages_importable():
    for module_name in MODULES:
        importlib.import_module(module_name)


def test_pytest_collects_and_runs():
    assert True
