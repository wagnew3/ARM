import importlib


def get_model(alias, test=False):
    module = importlib.import_module('genre.models.' + alias)
    if test:
        return module.Model_test
    return module.Model
