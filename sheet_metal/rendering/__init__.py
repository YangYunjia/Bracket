from .numpy_model import NumpyBracketModel

def __getattr__(name):
    if name == 'OCC_BracketModel':
        from .occ_model import OCC_BracketModel
        return OCC_BracketModel
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
