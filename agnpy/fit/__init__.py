import logging
try:
    from .models import *
    from .data import *
except ImportError:
    logging.warning("sherpa and gammapy are not installed, the agnpy.fit module cannot be used")
    pass
