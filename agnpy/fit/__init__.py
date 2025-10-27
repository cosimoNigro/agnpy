import logging

logger = logging.getLogger(__name__)

try:
    from .models import *
    from .data import *
except ImportError:
    logger.warning("sherpa and gammapy are not installed, the agnpy.fit module cannot be used")
    pass
