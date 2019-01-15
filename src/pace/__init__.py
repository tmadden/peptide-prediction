# Set the __version__ attribute.
import pkg_resources as _pgk_resources
__version__ = _pgk_resources.require("pace")[0].version

# Import the package-level API.
from .evaluation import evaluate
from .definitions import *
