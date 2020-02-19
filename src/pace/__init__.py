# Set the __version__ attribute.
import pkg_resources as _pgk_resources
__version__ = _pgk_resources.require("pace")[0].version

# Import the package-level API.
from .evaluation import evaluate
from .featurization import encode, get_allele_similarity_mat, get_similar_alleles
from .definitions import *

# Since this is a package, we need to explicitly tell Python that we want our
# globals exposed.
__all__ = [g for g in globals().keys() if not g.startswith('_')]
