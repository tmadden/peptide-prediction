# Set default logging handler to avoid "No handler found" warnings.
import logging as _logging

# _logging.getLogger(__name__).addHandler(_logging.NullHandler())

# Set the __version__ attribute.
import pkg_resources as _pgk_resources

# __version__ = _pgk_resources.require("peptide")[0].version
