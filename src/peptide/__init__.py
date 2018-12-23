# Set default logging handler to avoid "No handler found" warnings.
import logging as _logging
_logging.getLogger(__name__).addHandler(_logging.NullHandler())

# Set the __version__ attribute.
import pkg_resources as _pgk_resources
__version__ = _pgk_resources.require("puma")[0].version

# If the environment variable PUMA_MINIMAL_IMPORT isn't set, we import all
# services as submodules within the puma package (for easier access).
import os as _os
if 'PUMA_MINIMAL_IMPORT' not in _os.environ:
    import importlib as _importlib
    for module in [
        'calc', 'config', 'context', 'cradle', 'group', 'iam', 'iss', 'patient',
        'plan', 'realm', 'rks', 'token', 'user']:
        _importlib.import_module('.' + module, package='puma')
