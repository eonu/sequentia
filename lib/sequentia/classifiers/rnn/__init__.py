import importlib, platform
from pkg_resources import packaging
if packaging.version.parse(platform.python_version()) < packaging.version.parse('3.8'):
    import importlib_metadata as metadata
else:
    from importlib import metadata

MIN_TORCH_VERSION = '1.8'

def check_package(pkg, min_version, url):
    """Checks whether a specified package has been installed,
    and whether the installed version meets a specified minimum.

    Parameters
    ----------
    pkg: str
        Name of the package.

    min_version: str
        Minimum version for the package, e.g. `1.8`.

    url: str
        Package installation page URL (for help).
    """
    try:
        importlib.import_module(pkg)
    except ImportError:
        msg = ("Could not find a valid installation of '{pkg}' (>={min_version}), which Sequentia depends on.\n"
        "Visit {url} for more instructions on installing this package.").format(pkg=pkg, url=url, min_version=min_version)
        raise ModuleNotFoundError(msg)

    installed_version = metadata.version(pkg)
    if packaging.version.parse(installed_version) < packaging.version.parse(min_version):
        msg = ("Could not find a compatible installation of '{pkg}' (>={min_version}), which Sequentia depends on - got version {installed_version}.\n"
        "Visit {url} for more instructions on installing this package.").format(pkg=pkg, url=url, min_version=min_version, installed_version=installed_version)
        raise ImportWarning(msg)

# Check that (at least) the minimum dependency version is installed
check_package('torch', MIN_TORCH_VERSION, url='https://pytorch.org/')

# Import from the package
from .deepgru import DeepGRU, _EncoderNetwork, _AttentionModule, _Classifier
from .utils import collate_fn