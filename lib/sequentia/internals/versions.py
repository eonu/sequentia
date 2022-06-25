import importlib, platform
from pkg_resources import packaging
if packaging.version.parse(platform.python_version()) < packaging.version.parse('3.8'):
    import importlib_metadata as metadata
else:
    from importlib import metadata

def check_package(pkg, min_version, url, silent=False):
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

    silent: bool
        Whether or not to raise an error (or alternatively return whether or not the package is installed).

    Returns
    -------
    installed: bool
        Whether or not the package is installed (only in silent mode).
    """
    try:
        importlib.import_module(pkg)
    except ImportError:
        msg = ("Could not find a valid installation of '{pkg}' (>={min_version}), which Sequentia depends on.\n"
        "Visit {url} for more instructions on installing this package.").format(pkg=pkg, url=url, min_version=min_version)
        if silent:
            return False
        raise ModuleNotFoundError(msg)

    installed_version = metadata.version(pkg)
    if packaging.version.parse(installed_version) < packaging.version.parse(min_version):
        msg = ("Could not find a compatible installation of '{pkg}' (>={min_version}), which Sequentia depends on - got version {installed_version}.\n"
        "Visit {url} for more instructions on installing this package.").format(pkg=pkg, url=url, min_version=min_version, installed_version=installed_version)
        if silent:
            return False
        raise ModuleNotFoundError(msg)

    return True
