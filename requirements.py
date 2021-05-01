import platform
from pkg_resources import packaging

__all__ = ['python_requires', 'setup_requires', 'install_requires', 'extras_require']

pkg_versions = {
    'Cython': '>=0.28.5',
    'numpy': '>=1.17,<2',
    'scipy': '>=1.3,<2',
    'hmmlearn': '==0.2.4',
    'dtaidistance[numpy]': '>=2.2,<2.3',
    'scikit-learn': '>=0.22,<1',
    'tqdm': '>=4.36,<5',
    'joblib': '>=0.14,<1',
    # [torch]
    'torch': '>=1.8+cpu',
    # [test]
    'sphinx': '==3.5.4',
    # [docs]
    'pytest': '==5.3.2',
    # [notebooks]
    'jupyter': '==1.0.0',
    'requests': '==2.25.1',
    'matplotlib': '==3.3.3',
    'pandas': '==1.1.5',
    'seaborn': '==0.11.1',
    'torchaudio': '==0.8.0',
    'torchvision': '==0.9.0',
    'torchfsdd': '==0.1.2',
    'librosa': '==0.8.0'
}

extra_pkgs = {
    'torch': ['dev', 'torch', 'test', 'notebooks'],
    'sphinx': ['dev', 'docs'],
    'pytest': ['dev', 'test'],
    **{pkg:['dev', 'notebooks'] for pkg in [
        'jupyter', 'requests', 'matplotlib', 'pandas',
        'seaborn', 'tqdm', 'torchaudio', 'torchvision',
        'torchfsdd', 'librosa'
    ]},
}

def load_requires(*pkgs):
    return [pkg + pkg_versions[pkg] for pkg in pkgs]

def reverse_extra(extra):
    return [pkg + pkg_versions[pkg] for pkg, extras in extra_pkgs.items() if extra in extras]

python_requires = '>=3.6,<3.10'

setup_requires = load_requires('Cython', 'numpy', 'scipy')

install_requires = load_requires('numpy', 'hmmlearn', 'dtaidistance[numpy]', 'scipy', 'scikit-learn', 'tqdm', 'joblib')
if packaging.version.parse(platform.python_version()) < packaging.version.parse('3.8'):
    install_requires.append('importlib_metadata') # Backports for importlib.metadata in Python <3.8

extras_require = {extra:reverse_extra(extra) for extra in ('torch', 'dev', 'test', 'notebooks', 'docs')}