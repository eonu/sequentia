import pytest
import numpy as np
from sequentia.preprocessing import (
    Preprocess,
    trim_zeros, downsample, center, standardize, fft, filtrate,
    _trim_zeros, _downsample, _center, _standardize, _fft, _filtrate
)
from ...support import assert_equal, assert_all_equal

# Set seed for reproducible randomness
seed = 0
np.random.seed(seed)
rng = np.random.RandomState(seed)

# Sample data
X = rng.random((7, 2))
Xs = [i * rng.random((3 * i, 2)) for i in range(1, 4)]

# Zero-trimming preprocessor
trim = Preprocess()
trim.trim_zeros()

# Centering preprocessor
cent = Preprocess()
cent.center()

# Standardizing preprocessor
standard = Preprocess()
standard.standardize()

# Discrete Fourier Transform preprocessor
fourier = Preprocess()
fourier.fft()

# Downsampling preprocessor
down = Preprocess()
down_kwargs = {'n': 3, 'method': 'decimate'}
down.downsample(**down_kwargs)

# Filtering preprocessor
filt = Preprocess()
filt_kwargs = {'n': 3, 'method': 'median'}
filt.filtrate(**filt_kwargs)

# Combined preprocessor
combined = Preprocess()
combined.trim_zeros()
combined.center()
combined.standardize()
combined.filtrate(**filt_kwargs)
combined.downsample(**down_kwargs)
combined.fft()

# ======================= #
# Preprocess.trim_zeros() #
# ======================= #

def test_trim_zeros_adds_transform():
    """Applying a single zero-trimming transformation"""
    assert len(trim._transforms) == 1
    assert trim._transforms[0] == (_trim_zeros, {})

def test_trim_zeros_single():
    """Applying zero-trimming to a single observation sequence"""
    assert_equal(trim.transform(X), trim_zeros(X))

def test_trim_zeros_multiple():
    """Applying zero-trimming to multiple observation sequences"""
    assert_all_equal(trim.transform(Xs), trim_zeros(Xs))

def test_trim_zeros_summary(capsys):
    """Summary of a zero-trimming transformation"""
    trim.summary()
    assert capsys.readouterr().out == (
        'Preprocessing summary:\n'
        '======================\n'
        '1. Zero-trimming\n'
        '======================\n'
    )

# =================== #
# Preprocess.center() #
# =================== #

def test_center_adds_transform():
    """Applying a single centering transformation"""
    assert len(cent._transforms) == 1
    assert cent._transforms[0] == (_center, {})

def test_center_single():
    """Applying centering to a single observation sequence"""
    assert_equal(cent.transform(X), center(X))

def test_center_multiple():
    """Applying centering to multiple observation sequences"""
    assert_all_equal(cent.transform(Xs), center(Xs))

def test_center_summary(capsys):
    """Summary of a centering transformation"""
    cent.summary()
    assert capsys.readouterr().out == (
        'Preprocessing summary:\n'
        '======================\n'
        '1. Centering\n'
        '======================\n'
    )

# ======================== #
# Preprocess.standardize() #
# ======================== #

def test_standardize_adds_transform():
    """Applying a single standardizing transformation"""
    assert len(standard._transforms) == 1
    assert standard._transforms[0] == (_standardize, {})

def test_standardize_single():
    """Applying standardization to a single observation sequence"""
    assert_equal(standard.transform(X), standardize(X))

def test_standardize_multiple():
    """Applying standardization to multiple observation sequences"""
    assert_all_equal(standard.transform(Xs), standardize(Xs))

def test_standardize_summary(capsys):
    """Summary of a standardizing transformation"""
    standard.summary()
    assert capsys.readouterr().out == (
        'Preprocessing summary:\n'
        '======================\n'
        '1. Standardization\n'
        '======================\n'
    )

# ================ #
# Preprocess.fft() #
# ================ #

def test_fft_adds_transform():
    """Applying a single discrete fourier transformation"""
    assert len(fourier._transforms) == 1
    assert fourier._transforms[0] == (_fft, {})

def test_fft_single():
    """Applying discrete fourier transformation to a single observation sequence"""
    assert_equal(fourier.transform(X), fft(X))

def test_fft_multiple():
    """Applying discrete fourier transformation to multiple observation sequences"""
    assert_all_equal(fourier.transform(Xs), fft(Xs))

def test_fft_summary(capsys):
    """Summary of a discrete fourier transformation"""
    fourier.summary()
    assert capsys.readouterr().out == (
        '    Preprocessing summary:   \n'
        '=============================\n'
        '1. Discrete Fourier Transform\n'
        '=============================\n'
    )

# ======================= #
# Preprocess.downsample() #
# ======================= #

def test_downsample_adds_transform():
    """Applying a single downsampling transformation"""
    assert len(down._transforms) == 1
    assert down._transforms[0] == (_downsample, down_kwargs)

def test_downsample_single():
    """Applying downsampling to a single observation sequence"""
    assert_equal(down.transform(X), downsample(X, **down_kwargs))

def test_downsample_multiple():
    """Applying downsampling to multiple observation sequences"""
    assert_all_equal(down.transform(Xs), downsample(Xs, **down_kwargs))

def test_downsample_summary(capsys):
    """Summary of a downsampling transformation"""
    down.summary()
    assert capsys.readouterr().out == (
        '          Preprocessing summary:          \n'
        '==========================================\n'
        '1. Downsampling:\n'
        '   Decimation with downsample factor (n=3)\n'
        '==========================================\n'
    )

# ===================== #
# Preprocess.filtrate() #
# ===================== #

def test_filtrate_adds_transform():
    """Applying a single filtering transformation"""
    assert len(filt._transforms) == 1
    assert filt._transforms[0] == (_filtrate, filt_kwargs)

def test_filtrate_single():
    """Applying filtering to a single observation sequence"""
    assert_equal(filt.transform(X), filtrate(X, **filt_kwargs))

def test_filtrate_multiple():
    """Applying filtering to multiple observation sequences"""
    assert_all_equal(filt.transform(Xs), filtrate(Xs, **filt_kwargs))

def test_filtrate_summary(capsys):
    """Summary of a filtering transformation"""
    filt.summary()
    assert capsys.readouterr().out == (
        '         Preprocessing summary:        \n'
        '=======================================\n'
        '1. Filtering:\n'
        '   Median filter with window size (n=3)\n'
        '=======================================\n'
    )

# ======================== #
# Combined transformations #
# ======================== #

def test_combined_adds_transforms():
    """Applying multiple filtering transformations"""
    assert len(combined._transforms) == 6
    assert combined._transforms == [
        (_trim_zeros, {}),
        (_center, {}),
        (_standardize, {}),
        (_filtrate, filt_kwargs),
        (_downsample, down_kwargs),
        (_fft, {})
    ]

def test_combined_single():
    """Applying combined transformations to a single observation sequence"""
    X_pre = X
    X_pre = trim_zeros(X_pre)
    X_pre = center(X_pre)
    X_pre = standardize(X_pre)
    X_pre = filtrate(X_pre, **filt_kwargs)
    X_pre = downsample(X_pre, **down_kwargs)
    X_pre = fft(X_pre)
    assert_equal(combined.transform(X), X_pre)

def test_combined_multiple():
    """Applying combined transformations to multiple observation sequences"""
    Xs_pre = Xs
    Xs_pre = trim_zeros(Xs_pre)
    Xs_pre = center(Xs_pre)
    Xs_pre = standardize(Xs_pre)
    Xs_pre = filtrate(Xs_pre, **filt_kwargs)
    Xs_pre = downsample(Xs_pre, **down_kwargs)
    Xs_pre = fft(Xs_pre)
    assert_all_equal(combined.transform(Xs), Xs_pre)

def test_combined_summary(capsys):
    """Summary with combined transformations applied"""
    combined.summary()
    assert capsys.readouterr().out == (
        '          Preprocessing summary:          \n'
        '==========================================\n'
        '1. Zero-trimming\n'
        '------------------------------------------\n'
        '2. Centering\n'
        '------------------------------------------\n'
        '3. Standardization\n'
        '------------------------------------------\n'
        '4. Filtering:\n'
        '   Median filter with window size (n=3)\n'
        '------------------------------------------\n'
        '5. Downsampling:\n'
        '   Decimation with downsample factor (n=3)\n'
        '------------------------------------------\n'
        '6. Discrete Fourier Transform\n'
        '==========================================\n'
    )

# ==================== #
# Preprocess.summary() #
# ==================== #

def test_empty_summary():
    """Summary without any transformations applied"""
    with pytest.raises(RuntimeError) as e:
        Preprocess().summary()
    assert str(e.value) == 'At least one preprocessing transformation is required'