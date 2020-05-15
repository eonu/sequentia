import pytest, numpy as np
from sequentia.preprocessing import *
from ...support import assert_equal, assert_all_equal

# Set seed for reproducible randomness
seed = 0
np.random.seed(seed)
rng = np.random.RandomState(seed)

# Sample data
X = rng.random((7, 2))
Xs = [i * rng.random((3 * i, 2)) for i in range(1, 4)]

# Length equalizing preprocessor
equalize = Preprocess([Equalize()])

# Zero-trimming preprocessor
trim = Preprocess([TrimZeros()])

# Min-max scaling preprocessor
min_max_scale_kwargs = {'scale': (-5, 5), 'independent': False}
min_max_scale = Preprocess([MinMaxScale(**min_max_scale_kwargs)])

# Centering preprocessor
cent = Preprocess([Center()])

# Standardizing preprocessor
standard = Preprocess([Standardize()])

# Downsampling preprocessor
down_kwargs = {'factor': 3, 'method': 'decimate'}
down = Preprocess([Downsample(**down_kwargs)])

# Filtering preprocessor
filt_kwargs = {'window_size': 3, 'method': 'median'}
filt = Preprocess([Filter(**filt_kwargs)])

# Combined preprocessor
combined = Preprocess([
    Equalize(),
    TrimZeros(),
    MinMaxScale(**min_max_scale_kwargs),
    Center(),
    Standardize(),
    Filter(**filt_kwargs),
    Downsample(**down_kwargs)
])

# ======== #
# Equalize #
# ======== #

def test_equalize_single():
    """Applying length equalizing to a single observation sequence"""
    assert_equal(equalize(X), Equalize()(X))

def test_equalize_multiple():
    """Applying length equalizing to multiple observation sequences"""
    assert_all_equal(equalize(Xs), Equalize()(Xs))

def test_equalize_summary(capsys):
    """Summary of a length equalizing transformation"""
    equalize.summary()
    assert capsys.readouterr().out == (
        '   Preprocessing summary:   \n'
        '============================\n'
        '1. Equalize\n'
        '   Equalize sequence lengths\n'
        '============================\n'
    )

# ========= #
# TrimZeros #
# ========= #

def test_trim_zeros_single():
    """Applying zero-trimming to a single observation sequence"""
    assert_equal(trim(X), TrimZeros()(X))

def test_trim_zeros_multiple():
    """Applying zero-trimming to multiple observation sequences"""
    assert_all_equal(trim(Xs), TrimZeros()(Xs))

def test_trim_zeros_summary(capsys):
    """Summary of a zero-trimming transformation"""
    trim.summary()
    assert capsys.readouterr().out == (
        '   Preprocessing summary:  \n'
        '===========================\n'
        '1. TrimZeros\n   '
        'Remove zero-observations\n'
        '===========================\n'
    )

# =========== #
# MinMaxScale #
# =========== #

def test_min_max_scale_single():
    """Applying min-max scaling to a single observation sequence"""
    assert_equal(min_max_scale(X), MinMaxScale(**min_max_scale_kwargs)(X))

def test_min_max_scale_multiple():
    """Applying min-max scaling to multiple observation sequences"""
    assert_all_equal(min_max_scale(Xs), MinMaxScale(**min_max_scale_kwargs)(Xs))

def test_min_max_scale_summary(capsys):
    """Summary of a min-max scaling transformation"""
    min_max_scale.summary()
    assert capsys.readouterr().out == (
        '        Preprocessing summary:       \n'
        '=====================================\n'
        '1. MinMaxScale\n'
        '   Min-max scaling into range (-5, 5)\n'
        '=====================================\n'
    )

# ====== #
# Center #
# ====== #

def test_center_single():
    """Applying centering to a single observation sequence"""
    assert_equal(cent(X), Center()(X))

def test_center_multiple():
    """Applying centering to multiple observation sequences"""
    assert_all_equal(cent(Xs), Center()(Xs))

def test_center_summary(capsys):
    """Summary of a centering transformation"""
    cent.summary()
    assert capsys.readouterr().out == (
        '              Preprocessing summary:              \n'
        '==================================================\n'
        '1. Center\n'
        '   Centering around mean (zero mean) (independent)\n'
        '==================================================\n'
    )

# =========== #
# Standardize #
# =========== #

def test_standardize_single():
    """Applying standardization to a single observation sequence"""
    assert_equal(standard(X), Standardize()(X))

def test_standardize_multiple():
    """Applying standardization to multiple observation sequences"""
    assert_all_equal(standard(Xs), Standardize()(Xs))

def test_standardize_summary(capsys):
    """Summary of a standardizing transformation"""
    standard.summary()
    assert capsys.readouterr().out == (
        '                   Preprocessing summary:                   \n'
        '============================================================\n'
        '1. Standardize\n'
        '   Standard scaling (zero mean, unit variance) (independent)\n'
        '============================================================\n'
    )

# ========== #
# Downsample #
# ========== #

def test_downsample_single():
    """Applying downsampling to a single observation sequence"""
    assert_equal(down(X), Downsample(**down_kwargs)(X))

def test_downsample_multiple():
    """Applying downsampling to multiple observation sequences"""
    assert_all_equal(down(Xs), Downsample(**down_kwargs)(Xs))

def test_downsample_summary(capsys):
    """Summary of a downsampling transformation"""
    down.summary()
    assert capsys.readouterr().out == (
        '         Preprocessing summary:         \n'
        '========================================\n'
        '1. Downsample\n'
        '   Decimation downsampling with factor 3\n'
        '========================================\n'
    )

# ====== #
# Filter #
# ====== #

def test_filter_single():
    """Applying filtering to a single observation sequence"""
    assert_equal(filt(X), Filter(**filt_kwargs)(X))

def test_filter_multiple():
    """Applying filtering to multiple observation sequences"""
    assert_all_equal(filt(Xs), Filter(**filt_kwargs)(Xs))

def test_filter_summary(capsys):
    """Summary of a filtering transformation"""
    filt.summary()
    assert capsys.readouterr().out == (
        '        Preprocessing summary:        \n'
        '======================================\n'
        '1. Filter\n'
        '   Median filtering with window-size 3\n'
        '======================================\n'
    )

# ======================== #
# Combined transformations #
# ======================== #

combined = Preprocess([
    Equalize(),
    TrimZeros(),
    MinMaxScale(**min_max_scale_kwargs),
    Center(),
    Standardize(),
    Filter(**filt_kwargs),
    Downsample(**down_kwargs)
])

def test_combined_single():
    """Applying combined transformations to a single observation sequence"""
    X_pre = X
    X_pre = Equalize()(X_pre)
    X_pre = TrimZeros()(X_pre)
    X_pre = MinMaxScale(**min_max_scale_kwargs)(X_pre)
    X_pre = Center()(X_pre)
    X_pre = Standardize()(X_pre)
    X_pre = Filter(**filt_kwargs)(X_pre)
    X_pre = Downsample(**down_kwargs)(X_pre)
    assert_equal(combined(X), X_pre)

def test_combined_multiple():
    """Applying combined transformations to multiple observation sequences"""
    Xs_pre = Xs
    Xs_pre = Equalize()(Xs_pre)
    Xs_pre = TrimZeros()(Xs_pre)
    Xs_pre = MinMaxScale(**min_max_scale_kwargs)(Xs_pre)
    Xs_pre = Center()(Xs_pre)
    Xs_pre = Standardize()(Xs_pre)
    Xs_pre = Filter(**filt_kwargs)(Xs_pre)
    Xs_pre = Downsample(**down_kwargs)(Xs_pre)
    assert_all_equal(combined(Xs), Xs_pre)

def test_combined_summary(capsys):
    """Summary with combined transformations applied"""
    combined.summary()
    assert capsys.readouterr().out == (
        '                   Preprocessing summary:                   \n'
        '============================================================\n'
        '1. Equalize\n'
        '   Equalize sequence lengths\n'
        '------------------------------------------------------------\n'
        '2. TrimZeros\n'
        '   Remove zero-observations\n'
        '------------------------------------------------------------\n'
        '3. MinMaxScale\n'
        '   Min-max scaling into range (-5, 5)\n'
        '------------------------------------------------------------\n'
        '4. Center\n'
        '   Centering around mean (zero mean) (independent)\n'
        '------------------------------------------------------------\n'
        '5. Standardize\n'
        '   Standard scaling (zero mean, unit variance) (independent)\n'
        '------------------------------------------------------------\n'
        '6. Filter\n'
        '   Median filtering with window-size 3\n'
        '------------------------------------------------------------\n'
        '7. Downsample\n'
        '   Decimation downsampling with factor 3\n'
        '============================================================\n'
    )

def test_empty_summary():
    """Summary without any transformations applied"""
    with pytest.raises(RuntimeError) as e:
        Preprocess([]).summary()
    assert str(e.value) == 'At least one preprocessing transformation is required'