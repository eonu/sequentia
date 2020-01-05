import pytest
import numpy as np
from sequentia.preprocessing import trim_zeros, downsample, center, standardize, fft, filtrate
from ...support import assert_equal, assert_all_equal

# Set seed for reproducible randomness
seed = 0
np.random.seed(seed)
rng = np.random.RandomState(seed)

# Sample data
X_even = rng.random((6, 2))
X_odd = rng.random((7, 2))
Xs = [i * rng.random((3 * i, 2)) for i in range(1, 4)]

# Zero-padded sample data
zeros = np.zeros((3, 2))
X_padded = np.vstack((zeros, X_even, zeros))
Xs_padded = [np.vstack((zeros, x, zeros)) for x in Xs]

# ============ #
# trim_zeros() #
# ============ #

def test_trim_zeros_single():
    """Trim a single zero-padded observation sequence"""
    assert_equal(trim_zeros(X_padded), np.array([
        [0.5488135 , 0.71518937],
        [0.60276338, 0.54488318],
        [0.4236548 , 0.64589411],
        [0.43758721, 0.891773  ],
        [0.96366276, 0.38344152],
        [0.79172504, 0.52889492]
    ]))

def test_trim_zeros_multiple():
    """Trim multiple zero-padded observation sequences"""
    assert_all_equal(trim_zeros(Xs_padded), [
        np.array([
            [0.14335329, 0.94466892],
            [0.52184832, 0.41466194],
            [0.26455561, 0.77423369]
        ]),
        np.array([
            [0.91230066, 1.1368679 ],
            [0.0375796 , 1.23527099],
            [1.22419145, 1.23386799],
            [1.88749616, 1.3636406 ],
            [0.7190158 , 0.87406391],
            [1.39526239, 0.12045094]
        ]),
        np.array([
            [2.00030015, 2.01191361],
            [0.63114768, 0.38677889],
            [0.94628505, 1.09113231],
            [1.71059031, 1.31580454],
            [2.96512151, 0.30613443],
            [0.62663027, 0.48392855],
            [1.95932498, 0.75987481],
            [1.39893232, 0.73327678],
            [0.47690875, 0.33112542]
        ])
    ])

# ======== #
# center() #
# ======== #

def test_center_single_even():
    """Center a single even-length observation sequence"""
    assert_equal(center(X_even), np.array([
        [-0.07922094, 0.09684335 ],
        [-0.02527107, -0.07346283],
        [-0.20437965, 0.0275481  ],
        [-0.19044724, 0.27342698 ],
        [0.33562831 , -0.2349045 ],
        [0.16369059 , -0.0894511 ]
    ]))

def test_center_single_odd():
    """Center a single odd-length observation sequence"""
    assert_equal(center(X_odd), np.array([
        [0.14006915 , 0.2206014  ],
        [-0.35693936, -0.61786594],
        [-0.40775702, 0.1276246  ],
        [0.35018134 , 0.16501691 ],
        [0.55064293 , 0.09416332 ],
        [0.03350395 , 0.07553393 ],
        [-0.30970099, -0.06507422]
    ]))

def test_center_multiple():
    """Center multiple observation sequences"""
    assert_all_equal(center(Xs), [
        np.array([
            [-0.16656579, 0.23348073 ],
            [0.21192925 , -0.29652624],
            [-0.04536346, 0.06304551 ]
        ]),
        np.array([
            [-0.11700701, 0.14284084 ],
            [-0.99172808, 0.24124394 ],
            [0.19488377 , 0.23984094 ],
            [0.85818848 , 0.36961354 ],
            [-0.31029188, -0.11996315],
            [0.36595472 , -0.87357611]
        ]),
        np.array([
            [0.58749559 , 1.18747257 ],
            [-0.78165687, -0.43766215],
            [-0.46651951, 0.26669127 ],
            [0.29778575 , 0.4913635  ],
            [1.55231696 , -0.51830661],
            [-0.78617429, -0.34051249],
            [0.54652042 , -0.06456623],
            [-0.01387224, -0.09116426],
            [-0.93589581, -0.49331562]
        ])
    ])

# ============= #
# standardize() #
# ============= #

def test_standardize_single_even():
    """Standardize a single even-length observation sequence"""
    assert_equal(standardize(X_even), np.array([
        [-0.40964472,  0.60551094],
        [-0.13067455, -0.45932478],
        [-1.05682966,  0.17224387],
        [-0.98478635,  1.70959629],
        [ 1.73550526, -1.46873528],
        [ 0.84643002, -0.55929105]
    ]))

def test_standardize_single_odd():
    """Standardize a single odd-length observation sequence"""
    assert_equal(standardize(X_odd), np.array([
        [ 0.40527155,  0.83146609],
        [-1.03275681, -2.32879115],
        [-1.17979099,  0.48102837],
        [ 1.01320338,  0.62196325],
        [ 1.59321247,  0.35490986],
        [ 0.09693924,  0.28469405],
        [-0.89607884, -0.24527047]
    ]))

def test_standardize_multiple():
    """Standardize multiple observation sequences"""
    assert_all_equal(standardize(Xs), [
        np.array([
            [-1.05545468,  1.05686059],
            [ 1.34290313, -1.34223879],
            [-0.28744845,  0.2853782 ]
        ]),
        np.array([
            [-0.20256659,  0.34141162],
            [-1.71691396,  0.57661018],
            [ 0.33738952,  0.57325679],
            [ 1.4857256 ,  0.88343331],
            [-0.53718803, -0.28673041],
            [ 0.63355347, -2.08798149]
        ]),
        np.array([
            [ 0.75393018,  2.22884906],
            [-1.0030964 , -0.82147823],
            [-0.59868217,  0.50057122],
            [ 0.38214698,  0.922274  ],
            [ 1.99208067, -0.97284537],
            [-1.00889357, -0.63913134],
            [ 0.70134695, -0.12118881],
            [-0.01780218, -0.17111248],
            [-1.20103046, -0.92593806]
        ])
    ])

# ============ #
# downsample() #
# ============ #

def test_downsample_single_large_factor():
    """Downsample a single observation sequence with a downsample factor that is too large"""
    with pytest.raises(ValueError) as e:
        downsample(X_even, n=7)
    assert str(e.value) == 'Expected downsample factor to be no greater than the number of frames'

def test_downsample_single_decimate_max():
    """Downsample a single observation sequence with decimation and the maximum downsample factor"""
    assert_equal(downsample(X_even, n=6, method='decimate'), np.array([
        [0.548814, 0.715189]
    ]))

def test_downsample_single_decimate():
    """Downsample a single observation sequence with decimation"""
    assert_equal(downsample(X_odd, n=3, method='decimate'), np.array([
        [0.56804456, 0.92559664],
        [0.77815675, 0.87001215],
        [0.11827443, 0.63992102]
    ]))

def test_downsample_single_average_max():
    """Downsample a single observation sequence with averaging and the maximum downsample factor"""
    assert_equal(downsample(X_even, n=6, method='average'), np.array([
        [0.62803445, 0.61834602]
    ]))

def test_downsample_single_average():
    """Downsample a single observation sequence with averaging"""
    assert_equal(downsample(X_odd, n=3, method='average'), np.array([
        [0.21976634, 0.61511526],
        [0.73941815, 0.81656663],
        [0.11827443, 0.63992102]
    ]))

def test_downsample_multiple_large_factor():
    """Downsample multiple observation sequences with a downsample factor that is too large"""
    with pytest.raises(ValueError) as e:
        downsample(Xs, n=4)
    assert str(e.value) == 'Expected downsample factor to be no greater than the number of frames in the shortest sequence'

def test_downsample_multiple_decimate_max():
    """Downsample multiple observation sequences with decimation and the maximum downsample factor"""
    assert_all_equal(downsample(Xs, n=3, method='decimate'), [
        np.array([[0.14335329, 0.94466892]]),
        np.array([
            [0.91230066, 1.1368679],
            [1.88749616, 1.3636406]
        ]),
        np.array([
            [2.00030015, 2.01191361],
            [1.71059031, 1.31580454],
            [1.95932498, 0.75987481]
        ])
    ])

def test_downsample_multiple_decimate():
    """Downsample multiple observation sequences with decimation"""
    assert_all_equal(downsample(Xs, n=2, method='decimate'), [
        np.array([
            [0.14335329, 0.94466892],
            [0.26455561, 0.77423369]
        ]),
        np.array([
            [0.91230066, 1.1368679 ],
            [1.22419145, 1.23386799],
            [0.7190158 , 0.87406391]
        ]),
        np.array([
            [2.00030015, 2.01191361],
            [0.94628505, 1.09113231],
            [2.96512151, 0.30613443],
            [1.95932498, 0.75987481],
            [0.47690875, 0.33112542]
        ])
    ])

def test_downsample_multiple_average_max():
    """Downsample multiple observation sequences with averaging and the maximum downsample factor"""
    assert_all_equal(downsample(Xs, n=3, method='average'), [
        np.array([[0.30991907, 0.71118818]]),
        np.array([
            [0.72469057, 1.2020023 ],
            [1.33392478, 0.78605182]
        ]),
        np.array([
            [1.19257763, 1.16327494],
            [1.76744736, 0.70195584],
            [1.27838868, 0.60809234]
        ])
    ])

def test_downsample_multiple_average():
    """Downsample multiple observation sequences with averaging"""
    assert_all_equal(downsample(Xs, n=2, method='average'), [
        np.array([
            [0.3326008 , 0.67966543],
            [0.26455561, 0.77423369]
        ]),
        np.array([
            [0.47494013, 1.18606945],
            [1.5558438 , 1.2987543 ],
            [1.0571391 , 0.49725743]
        ]),
        np.array([
            [1.31572391, 1.19934625],
            [1.32843768, 1.20346843],
            [1.79587589, 0.39503149],
            [1.67912865, 0.74657579],
            [0.47690875, 0.33112542]
        ])
    ])

# ===== #
# fft() #
# ===== #

def test_fft_single_even():
    """Fourier-transform a single even-length observation sequence"""
    assert_equal(fft(X_even), np.array([
        [3.76820669 , 3.7100761  ],
        [0.11481172 , -0.1543624 ],
        [0.63130621 , -0.24113686],
        [-0.40450227, 0.5554055  ],
        [-0.30401501, 0.21344437 ],
        [0.10405544 , -0.22102611]
    ]))

def test_fft_single_odd():
    """Fourier-transform a single odd-length observation sequence"""
    assert_equal(fft(X_odd), np.array([
        [2.9958279  , 4.93496669 ],
        [-1.00390978, -0.48392518],
        [0.5541071  , 0.35066311 ],
        [1.18725568 , 0.35112659 ],
        [-0.30212914, 0.61692894 ],
        [ 0.30689611, 0.90490347 ],
        [-0.12906015, 0.21149633 ]
    ]))

def test_fft_multiple():
    """Fourier-transform multiple observation sequences"""
    assert_all_equal(fft(Xs), [
        np.array([
            [0.92975722 , 2.13356455],
            [-0.24984868, 0.3502211 ],
            [-0.22282202, 0.31139827]
        ]),
        np.array([
            [6.17584606 , 5.96416233 ],
            [-1.23037812, -0.60287768],
            [0.73829285 , -1.27706196],
            [1.1117722  , 0.76868158 ],
            [1.61328273 , -0.65386301],
            [-0.46483024, 0.52543726 ]
        ]),
        np.array([
            [1.27152410e+01 , 7.41996935e+00 ],
            [-1.95373695e+00, 1.09840950e+00 ],
            [-2.37772910e-01, -8.08832368e-01],
            [9.05412519e-01 , -1.04236869e-02],
            [1.29066145e+00 , 1.89963643e-01 ],
            [2.14770264e+00 , 2.42140476e+00 ],
            [-2.55077169e+00, 4.15688893e-01 ],
            [1.54435193e+00 , 1.83423599e+00 ],
            [2.17466597e+00 , -4.45551803e-01]
        ])
    ])

# ========== #
# filtrate() #
# ========== #

def test_filtrate_single_large_window():
    """Filter a single observation sequence with a window size that is too large"""
    with pytest.raises(ValueError) as e:
        filtrate(X_even, n=7)
    assert str(e.value) == 'Expected window size to be no greater than the number of frames'

def test_filtrate_single_median_max():
    """Filter a single observation sequence with median filtering and the maximum window size"""
    assert_equal(filtrate(X_even, n=6, method='median'), np.array([
        [0.49320036, 0.68054174],
        [0.5488135 , 0.64589411],
        [0.57578844, 0.59538865],
        [0.60276338, 0.54488318],
        [0.61465612, 0.58739452],
        [0.79172504, 0.52889492]
    ]))

def test_filtrate_single_median():
    """Filter a single observation sequence with median filtering"""
    assert_equal(filtrate(X_odd, n=3, method='median'), np.array([
        [0.31954031, 0.50636297],
        [0.07103606, 0.83261985],
        [0.07103606, 0.83261985],
        [0.77815675, 0.83261985],
        [0.77815675, 0.79915856],
        [0.46147936, 0.78052918],
        [0.28987689, 0.7102251 ]
    ]))

def test_filtrate_single_mean_max():
    """Filter a single observation sequence with mean filtering and the maximum window size"""
    assert_equal(filtrate(X_even, n=6, method='mean'), np.array([
        [0.50320472, 0.69943492],
        [0.59529633, 0.63623624],
        [0.62803445, 0.61834602],
        [0.64387864, 0.59897735],
        [0.65415745, 0.61250089],
        [0.73099167, 0.60136981]
    ]))

def test_filtrate_single_mean():
    """Filter a single observation sequence with mean filtering"""
    assert_equal(filtrate(X_odd, n=3, method='mean'), np.array([
        [0.31954031, 0.50636297],
        [0.21976634, 0.61511526],
        [0.28980374, 0.5965871 ],
        [0.59233116, 0.83393019],
        [0.73941815, 0.81656663],
        [0.51945738, 0.73986959],
        [0.28987689, 0.7102251 ]
    ]))

def test_filtrate_multiple_large_window():
    """Filter multiple observation sequences with a window size that is too large"""
    with pytest.raises(ValueError) as e:
        filtrate(Xs, n=4)
    assert str(e.value) == 'Expected window size to be no greater than the number of frames in the shortest sequence'

def test_filtrate_multiple_median_max():
    """Filter multiple observation sequences with median filtering and the maximum window size"""
    assert_all_equal(filtrate(Xs, n=3, method='median'), [
        np.array([
            [0.3326008 , 0.67966543],
            [0.26455561, 0.77423369],
            [0.39320197, 0.59444781]
        ]),
        np.array([
            [0.47494013, 1.18606945],
            [0.91230066, 1.23386799],
            [1.22419145, 1.23527099],
            [1.22419145, 1.23386799],
            [1.39526239, 0.87406391],
            [1.0571391 , 0.49725743]
        ]),
        np.array([
            [1.31572391, 1.19934625],
            [0.94628505, 1.09113231],
            [0.94628505, 1.09113231],
            [1.71059031, 1.09113231],
            [1.71059031, 0.48392855],
            [1.95932498, 0.48392855],
            [1.39893232, 0.73327678],
            [1.39893232, 0.73327678],
            [0.93792053, 0.5322011 ]
        ])
    ])

def test_filtrate_multiple_median():
    """Filter multiple observation sequences with median filtering"""
    assert_all_equal(filtrate(Xs, n=2, method='median'), [
        np.array([
            [0.3326008 , 0.67966543],
            [0.39320197, 0.59444781],
            [0.26455561, 0.77423369]
        ]),
        np.array([
            [0.47494013, 1.18606945],
            [0.63088552, 1.23456949],
            [1.5558438 , 1.2987543 ],
            [1.30325598, 1.11885225],
            [1.0571391 , 0.49725743],
            [1.39526239, 0.12045094]
        ]),
        np.array([
            [1.31572391, 1.19934625],
            [0.78871637, 0.7389556 ],
            [1.32843768, 1.20346843],
            [2.33785591, 0.81096949],
            [1.79587589, 0.39503149],
            [1.29297762, 0.62190168],
            [1.67912865, 0.74657579],
            [0.93792053, 0.5322011 ],
            [0.47690875, 0.33112542]
        ])
    ])

def test_filtrate_multiple_average_max():
    """Filter multiple observation sequences with mean filtering and the maximum window size"""
    assert_all_equal(filtrate(Xs, n=3, method='mean'), [
        np.array([
            [0.3326008 , 0.67966543],
            [0.30991907, 0.71118818],
            [0.39320197, 0.59444781]
        ]),
        np.array([
            [0.47494013, 1.18606945],
            [0.72469057, 1.2020023 ],
            [1.04975573, 1.2775932 ],
            [1.27690113, 1.15719083],
            [1.33392478, 0.78605182],
            [1.0571391 , 0.49725743]
        ]),
        np.array([
            [1.31572391, 1.19934625],
            [1.19257763, 1.16327494],
            [1.09600768, 0.93123858],
            [1.87399896, 0.9043571 ],
            [1.76744736, 0.70195584],
            [1.85035892, 0.51664593],
            [1.32829585, 0.65902671],
            [1.27838868, 0.60809234],
            [0.93792053, 0.5322011 ]
        ])
    ])

def test_filtrate_multiple_mean():
    """Filter multiple observation sequences with mean filtering"""
    assert_all_equal(filtrate(Xs, n=2, method='mean'), [
        np.array([
            [0.3326008 , 0.67966543],
            [0.39320197, 0.59444781],
            [0.26455561, 0.77423369]
        ]),
        np.array([
            [0.47494013, 1.18606945],
            [0.63088552, 1.23456949],
            [1.5558438 , 1.2987543 ],
            [1.30325598, 1.11885225],
            [1.0571391 , 0.49725743],
            [1.39526239, 0.12045094]
        ]),
        np.array([
            [1.31572391, 1.19934625],
            [0.78871637, 0.7389556 ],
            [1.32843768, 1.20346843],
            [2.33785591, 0.81096949],
            [1.79587589, 0.39503149],
            [1.29297762, 0.62190168],
            [1.67912865, 0.74657579],
            [0.93792053, 0.5322011 ],
            [0.47690875, 0.33112542]
        ])
    ])