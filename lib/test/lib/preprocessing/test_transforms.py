import pytest, numpy as np
from sequentia.preprocessing.transforms import *
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

# ======== #
# Equalize #
# ======== #

def test_equalize_single_odd():
    """Equalize a single odd-length observation sequence"""
    assert_equal(Equalize()(X_odd), np.array([
        [0.56804456, 0.92559664],
        [0.07103606, 0.0871293 ],
        [0.0202184 , 0.83261985],
        [0.77815675, 0.87001215],
        [0.97861834, 0.79915856],
        [0.46147936, 0.78052918],
        [0.11827443, 0.63992102]
    ]))

def test_equalize_single_even():
    """Equalize a single even-length observation sequence"""
    assert_equal(Equalize()(X_even), np.array([
        [0.5488135 , 0.71518937],
        [0.60276338, 0.54488318],
        [0.4236548 , 0.64589411],
        [0.43758721, 0.891773  ],
        [0.96366276, 0.38344152],
        [0.79172504, 0.52889492]
    ]))

def test_equalize_multiple():
    """Equalize multiple observation sequences"""
    assert_all_equal(Equalize()(Xs), [
        np.array([
            [0.14335329, 0.94466892],
            [0.52184832, 0.41466194],
            [0.26455561, 0.77423369],
            [0.        , 0.        ],
            [0.        , 0.        ],
            [0.        , 0.        ],
            [0.        , 0.        ],
            [0.        , 0.        ],
            [0.        , 0.        ]
        ]),
        np.array([
            [0.91230066, 1.1368679 ],
            [0.0375796 , 1.23527099],
            [1.22419145, 1.23386799],
            [1.88749616, 1.3636406 ],
            [0.7190158 , 0.87406391],
            [1.39526239, 0.12045094],
            [0.        , 0.        ],
            [0.        , 0.        ],
            [0.        , 0.        ]
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

def test_equalize_multiple_with_longer():
    """Fit an equalizing transformation and equalize a longer sequence"""
    transform = Equalize()
    transform.fit(Xs)
    assert_equal(transform(rng.random((15, 2))), np.array([
        [0.65632959, 0.13818295],
        [0.19658236, 0.36872517],
        [0.82099323, 0.09710128],
        [0.83794491, 0.09609841],
        [0.97645947, 0.4686512 ],
        [0.97676109, 0.60484552],
        [0.73926358, 0.03918779],
        [0.28280696, 0.12019656],
        [0.2961402 , 0.11872772]
    ]))

# ========= #
# TrimZeros #
# ========= #

def test_trim_zeros_single():
    """Trim a single zero-padded observation sequence"""
    assert_equal(TrimZeros()(X_padded), np.array([
        [0.5488135 , 0.71518937],
        [0.60276338, 0.54488318],
        [0.4236548 , 0.64589411],
        [0.43758721, 0.891773  ],
        [0.96366276, 0.38344152],
        [0.79172504, 0.52889492]
    ]))

def test_trim_zeros_multiple():
    """Trim multiple zero-padded observation sequences"""
    assert_all_equal(TrimZeros()(Xs_padded), [
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

# =========== #
# MinMaxScale #
# =========== #

def test_min_max_scale_single_independent_even_0_1():
    """Min-max scale (independently) a single even-length observation sequence into range [0, 1]"""
    assert_equal(MinMaxScale(scale=(0, 1), independent=True)(X_even), np.array([
        [0.23177196, 0.65262109],
        [0.33167766, 0.31759132],
        [0.        , 0.51630207],
        [0.02580038, 1.        ],
        [1.        , 0.        ],
        [0.6816015 , 0.28613888]
    ]))

def test_min_max_scale_single_independent_odd_0_1():
    """Min-max scale (independently) a single odd-length observation sequence into range [0, 1]"""
    assert_equal(MinMaxScale(scale=(0, 1), independent=True)(X_odd), np.array([
        [0.57160496, 1.        ],
        [0.05302344, 0.        ],
        [0.        , 0.88911101],
        [0.79083723, 0.93370703],
        [1.        , 0.84920334],
        [0.46041422, 0.82698496],
        [0.10231222, 0.65928832]
    ]))

def test_min_max_scale_single_non_independent_even_0_1():
    """Min-max scale (non-independently) a single even-length observation sequence into range [0, 1]"""
    assert_equal(MinMaxScale(scale=(0, 1), independent=False)(X_even), np.array([
        [0.23177196, 0.65262109],
        [0.33167766, 0.31759132],
        [0.        , 0.51630207],
        [0.02580038, 1.        ],
        [1.        , 0.        ],
        [0.6816015 , 0.28613888]
    ]))

def test_min_max_scale_single_non_independent_odd_0_1():
    """Min-max scale (non-independently) a single odd-length observation sequence into range [0, 1]"""
    assert_equal(MinMaxScale(scale=(0, 1), independent=False)(X_odd), np.array([
        [0.57160496, 1.        ],
        [0.05302344, 0.        ],
        [0.        , 0.88911101],
        [0.79083723, 0.93370703],
        [1.        , 0.84920334],
        [0.46041422, 0.82698496],
        [0.10231222, 0.65928832]
    ]))

def test_min_max_scale_multiple_independent_0_1():
    """Min-max scale (independently) multiple observation sequences into range [0, 1]"""
    assert_all_equal(MinMaxScale(scale=(0, 1), independent=True)(Xs), [
        np.array([
            [0.        , 1.        ],
            [1.        , 0.        ],
            [0.3202217 , 0.67842833]
        ]),
        np.array([
            [0.47284352, 0.81758801],
            [0.        , 0.89674174],
            [0.64144074, 0.89561319],
            [1.        , 1.        ],
            [0.36836051, 0.60619308],
            [0.73391569, 0.        ]
        ]),
        np.array([
            [0.61224322, 1.        ],
            [0.06198784, 0.0472772 ],
            [0.18863994, 0.46019901],
            [0.49581032, 0.59191138],
            [1.        , 0.        ],
            [0.06017231, 0.10423044],
            [0.59577551, 0.26600183],
            [0.37055656, 0.25040893],
            [0.        , 0.01465078]
        ])
    ])

def test_min_max_scale_multiple_non_independent_0_1():
    """Min-max scale (non-independently) multiple observation sequences into range [0, 1]"""
    assert_all_equal(MinMaxScale(scale=(0, 1), independent=False)(Xs), [
        np.array([
            [0.03613055, 0.43575693],
            [0.1654182 , 0.15554682],
            [0.07753126, 0.3456493 ]
        ]),
        np.array([
            [0.29879028, 0.53737088],
            [0.        , 0.58939575],
            [0.40532702, 0.58865399],
            [0.63190096, 0.65726365],
            [0.23276736, 0.39842868],
            [0.46376203, 0.        ]
        ]),
        np.array([
            [0.67043294, 1.        ],
            [0.20275306, 0.14080529],
            [0.31039878, 0.51319087],
            [0.57147285, 0.63197314],
            [1.        , 0.09816926],
            [0.20120999, 0.19216748],
            [0.6564365 , 0.33805788],
            [0.46501562, 0.32399573],
            [0.15006759, 0.11138178]
        ])
    ])

def test_min_max_scale_single_independent_even_m5_5():
    """Min-max scale (independently) a single even-length observation sequence into range [-5, 5]"""
    assert_equal(MinMaxScale(scale=(-5, 5), independent=True)(X_even), np.array([
        [-2.68228038,  1.52621093],
        [-1.6832234 , -1.82408684],
        [-5.        ,  0.16302066],
        [-4.74199618,  5.        ],
        [ 5.        , -5.        ],
        [ 1.81601504, -2.1386112 ]
    ]))

def test_min_max_scale_single_independent_odd_m5_5():
    """Min-max scale (independently) a single odd-length observation sequence into range [-5, 5]"""
    assert_equal(MinMaxScale(scale=(-5, 5), independent=True)(X_odd), np.array([
        [ 0.71604962,  5.        ],
        [-4.46976561, -5.        ],
        [-5.        ,  3.89111014],
        [ 2.90837226,  4.3370703 ],
        [ 5.        ,  3.4920334 ],
        [-0.39585778,  3.26984958],
        [-3.97687777,  1.59288318]
    ]))

def test_min_max_scale_single_non_independent_even_m5_5():
    """Min-max scale (non-independently) a single even-length observation sequence into range [-5, 5]"""
    assert_equal(MinMaxScale(scale=(-5, 5), independent=False)(X_even), np.array([
        [-2.68228038,  1.52621093],
        [-1.6832234 , -1.82408684],
        [-5.        ,  0.16302066],
        [-4.74199618,  5.        ],
        [ 5.        , -5.        ],
        [ 1.81601504, -2.1386112 ]
    ]))

def test_min_max_scale_single_non_independent_odd_m5_5():
    """Min-max scale (non-independently) a single odd-length observation sequence into range [-5, 5]"""
    assert_equal(MinMaxScale(scale=(-5, 5), independent=False)(X_odd), np.array([
        [ 0.71604962,  5.        ],
        [-4.46976561, -5.        ],
        [-5.        ,  3.89111014],
        [ 2.90837226,  4.3370703 ],
        [ 5.        ,  3.4920334 ],
        [-0.39585778,  3.26984958],
        [-3.97687777,  1.59288318]
    ]))

def test_min_max_scale_multiple_independent_m5_5():
    """Min-max scale (independently) multiple observation sequences into range [-5, 5]"""
    assert_all_equal(MinMaxScale(scale=(-5, 5), independent=True)(Xs), [
        np.array([
            [-5.        ,  5.        ],
            [ 5.        , -5.        ],
            [-1.79778296,  1.78428332]
        ]),
        np.array([
            [-0.27156476,  3.17588009],
            [-5.        ,  3.96741737],
            [ 1.4144074 ,  3.95613188],
            [ 5.        ,  5.        ],
            [-1.31639493,  1.06193079],
            [ 2.33915693, -5.        ]
        ]),
        np.array([
            [ 1.1224322 ,  5.        ],
            [-4.38012161, -4.52722802],
            [-3.11360062, -0.39800995],
            [-0.04189682,  0.91911381],
            [ 5.        , -5.        ],
            [-4.39827687, -3.95769556],
            [ 0.95775509, -2.33998174],
            [-1.29443438, -2.49591067],
            [-5.        , -4.85349222]
        ])
    ])

def test_min_max_scale_multiple_non_independent_m5_5():
    """Min-max scale (non-independently) multiple observation sequences into range [-5, 5]"""
    assert_all_equal(MinMaxScale(scale=(-5, 5), independent=False)(Xs), [
        np.array([
            [-4.63869454, -0.64243065],
            [-3.34581798, -3.44453183],
            [-4.22468741, -1.543507  ]
        ]),
        np.array([
            [-2.01209722,  0.37370879],
            [-5.        ,  0.89395747],
            [-0.94672978,  0.88653993],
            [ 1.31900964,  1.5726365 ],
            [-2.67232641, -1.01571325],
            [-0.36237966, -5.        ]
        ]),
        np.array([
            [ 1.70432945,  5.        ],
            [-2.9724694 , -3.5919471 ],
            [-1.89601215,  0.13190869],
            [ 0.71472846,  1.31973138],
            [ 5.        , -4.01830741],
            [-2.98790014, -3.07832522],
            [ 1.56436503, -1.61942117],
            [-0.3498438 , -1.76004267],
            [-3.49932413, -3.88618219]
        ])
    ])

# ====== #
# Center #
# ====== #

def test_center_single_independent_even():
    """Center (independently) a single even-length observation sequence"""
    assert_equal(Center(independent=True)(X_even), np.array([
        [-0.07922094, 0.09684335 ],
        [-0.02527107, -0.07346283],
        [-0.20437965, 0.0275481  ],
        [-0.19044724, 0.27342698 ],
        [0.33562831 , -0.2349045 ],
        [0.16369059 , -0.0894511 ]
    ]))

def test_center_single_independent_odd():
    """Center (independently) a single odd-length observation sequence"""
    assert_equal(Center(independent=True)(X_odd), np.array([
        [0.14006915 , 0.2206014  ],
        [-0.35693936, -0.61786594],
        [-0.40775702, 0.1276246  ],
        [0.35018134 , 0.16501691 ],
        [0.55064293 , 0.09416332 ],
        [0.03350395 , 0.07553393 ],
        [-0.30970099, -0.06507422]
    ]))

def test_center_single_non_independent_even():
    """Center (non-independently) a single even-length observation sequence"""
    assert_equal(Center(independent=False)(X_even), np.array([
        [-0.07922094, 0.09684335 ],
        [-0.02527107, -0.07346283],
        [-0.20437965, 0.0275481  ],
        [-0.19044724, 0.27342698 ],
        [0.33562831 , -0.2349045 ],
        [0.16369059 , -0.0894511 ]
    ]))

def test_center_single_non_independent_odd():
    """Center (non-independently) a single odd-length observation sequence"""
    assert_equal(Center(independent=False)(X_odd), np.array([
        [0.14006915 , 0.2206014  ],
        [-0.35693936, -0.61786594],
        [-0.40775702, 0.1276246  ],
        [0.35018134 , 0.16501691 ],
        [0.55064293 , 0.09416332 ],
        [0.03350395 , 0.07553393 ],
        [-0.30970099, -0.06507422]
    ]))

def test_center_multiple_independent():
    """Center (independently) multiple observation sequences"""
    assert_all_equal(Center(independent=True)(Xs), [
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

def test_center_multiple_non_independent():
    """Center (non-independently) multiple observation sequences"""
    assert_all_equal(Center(independent=False)(Xs), [
        np.array([
            [-0.95780473,  0.08257468],
            [-0.5793097 , -0.44743229],
            [-0.8366024 , -0.08786055]
        ]),
        np.array([
            [-0.18885735,  0.27477366],
            [-1.06357842,  0.37317676],
            [ 0.12303343,  0.37177376],
            [ 0.78633814,  0.50154636],
            [-0.38214222,  0.01196967],
            [ 0.29410437, -0.74164329]
        ]),
        np.array([
            [ 0.89914213,  1.14981937],
            [-0.47001033, -0.47531534],
            [-0.15487296,  0.22903808],
            [ 0.60943229,  0.45371031],
            [ 1.8639635 , -0.5559598 ],
            [-0.47452775, -0.37816568],
            [ 0.85816696, -0.10221943],
            [ 0.2977743 , -0.12881746],
            [-0.62424927, -0.53096881]
        ])
    ])

# =========== #
# Standardize #
# =========== #

def test_standardize_single_independent_even():
    """Standardize (independently) a single even-length observation sequence"""
    assert_equal(Standardize(independent=True)(X_even), np.array([
        [-0.40964472,  0.60551094],
        [-0.13067455, -0.45932478],
        [-1.05682966,  0.17224387],
        [-0.98478635,  1.70959629],
        [ 1.73550526, -1.46873528],
        [ 0.84643002, -0.55929105]
    ]))

def test_standardize_single_independent_odd():
    """Standardize (independently) a single odd-length observation sequence"""
    assert_equal(Standardize(independent=True)(X_odd), np.array([
        [ 0.40527155,  0.83146609],
        [-1.03275681, -2.32879115],
        [-1.17979099,  0.48102837],
        [ 1.01320338,  0.62196325],
        [ 1.59321247,  0.35490986],
        [ 0.09693924,  0.28469405],
        [-0.89607884, -0.24527047]
    ]))

def test_standardize_single_non_independent_even():
    """Standardize (non-independently) a single even-length observation sequence"""
    assert_equal(Standardize(independent=False)(X_even), np.array([
        [-0.40964472,  0.60551094],
        [-0.13067455, -0.45932478],
        [-1.05682966,  0.17224387],
        [-0.98478635,  1.70959629],
        [ 1.73550526, -1.46873528],
        [ 0.84643002, -0.55929105]
    ]))

def test_standardize_single_non_independent_odd():
    """Standardize (non-independently) a single odd-length observation sequence"""
    assert_equal(Standardize(independent=False)(X_odd), np.array([
        [ 0.40527155,  0.83146609],
        [-1.03275681, -2.32879115],
        [-1.17979099,  0.48102837],
        [ 1.01320338,  0.62196325],
        [ 1.59321247,  0.35490986],
        [ 0.09693924,  0.28469405],
        [-0.89607884, -0.24527047]
    ]))

def test_standardize_multiple_independent():
    """Standardize (independently) multiple observation sequences"""
    assert_all_equal(Standardize(independent=True)(Xs), [
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

def test_standardize_multiple_non_independent():
    """Standardize (non-independently) multiple observation sequences"""
    assert_all_equal(Standardize(independent=False)(Xs), [
        np.array([
            [-1.26465246,  0.17656713],
            [-0.76490062, -0.95673193],
            [-1.10462107, -0.18786974]
        ]),
        np.array([
            [-0.24936076,  0.58754082],
            [-1.4043124 ,  0.7979534 ],
            [ 0.16244911,  0.7949534 ],
            [ 1.03825387,  1.07244252],
            [-0.50456745,  0.02559442],
            [ 0.38832531, -1.58583505]]),
        np.array([
            [ 1.18719638,  2.45862652],
            [-0.6205855 , -1.01635347],
            [-0.20448894,  0.4897457 ],
            [ 0.80467346,  0.97015602],
            [ 2.46111337, -1.18879326],
            [-0.62655014, -0.80862107],
            [ 1.13309417, -0.21857294],
            [ 0.39317096, -0.27544676],
            [-0.82423729, -1.13535572]
        ])
    ])

# ========== #
# Downsample #
# ========== #

def test_downsample_single_large_factor():
    """Downsample a single observation sequence with a downsample factor that is too large"""
    with pytest.raises(ValueError) as e:
        Downsample(factor=7)(X_even)
    assert str(e.value) == 'Expected downsample factor to be no greater than the number of frames'

def test_downsample_single_decimate_max():
    """Downsample a single observation sequence with decimation and the maximum downsample factor"""
    assert_equal(Downsample(factor=6, method='decimate')(X_even), np.array([
        [0.548814, 0.715189]
    ]))

def test_downsample_single_decimate():
    """Downsample a single observation sequence with decimation"""
    assert_equal(Downsample(factor=3, method='decimate')(X_odd), np.array([
        [0.56804456, 0.92559664],
        [0.77815675, 0.87001215],
        [0.11827443, 0.63992102]
    ]))

def test_downsample_single_mean_max():
    """Downsample a single observation sequence with mean downsamping and the maximum downsample factor"""
    assert_equal(Downsample(factor=6, method='mean')(X_even), np.array([
        [0.62803445, 0.61834602]
    ]))

def test_downsample_single_mean():
    """Downsample a single observation sequence with mean downsampling"""
    assert_equal(Downsample(factor=3, method='mean')(X_odd), np.array([
        [0.21976634, 0.61511526],
        [0.73941815, 0.81656663],
        [0.11827443, 0.63992102]
    ]))

def test_downsample_multiple_large_factor():
    """Downsample multiple observation sequences with a downsample factor that is too large"""
    with pytest.raises(ValueError) as e:
        Downsample(factor=4)(Xs)
    assert str(e.value) == 'Expected downsample factor to be no greater than the number of frames in the shortest sequence'

def test_downsample_multiple_decimate_max():
    """Downsample multiple observation sequences with decimation and the maximum downsample factor"""
    assert_all_equal(Downsample(factor=3, method='decimate')(Xs), [
        np.array([
            [0.14335329, 0.94466892]
        ]),
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
    assert_all_equal(Downsample(factor=2, method='decimate')(Xs), [
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

def test_downsample_multiple_mean_max():
    """Downsample multiple observation sequences with mean downsampling and the maximum downsample factor"""
    assert_all_equal(Downsample(factor=3, method='mean')(Xs), [
        np.array([
            [0.30991907, 0.71118818]
        ]),
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

def test_downsample_multiple_mean():
    """Downsample multiple observation sequences with mean downsampling"""
    assert_all_equal(Downsample(factor=2, method='mean')(Xs), [
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

# ====== #
# Filter #
# ====== #

def test_filter_single_large_window():
    """Filter a single observation sequence with a window size that is too large"""
    with pytest.raises(ValueError) as e:
        Filter(window_size=7)(X_even)
    assert str(e.value) == 'Expected window size to be no greater than the number of frames'

def test_filter_single_median_max():
    """Filter a single observation sequence with median filtering and the maximum window size"""
    assert_equal(Filter(window_size=6, method='median')(X_even), np.array([
        [0.49320036, 0.68054174],
        [0.5488135 , 0.64589411],
        [0.57578844, 0.59538865],
        [0.60276338, 0.54488318],
        [0.61465612, 0.58739452],
        [0.79172504, 0.52889492]
    ]))

def test_filter_single_median():
    """Filter a single observation sequence with median filtering"""
    assert_equal(Filter(window_size=3, method='median')(X_odd), np.array([
        [0.31954031, 0.50636297],
        [0.07103606, 0.83261985],
        [0.07103606, 0.83261985],
        [0.77815675, 0.83261985],
        [0.77815675, 0.79915856],
        [0.46147936, 0.78052918],
        [0.28987689, 0.7102251 ]
    ]))

def test_filter_single_mean_max():
    """Filter a single observation sequence with mean filtering and the maximum window size"""
    assert_equal(Filter(window_size=6, method='mean')(X_even), np.array([
        [0.50320472, 0.69943492],
        [0.59529633, 0.63623624],
        [0.62803445, 0.61834602],
        [0.64387864, 0.59897735],
        [0.65415745, 0.61250089],
        [0.73099167, 0.60136981]
    ]))

def test_filter_single_mean():
    """Filter a single observation sequence with mean filtering"""
    assert_equal(Filter(window_size=3, method='mean')(X_odd), np.array([
        [0.31954031, 0.50636297],
        [0.21976634, 0.61511526],
        [0.28980374, 0.5965871 ],
        [0.59233116, 0.83393019],
        [0.73941815, 0.81656663],
        [0.51945738, 0.73986959],
        [0.28987689, 0.7102251 ]
    ]))

def test_filter_multiple_large_window():
    """Filter multiple observation sequences with a window size that is too large"""
    with pytest.raises(ValueError) as e:
        Filter(window_size=4)(Xs)
    assert str(e.value) == 'Expected window size to be no greater than the number of frames in the shortest sequence'

def test_filter_multiple_median_max():
    """Filter multiple observation sequences with median filtering and the maximum window size"""
    assert_all_equal(Filter(window_size=3, method='median')(Xs), [
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

def test_filter_multiple_median():
    """Filter multiple observation sequences with median filtering"""
    assert_all_equal(Filter(window_size=2, method='median')(Xs), [
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

def test_filter_multiple_mean_max():
    """Filter multiple observation sequences with mean filtering and the maximum window size"""
    assert_all_equal(Filter(window_size=3, method='mean')(Xs), [
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

def test_filter_multiple_mean():
    """Filter multiple observation sequences with mean filtering"""
    assert_all_equal(Filter(window_size=2, method='mean')(Xs), [
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