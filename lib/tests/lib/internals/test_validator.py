import pytest
import numpy as np
from numpy.testing import assert_array_equal
from sequentia.internals import Validator

val = Validator()

# ================================= #
# Validator.observation_sequences() #
# ================================= #

def test_single_observation_sequence_with_single():
    """Single observation sequence with allow_single=True"""
    x = np.arange(8).reshape(-1, 2)
    assert_array_equal(x, val.observation_sequences(x, allow_single=True))

def test_single_observation_sequence_1d_flat_with_single():
    """Single flat 1D observation sequence with allow_single=True"""
    x = np.arange(4)
    with pytest.raises(ValueError) as e:
        val.observation_sequences(x, allow_single=True)
    assert str(e.value) == 'Observation sequence must be two-dimensional'

def test_single_observation_sequence_1d_with_single():
    """Single non-flat 1D observation sequence with allow_single=True"""
    x = np.arange(4).reshape(-1, 1)
    assert_array_equal(x, val.observation_sequences(x, allow_single=True))

def test_single_observation_sequence_wrong_type_with_single():
    """Single observation sequence with wrong type and allow_single=True"""
    x = 1
    with pytest.raises(TypeError) as e:
        val.observation_sequences(x, allow_single=True)
    assert str(e.value) == 'Expected an individual observation sequence or a list of multiple observation sequences, each of type numpy.ndarray'

def test_multiple_observation_sequences_with_single():
    """Multiple observation sequences with allow_single=True"""
    X = [np.arange(8).reshape(-1, 2), np.arange(12).reshape(-1, 2)]
    assert X == val.observation_sequences(X, allow_single=True)

def test_multiple_observation_sequences_diff_dims_with_single():
    """Multiple observation sequences with different dimensionality and allow_single=True"""
    X = [np.arange(8).reshape(-1, 2), np.arange(12).reshape(-1, 3)]
    with pytest.raises(ValueError) as e:
        val.observation_sequences(X, allow_single=True)
    assert str(e.value) == 'Each observation sequence must have the same dimensionality'

def test_multiple_observation_sequences_1d_some_flat_with_single():
    """Multiple 1D (flat and non-flat) observation sequences with allow_single=True"""
    X = [np.arange(4).reshape(-1, 1), np.arange(8)]
    with pytest.raises(ValueError) as e:
        val.observation_sequences(X, allow_single=True)
    assert str(e.value) == 'Each observation sequence must be two-dimensional'

def test_multiple_observation_sequences_1d_all_flat_with_single():
    """Multiple flat 1D observation sequences with allow_single=True"""
    X = [np.arange(4), np.arange(8)]
    with pytest.raises(ValueError) as e:
        val.observation_sequences(X, allow_single=True)
    assert str(e.value) == 'Each observation sequence must be two-dimensional'

def test_multiple_observation_sequences_1d_with_single():
    """Multiple 1D observation sequences with allow_single=True"""
    X = [np.arange(8).reshape(-1, 1), np.arange(12).reshape(-1, 1)]
    assert X == val.observation_sequences(X, allow_single=True)

def test_multiple_observation_sequences_some_wrong_type_with_single():
    """Multiple observation sequences with different types and allow_single=True"""
    X = [np.arange(4).reshape(-1, 1), np.arange(8).reshape(-1, 1), 3]
    with pytest.raises(TypeError) as e:
        val.observation_sequences(X, allow_single=True)
    assert str(e.value) == 'Each observation sequence must be a numpy.ndarray'

def test_multiple_observation_sequences_all_wrong_type_with_single():
    """Multiple observation sequences with the wrong type with allow_single=True"""
    X = [1, 2, 3, 4]
    with pytest.raises(TypeError) as e:
        val.observation_sequences(X, allow_single=True)
    assert str(e.value) == 'Each observation sequence must be a numpy.ndarray'

def test_multiple_observation_sequences_wrong_list_type_with_single():
    """Multiple observation sequences with the wrong list type with allow_single=True"""
    X = [[1, 2, 3, 4], [4, 3, 2, 1]]
    with pytest.raises(TypeError) as e:
        val.observation_sequences(X, allow_single=True)
    assert str(e.value) == 'Each observation sequence must be a numpy.ndarray'

def test_single_observation_sequence_without_single():
    """Single observation sequence with allow_single=False"""
    x = np.arange(8).reshape(-1, 2)
    with pytest.raises(TypeError) as e:
        val.observation_sequences(x, allow_single=False)
    assert str(e.value) == 'Expected a list of observation sequences, each of type numpy.ndarray'

def test_single_observation_sequence_1d_flat_without_single():
    """Single flat 1D observation sequence with allow_single=False"""
    x = np.arange(4)
    with pytest.raises(TypeError) as e:
        val.observation_sequences(x, allow_single=False)
    assert str(e.value) == 'Expected a list of observation sequences, each of type numpy.ndarray'

def test_single_observation_sequence_1d_without_single():
    """Single non-flat 1D observation sequence with allow_single=False"""
    x = np.arange(4).reshape(-1, 1)
    with pytest.raises(TypeError) as e:
        val.observation_sequences(x, allow_single=False)
    assert str(e.value) == 'Expected a list of observation sequences, each of type numpy.ndarray'

def test_single_observation_sequence_wrong_type_without_single():
    """Single observation sequence with wrong type and allow_single=False"""
    x = 1
    with pytest.raises(TypeError) as e:
        val.observation_sequences(x, allow_single=False)
    assert str(e.value) == 'Expected a list of observation sequences, each of type numpy.ndarray'

def test_multiple_observation_sequences_without_single():
    """Multiple observation sequences with allow_single=False"""
    X = [np.arange(8).reshape(-1, 2), np.arange(12).reshape(-1, 2)]
    assert X == val.observation_sequences(X, allow_single=False)

def test_multiple_observation_sequences_diff_dims_without_single():
    """Multiple observation sequences with different dimensionality and allow_single=False"""
    X = [np.arange(8).reshape(-1, 2), np.arange(12).reshape(-1, 3)]
    with pytest.raises(ValueError) as e:
        val.observation_sequences(X, allow_single=False)
    assert str(e.value) == 'Each observation sequence must have the same dimensionality'

def test_multiple_observation_sequences_1d_some_flat_without_single():
    """Multiple 1D (flat and non-flat) observation sequences with allow_single=False"""
    X = [np.arange(4).reshape(-1, 1), np.arange(8)]
    with pytest.raises(ValueError) as e:
        val.observation_sequences(X, allow_single=False)
    assert str(e.value) == 'Each observation sequence must be two-dimensional'

def test_multiple_observation_sequences_1d_all_flat_without_single():
    """Multiple flat 1D observation sequences with allow_single=False"""
    X = [np.arange(4), np.arange(8)]
    with pytest.raises(ValueError) as e:
        val.observation_sequences(X, allow_single=False)
    assert str(e.value) == 'Each observation sequence must be two-dimensional'

def test_multiple_observation_sequences_1d_without_single():
    """Multiple 1D observation sequences with allow_single=False"""
    X = [np.arange(8).reshape(-1, 1), np.arange(12).reshape(-1, 1)]
    assert X == val.observation_sequences(X, allow_single=False)

def test_multiple_observation_sequence_some_wrong_type_without_single():
    """Multiple observation sequences with different types and allow_single=False"""
    X = [np.arange(4).reshape(-1, 1), np.arange(8).reshape(-1, 1), 3]
    with pytest.raises(TypeError) as e:
        val.observation_sequences(X, allow_single=False)
    assert str(e.value) == 'Each observation sequence must be a numpy.ndarray'

def test_multiple_observation_sequence_wrong_type_without_single():
    """Multiple observation sequences with the wrong type with allow_single=False"""
    X = [1, 2, 3, 4]
    with pytest.raises(TypeError) as e:
        val.observation_sequences(X, allow_single=False)
    assert str(e.value) == 'Each observation sequence must be a numpy.ndarray'

def test_multiple_observation_sequence_wrong_list_type_without_single():
    """Multiple observation sequences with the wrong list type with allow_single=False"""
    X = [[1, 2, 3, 4], [4, 3, 2, 1]]
    with pytest.raises(TypeError) as e:
        val.observation_sequences(X, allow_single=False)
    assert str(e.value) == 'Each observation sequence must be a numpy.ndarray'

# ============================================ #
# Validator.observation_sequences_and_labels() #
# ============================================ #

def test_observation_sequences_and_labels_same_length():
    """Observation sequences and labels with the same length."""
    X = [np.arange(8).reshape(-1, 1), np.arange(12).reshape(-1, 1)]
    y = ['c1', 'c2']
    assert val.observation_sequences_and_labels(X, y) == (X, y)

def test_observation_sequences_and_labels_diff_length():
    """Observation sequences and labels with different lengths."""
    X = [np.arange(8).reshape(-1, 1), np.arange(12).reshape(-1, 1)]
    y = ['c1', 'c2', 'c1']
    with pytest.raises(ValueError) as e:
        val.observation_sequences_and_labels(X, y)
    assert str(e.value) == 'Expected the same number of observation sequences and labels'

# =================== #
# Validator.integer() #
# =================== #

def test_integer_with_correct_type():
    """Integer type"""
    assert val.integer(1, desc='test') == 1

def test_integer_with_float():
    """Float type"""
    with pytest.raises(TypeError) as e:
        val.integer(1., desc='test')
    assert str(e.value) == 'Expected test to be an integer'

def test_integer_with_wrong_type():
    """Incorrect type"""
    with pytest.raises(TypeError) as e:
        val.integer('a', desc='test')
    assert str(e.value) == 'Expected test to be an integer'

# ================== #
# Validator.string() #
# ================== #

def test_string_with_correct_type():
    """String type"""
    assert val.string('test', desc='test') == 'test'

def test_string_with_wrong_type():
    """Incorrect type"""
    with pytest.raises(TypeError) as e:
        val.string(1, desc='test')
    assert str(e.value) == 'Expected test to be a string'

# =================== #
# Validator.boolean() #
# =================== #

def test_boolean_with_correct_type():
    """Boolean type"""
    assert val.boolean(True, desc='test') == True

def test_boolean_with_wrong_type():
    """Incorrect type"""
    with pytest.raises(TypeError) as e:
        val.boolean(1, desc='test')
    assert str(e.value) == 'Expected test to be a boolean'

# ================== #
# Validator.one_of() #
# ================== #

def test_one_of_correct_with_multiple_types():
    """List of multiple types with a correct input"""
    assert val.one_of(2, [True, 'test', 2], desc='test') == 2

def test_one_of_incorrect_with_multiple_types():
    """List of multiple types with an incorrect input"""
    with pytest.raises(ValueError) as e:
        val.one_of(2, [True, 'test', 2.1], desc='test')
    assert str(e.value) == "Expected test to be one of [True, 'test', 2.1]"

def test_one_of_correct_with_single_type():
    """List of single type with a correct input"""
    assert val.one_of(2, [0, 1, 2], desc='test') == 2

def test_one_of_incorrect_with_single_type():
    """List of single type with an incorrect input"""
    with pytest.raises(ValueError) as e:
        val.one_of(2, [0, 1, 3], desc='test')
    assert str(e.value) == "Expected test to be one of [0, 1, 3]"

# ============================== #
# Validator.restricted_integer() #
# ============================== #

def test_restricted_integer_wrong_type_meets_condition():
    """Incorrect type that meets the condition"""
    with pytest.raises(TypeError) as e:
        val.restricted_integer('test', lambda x: len(x) == 4, 'test', 'not false')
    assert str(e.value) == 'Expected test to be an integer'

def test_restricted_integer_wrong_type_does_not_meet_condition():
    """Incorrect type that does not meet the condition"""
    with pytest.raises(TypeError) as e:
        val.restricted_integer('test', lambda x: len(x) != 4, 'test', 'not false')
    assert str(e.value) == 'Expected test to be an integer'

def test_restricted_integer_correct_type_meets_condition():
    """Correct type that meets the condition"""
    assert val.restricted_integer(1, lambda x: x > 0, 'test', 'greater than zero') == 1

def test_restricted_integer_correct_type_does_not_meet_condition():
    """Correct type that does not meet the condition"""
    with pytest.raises(ValueError) as e:
        val.restricted_integer(-1, lambda x: x > 0, 'test', 'greater than zero')
    assert str(e.value) == 'Expected test to be greater than zero'

# ============================ #
# Validator.restricted_float() #
# ============================ #

def test_restricted_integer_wrong_type_meets_condition():
    """Incorrect type that meets the condition"""
    with pytest.raises(TypeError) as e:
        val.restricted_float('test', lambda x: len(x) == 4, 'test', 'not false')
    assert str(e.value) == 'Expected test to be a float'

def test_restricted_integer_wrong_type_does_not_meet_condition():
    """Incorrect type that does not meet the condition"""
    with pytest.raises(TypeError) as e:
        val.restricted_float('test', lambda x: len(x) != 4, 'test', 'not false')
    assert str(e.value) == 'Expected test to be a float'

def test_restricted_integer_correct_type_meets_condition():
    """Correct type that meets the condition"""
    assert val.restricted_float(1.1, lambda x: x > 0, 'test', 'greater than zero') == 1.1

def test_restricted_integer_correct_type_does_not_meet_condition():
    """Correct type that does not meet the condition"""
    with pytest.raises(ValueError) as e:
        val.restricted_float(-1.1, lambda x: x > 0, 'test', 'greater than zero')
    assert str(e.value) == 'Expected test to be greater than zero'

# =========================== #
# Validator.list_of_strings() #
# =========================== #

def test_list_of_strings_wrong_type():
    """Incorrect type"""
    with pytest.raises(TypeError) as e:
        val.list_of_strings(1, 'test')
    assert str(e.value) == 'Expected test to be a list of strings'

def test_list_of_strings_wrong_list_type():
    """Correct type but wrong list element types"""
    with pytest.raises(ValueError) as e:
        val.list_of_strings([1, True, 'test'], 'test')
    assert str(e.value) == 'Expected all test to be strings'

def test_list_of_strings_correct_list_type():
    """Correct list element types"""
    assert val.list_of_strings(['a', 'b', 'c'], 'test') == ['a', 'b', 'c']