from enum import Enum, unique
from typing import Optional, Union, Callable, Literal

import numpy as np
from pydantic import NegativeInt ,PositiveInt, confloat

from ...utils import validate_params, requires_fit, override_params, Validator

# TODO: Make uninstantiatable

@unique
class _Weighting(Enum):
    UNIFORM = 'uniform'

class KNNType:
    Weighting = _Weighting

class KNNConfig(Validator):
    k: PositiveInt = 1,
    weighting: Union[Literal[KNNType.Weighting.UNIFORM], Callable] = KNNType.Weighting.UNIFORM,
    window: confloat(ge=0, le=1) = 1,
    independent: bool = False,
    use_c: bool = False,
    n_jobs: Union[NegativeInt, PositiveInt] = 1,
    random_state: Optional[Union[int, np.random.RandomState]] = None

    class Config:
        arbitrary_types_allowed = True

class KNNMixin:
    # Expects:
    # - X, y
    # all constructor params

    @requires_fit
    @validate_params(using=KNNConfig)
    @override_params(['k', 'weighting', 'window', 'independent'])
    def query_neighbors(self, X, lengths=None, sorted=True, **kwargs): # kwargs defaults to init params
        """Queries the k nearest neighbors in the training set for each sequence in X"""
        # check_X_lengths (if lengths is none, use whole input as single sequence)

        # NOTE: This isn't used in predict() as we have to maintain correct order of the top k here
        #   So will have to do sort after argpartition, which may be slow

        # self.compute_distance_matrix()
        #

        # Indices of k nearest in self.X for each element in X (M x K)
        # Distances of k nearest in self.X for each elemnet in X (M x K)
        #
        pass

    @requires_fit
    @validate_params(using=KNNConfig)
    @override_params(['window', 'independent'])
    def compute_distance_matrix(self, X, lengths, **kwargs):
        # check_X_lengths (if lengths is none, use whole input as single sequence)
        # X, lengths = check_X_lengths(X, lengths)

        # If len(lengths) = 1 paralellize over columns for single sequence and still produce 2d dist mat
        # np.atleast_2d

        # TODO: Paralellize over rows (and maybe columns)
        mat = np.zeros((len(lengths), len(self.lengths)))
        for i, x1 in enumerate(iter_X_lengths(X, lengths)):
            row = np.zeros(len(self.lengths))
            for j, x2 in enumerate(iter_X_lengths(self.X, self.lengths)):
                row[j] = self.dtw_(x1, x2)
            mat[i] = row

        return mat

    # def _find_k_nearest(self, distances):
    #     """Returns"""

    # def _find_k_nearest(self, distances):
    #     """Returns the labels and weightings (or scores) of the k-nearest neighbors"""
    #     idx = np.argpartition(distances, self._k)[:self._k]
    #     return self._y_[idx], self._weighting(distances[idx])

    # def _find_max_labels(self, nearest_labels, nearest_scores):
    #     """Returns the mode label(s) of the k nearest neighbors.
    #     Vectorization from: https://stackoverflow.com/a/49239335
    #     """
    #     # Sort the labels in ascending order (and sort distances in the same order)
    #     sorted_labels_idx = nearest_labels.argsort()
    #     sorted_labels, sorted_scores = nearest_labels[sorted_labels_idx], nearest_scores[sorted_labels_idx]
    #     # Identify the indices where the sorted labels change (so we can group by labels)
    #     change_idx = np.concatenate(([0], np.nonzero(np.diff(sorted_labels))[0] + 1))
    #     # Calculate the total score for each label
    #     label_scores = np.add.reduceat(sorted_scores, change_idx)
    #     # Find the change index of the maximum score(s)
    #     max_score_idx = change_idx[self._multi_argmax(label_scores)]
    #     # Map the change index of the maximum scores back to the actual label(s)
    #     max_labels = sorted_labels[max_score_idx]
    #     # Store class scores
    #     scores = np.full(len(self.classes_), -np.inf)
    #     scores[sorted_labels[change_idx]] = label_scores
    #     # Map the change index of the maximum scores back to the actual label(s), and return scores
    #     return max_labels, scores

    # def _predict(self, x1, verbose=False):
    #     """Makes a prediction for a single observation sequence."""
    #     # Calculate DTW distances between x1 and all other sequences
    #     distances = np.array([self._dtw(x1, x2) for x2 in tqdm(self._X_, desc='Calculating distances', disable=not(verbose))])
    #     # Find the k-nearest neighbors by DTW distance
    #     nearest_labels, nearest_scores = self._find_k_nearest(distances)
    #     # Out of the k-nearest neighbors, find the label(s) which had the highest total weighting
    #     max_labels, scores = self._find_max_labels(nearest_labels, nearest_scores)
    #     # Randomly pick from the set of labels with the maximum label score
    #     label = self._random_state.choice(max_labels, size=1)
    #     # Combine the label with the scores
    #     return np.concatenate((label, scores))

    # def _dtw_1d(self, a, b, window):
    #     pass

    # def _dtwi(self, A, B):
    #     pass

    # def _dtwd(self, A, B):
    #     pass

    # def _multi_argmax(self, arr):
    #     pass

