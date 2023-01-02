from typing import Optional

from sklearn.base import TransformerMixin, BaseEstimator

from sequentia.utils.validation import Array

__all__ = ['Transform']


class Transform(TransformerMixin, BaseEstimator):
    def fit_transform(
        self,
        X: Array,
        lengths: Optional[Array] = None,
    ) -> Array:
        """Fits the transformer to the sequence(s) in ``X`` and returns a transformed version of ``X``.

        :param X: Univariate or multivariate observation sequence(s).

            - Should be a single 1D or 2D array.
            - Should have length as the 1st dimension and features as the 2nd dimension.
            - Should be a concatenated sequence if multiple sequences are provided,
              with respective sequence lengths being provided in the ``lengths`` argument for decoding the original sequences.

        :param lengths: Lengths of the observation sequence(s) provided in ``X``.

            - If ``None``, then ``X`` is assumed to be a single observation sequence.
            - ``len(X)`` should be equal to ``sum(lengths)``.

        :return: The transformed data.
        """
        return self.fit(X, lengths).transform(X, lengths)
