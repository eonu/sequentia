.. _splitting:

Cross-validation splitting methods
==================================

During cross-validation, a dataset is divided into splits for training and validation. 

This can be either be done using a single basic split, or alternatively via successive 
*folds* which re-use parts of the dataset for different splits.

:mod:`sklearn.model_selection` provides such cross-validation splitting methods,
but does not support sequence data. Sequentia provides modified
versions of these methods to support sequence data.

API reference
-------------

Classes
^^^^^^^

.. autosummary::

   ~sequentia.model_selection.KFold
   ~sequentia.model_selection.StratifiedKFold
   ~sequentia.model_selection.ShuffleSplit
   ~sequentia.model_selection.StratifiedShuffleSplit
   ~sequentia.model_selection.RepeatedKFold
   ~sequentia.model_selection.RepeatedStratifiedKFold

Example
^^^^^^^

Using :class:`.GridSearchCV` with :class:`.StratifiedKFold` to 
cross-validate a :class:`.KNNClassifier` training pipeline. ::

    import numpy as np

    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import minmax_scale

    from sequentia.datasets import load_digits
    from sequentia.models import KNNClassifier
    from sequentia.preprocessing import IndependentFunctionTransformer
    from sequentia.model_selection import StratifiedKFold, GridSearchCV

    EPS: np.float32 = np.finfo(np.float32).eps

    # Define model and hyper-parameter search space
    search = GridSearchCV(
        # Create a basic pipeline with a KNNClassifier to be optimized
        estimator=Pipeline(
            [
                ("scale", IndependentFunctionTransformer(minmax_scale)),
                ("clf", KNNClassifier(use_c=True, n_jobs=-1))
            ]
        ),
        # Optimize over k, weighting function and window size
        param_grid={
            "clf__k": [1, 2, 3, 4, 5],
            "clf__weighting": [
                None, lambda x: 1 / (x + EPS), lambda x: np.exp(-x)
            ],
            "clf__window": [1.0, 0.75, 0.5, 0.25, 0.1],
        },
        # Use StratifiedKFold cross-validation
        cv=StratifiedKFold(),
        n_jobs=-1,
    )

    # Load the spoken digit dataset with a train/test set split
    data = load_digits()
    train_data, test_data = data.split(test_size=0.2, stratify=True)

    # Perform cross-validation over accuracy and retrieve the best model
    search.fit(train_data.X, train_data.y, lengths=train_data.lengths)
    clf = search.best_estimator_

    # Calculate accuracy on the test set split
    acc = clf.score(test_data.X, test_data.y, lengths=test_data.lengths)

.. _definitions:

Definitions
^^^^^^^^^^^

.. autoclass:: sequentia.model_selection.KFold
   :members:
   :inherited-members:
   :exclude-members: get_metadata_routing, get_n_splits, split

.. autoclass:: sequentia.model_selection.StratifiedKFold
   :members:
   :inherited-members:
   :exclude-members: get_metadata_routing, get_n_splits, split

.. autoclass:: sequentia.model_selection.ShuffleSplit
   :members:
   :inherited-members:
   :exclude-members: get_metadata_routing, get_n_splits, split

.. autoclass:: sequentia.model_selection.StratifiedShuffleSplit
   :members:
   :inherited-members:
   :exclude-members: get_metadata_routing, get_n_splits, split

.. autoclass:: sequentia.model_selection.RepeatedKFold
   :members:
   :inherited-members:
   :exclude-members: get_metadata_routing, get_n_splits, split

.. autoclass:: sequentia.model_selection.RepeatedStratifiedKFold
   :members:
   :inherited-members:
   :exclude-members: get_metadata_routing, get_n_splits, split
