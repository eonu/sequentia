.. _searching:

Hyper-parameter search methods
==============================

In order to optimize the hyper-parameters for a specific model,
hyper-parameter search methods are used (often in conjunction with
:ref:`cross-validation methods <splitting>`) to evaluate the performance of a model 
with different configurations and find the optimal settings.

:mod:`sklearn.model_selection` provides such hyper-parameter search methods,
but does not support sequence data. Sequentia provides modified
versions of these methods to support sequence data.

API reference
-------------

Classes/Methods
^^^^^^^^^^^^^^^

.. autosummary::

   ~sequentia.model_selection.param_grid
   ~sequentia.model_selection.GridSearchCV
   ~sequentia.model_selection.RandomizedSearchCV
   ~sequentia.model_selection.HalvingGridSearchCV
   ~sequentia.model_selection.HalvingRandomSearchCV

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

.. autofunction:: sequentia.model_selection.param_grid

.. autoclass:: sequentia.model_selection.GridSearchCV
   :members: __init__
   :exclude-members: __new__

.. autoclass:: sequentia.model_selection.RandomizedSearchCV
   :members: __init__
   :exclude-members: __new__

.. autoclass:: sequentia.model_selection.HalvingGridSearchCV
   :members: __init__
   :exclude-members: __new__

.. autoclass:: sequentia.model_selection.HalvingRandomSearchCV
   :members: __init__
   :exclude-members: __new__