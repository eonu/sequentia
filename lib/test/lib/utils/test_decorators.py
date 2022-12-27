import pytest
from typing import List
from dataclasses import dataclass
from operator import itemgetter

from pydantic import BaseModel, ValidationError
from sklearn.exceptions import NotFittedError

from sequentia.models import KNNClassifier
from sequentia.utils.decorators import _validate_params, _requires_fit, _override_params

def test_validate_params():
    class Validator(BaseModel):
        param: List[int]

    @_validate_params(using=Validator)
    def to_validate(*, param):
        pass

    with pytest.raises(ValidationError):
        to_validate(param=None)


def test_requires_fit():
    class Model:
        def fit(self):
            self.fitted_ = True

        @_requires_fit
        def predict(self):
            pass

    model = Model()

    with pytest.raises(NotFittedError):
        model.predict()

    model.fit()
    model.predict()


@pytest.mark.parametrize('temporary', [True, False])
@pytest.mark.parametrize('error', [True, False])
def test_override_params(temporary, error):
    @dataclass
    class Model:
        b: int = 1
        c: int = 2

        @_override_params(['b', 'c'], temporary=temporary)
        def evaluate(self, a, **kwargs):
            for param in ('b', 'c'):
                if param in kwargs:
                    assert getattr(self, param) == kwargs[param]
            if error:
                raise ValueError()

    model = Model()

    try:
        model.evaluate(a=0)
    except ValueError:
        pass

    assert model.b == 1 and model.c == 2

    if temporary:
        try:
            model.evaluate(a=0, b=2, c=1)
        except ValueError:
            pass

        assert model.b == 1 and model.c == 2
    else:
        try:
            model.evaluate(a=0, b=2, c=1)
        except ValueError:
            pass

        assert model.b == 2 and model.c == 1
