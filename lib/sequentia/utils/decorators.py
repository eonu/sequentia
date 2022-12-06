import inspect, functools

from sequentia.utils.validation import _check_is_fitted

def _validate_params(using):
    def decorator(function):
        @functools.wraps(function)
        def wrapper(self=None, *args, **kwargs):
            spec = inspect.getfullargspec(function)
            if spec.varkw == 'kwargs' or len(spec.kwonlyargs) > 0:
                using.parse_obj(kwargs)
            if self is None:
                return function(*args, **kwargs)
            return function(self, *args, **kwargs)
        return wrapper
    return decorator

def _requires_fit(function):
    @functools.wraps(function)
    def wrapper(self, *args, **kwargs):
        _check_is_fitted(self)
        return function(self, *args, **kwargs)
    return wrapper

def _override_params(params, temporary=True):
    def decorator(function):
        @functools.wraps(function)
        def wrapper(self, *args, **kwargs):
            original_params = {}

            for param in params:
                if not hasattr(self, param):
                    raise AttributeError(f"'{type(self).__name__}' object has no attribute '{param}'")

            for param in params:
                if param in kwargs:
                    original_params[param] = getattr(self, param)
                    setattr(self, param, kwargs[param])

            try:
                return function(self, *args, **kwargs)
            finally:
                if temporary:
                    for param, value in original_params.items():
                        setattr(self, param, value)

        return wrapper
    return decorator

def _check_plotting_dependencies(function):
    @functools.wraps(function)
    def wrapper(self, *args, **kwargs):
        try:
            import matplotlib
        except ImportError as e:
            raise ImportError(f'The {function.__name__} function requires a working installation of matplotlib') from e
        return function(self, *args, **kwargs)
    return wrapper
