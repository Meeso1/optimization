### Optimization

This package contains `bayes_search()` function, which is a simple wrapper for scikit-optimize implementation of Bayesian optimization.

As scikit-optimize implementation is designed to optimize hyperparameters of scikit-learn models, it is impossible to directly use it to optimize any function, or even ML models from different frameworks. Also, it is not possible to easily opt out of the default cross-validation.

This package solves these problems by some weird hacks, exporting a simple interface that can be used to maximize any function that takes keyword arguments and returns a `float`.

### Example usage

```python
from optimization import bayes_search
from skopt.space import Real, Integer


def f(x: float, mode: int) -> float:
    # TODO: Continue
    pass

spaces = {
    "x": Real(1e-9, 1, prior="log-uniform"),
    "mode" Integer(1, 4)
}

best_params, max_value = bayes_search(
    f,
    spaces,
    iterations=20
)
```
