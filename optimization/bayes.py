from dataclasses import dataclass
from typing import Any, Callable
from sklearn.metrics import make_scorer
from skopt import BayesSearchCV
from skopt.space import Dimension


class SearchWrapper:
    """
    Wrapper for a function that allows it to be used in a scikit-optimize implementation of Bayesian search.
    This class mimics scikit-learn model interface.
    """
    optimized_function: Callable[[dict[str, Any]], float] | None = None
    quiet: bool = False

    @classmethod
    def set_function(cls, function: Callable[[dict[str, Any]], float]) -> None:
        cls.optimized_function = function

    @classmethod
    def clear_function(cls) -> None:
        cls.optimized_function = None

    @staticmethod
    def get_params_string(params: dict[str, Any]) -> str:
        def _format_value(value: Any) -> str:
            if isinstance(value, float):
                if value < 1e-3 or value > 1e3:
                    return f"{value:.4e}"
                return f"{value:.4f}"
            return str(value)

        return ", ".join([f"{key}={_format_value(value)}" for key, value in params.items()])

    def __init__(self, **kwargs):
        self.params = kwargs
        self.computed_value: float | None = None

    def fit(self, train_x, train_y):
        if self.optimized_function is None:
            raise ValueError("Search parameters were not set - call `set_function()` first")

        if not SearchWrapper.quiet:
            print(f"{{{SearchWrapper.get_params_string(self.params)}}}", end=" ")

        self.computed_value = self.optimized_function(self.params)

        if not SearchWrapper.quiet:
            print(f"-> {self.computed_value:.5f}")

        return self

    def predict(self, x):
        return self.computed_value

    def get_params(self, deep=False):
        return self.params

    def set_params(self, **params):
        return SearchWrapper(**params)


class NullFoldGenerator:
    """
    Fold generator for bayesian search that returns a single fold.
    Scikit-optimize does not allow to disable cross-validation, so instead we need to implement this.
    This way cross-validation can be done in other places and in different ways.
    """
    def __init__(self, n_splits=1):
        self.n_splits = n_splits

    def split(self, X, y, groups=None):
        return [([0], [0]) for _ in range(self.n_splits)]

    def get_n_splits(self, X, y, groups=None):
        return self.n_splits


def bayes_search(optimized_function: Callable[[dict[str, Any]], float],
                spaces_dict: dict[str, Dimension],
                iterations: int,
                *,
                quiet: bool = False) \
        -> tuple[dict[str, Any], float]:
    """
    Perform Bayesian search for function arguments.

    Args:
        optimized_function: A function to optimize.
        spaces_dict: A dictionary with arguments and their search spaces.
        iterations: Number of iterations of the search. Model will be fitted `iterations * repeats` times.
        quiet: Whether to suppress output.

    Returns:
        A tuple with best arguments and their corresponding value.
    """

    SearchWrapper.quiet = quiet
    SearchWrapper.set_function(optimized_function)

    bayes_search = BayesSearchCV(
        estimator=SearchWrapper(),
        search_spaces=spaces_dict,
        n_iter=iterations,
        cv=NullFoldGenerator(),
        scoring=make_scorer(lambda expected, predicted: predicted, greater_is_better=True),
        n_jobs=1,
        refit=False,
        return_train_score=False
    )

    bayes_search.fit([0]*5, [0]*5)

    SearchWrapper.clear_function()

    if not quiet:
        print(f"\nBest parameters: {SearchWrapper.get_params_string(bayes_search.best_params_)}")
        print(f"Best score: {bayes_search.best_score_:.5f}")

    return bayes_search.best_params_, bayes_search.best_score_
