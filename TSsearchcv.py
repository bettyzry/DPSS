import warnings

try:
    from collections.abc import Sized
except ImportError:
    from collections import Sized

import numpy as np
from scipy.stats import rankdata

from sklearn.model_selection._search import BaseSearchCV
from sklearn.utils import check_random_state

from sklearn.utils.validation import check_is_fitted
try:
    from sklearn.metrics import check_scoring
except ImportError:
    from sklearn.metrics.scorer import check_scoring

from skopt import Optimizer
from skopt.utils import point_asdict, dimensions_aslist, eval_callbacks
from skopt.space import check_dimension
from skopt.callbacks import check_callback


class DPSS(BaseSearchCV):
    def __init__(self, estimator, search_spaces, optimizer_kwargs=None,
                 n_iter=50, scoring=None, fit_params=None, n_jobs=1,
                 n_points=1, iid='deprecated', refit=True, cv=None, verbose=0,
                 pre_dispatch='2*n_jobs', random_state=None,
                 error_score='raise', return_train_score=False):

        self.search_spaces = search_spaces
        self.n_iter = n_iter
        self.n_points = n_points
        self.random_state = random_state
        self.optimizer_kwargs = optimizer_kwargs
        self._check_search_space(self.search_spaces)
        # Temporary fix for compatibility with sklearn 0.20 and 0.21
        # See scikit-optimize#762
        # To be consistent with sklearn 0.21+, fit_params should be deprecated
        # in the constructor and be passed in ``fit``.
        self.fit_params = fit_params

        if iid != "deprecated":
            warnings.warn("The `iid` parameter has been deprecated "
                          "and will be ignored.")
        self.iid = iid  # For sklearn repr pprint

        super(DPSS, self).__init__(
             estimator=estimator, scoring=scoring,
             n_jobs=n_jobs, refit=refit, cv=cv, verbose=verbose,
             pre_dispatch=pre_dispatch, error_score=error_score,
             return_train_score=return_train_score)

    def _check_search_space(self, search_space):
        """Checks whether the search space argument is correct"""

        if len(search_space) == 0:
            raise ValueError(
                "The search_spaces parameter should contain at least one"
                "non-empty search space, got %s" % search_space
            )

        # check if space is a single dict, convert to list if so
        if isinstance(search_space, dict):
            search_space = [search_space]

        # check if the structure of the space is proper
        if isinstance(search_space, list):
            # convert to just a list of dicts
            dicts_only = []

            # 1. check the case when a tuple of space, n_iter is provided
            for elem in search_space:
                if isinstance(elem, tuple):
                    if len(elem) != 2:
                        raise ValueError(
                            "All tuples in list of search spaces should have"
                            "length 2, and contain (dict, int), got %s" % elem
                        )
                    subspace, n_iter = elem

                    if (not isinstance(n_iter, int)) or n_iter < 0:
                        raise ValueError(
                            "Number of iterations in search space should be"
                            "positive integer, got %s in tuple %s " %
                            (n_iter, elem)
                        )

                    # save subspaces here for further checking
                    dicts_only.append(subspace)
                elif isinstance(elem, dict):
                    dicts_only.append(elem)
                else:
                    raise TypeError(
                        "A search space should be provided as a dict or"
                        "tuple (dict, int), got %s" % elem)

            # 2. check all the dicts for correctness of contents
            for subspace in dicts_only:
                for k, v in subspace.items():
                    check_dimension(v)
        else:
            raise TypeError(
                "Search space should be provided as a dict or list of dict,"
                "got %s" % search_space)

    @property
    def optimizer_results_(self):
        check_is_fitted(self, '_optim_results')
        return self._optim_results

    def _make_optimizer(self, params_space):
        """Instantiate skopt Optimizer class.

        Parameters
        ----------
        params_space : dict
            Represents parameter search space. The keys are parameter
            names (strings) and values are skopt.space.Dimension instances,
            one of Real, Integer or Categorical.

        Returns
        -------
        optimizer: Instance of the `Optimizer` class used for for search
            in some parameter space.

        """

        kwargs = self.optimizer_kwargs_.copy()
        kwargs['dimensions'] = dimensions_aslist(params_space)
        optimizer = Optimizer(**kwargs)
        for i in range(len(optimizer.space.dimensions)):
            if optimizer.space.dimensions[i].name is not None:
                continue
            optimizer.space.dimensions[i].name = list(sorted(
                params_space.keys()))[i]

        return optimizer

    def _step(self, search_space, optimizer, evaluate_candidates, n_points=1):
        """Generate n_jobs parameters and evaluate them in parallel.
        """
        # get parameter values to evaluate
        params = optimizer.ask(n_points=n_points)

        # convert parameters to python native types
        params = [[np.array(v).item() for v in p] for p in params]

        # make lists into dictionaries
        params_dict = [point_asdict(search_space, p) for p in params]

        all_results = evaluate_candidates(params_dict)
        # Feed the point and objective value back into optimizer
        # Optimizer minimizes objective, hence provide negative score
        local_results = all_results["mean_test_score"][-len(params):]
        return optimizer.tell(params, [-score for score in local_results])

    @property
    def total_iterations(self):
        """
        Count total iterations that will be taken to explore
        all subspaces with `fit` method.

        Returns
        -------
        max_iter: int, total number of iterations to explore
        """
        total_iter = 0

        for elem in self.search_spaces:

            if isinstance(elem, tuple):
                space, n_iter = elem
            else:
                n_iter = self.n_iter

            total_iter += n_iter

        return total_iter

    # TODO: Accept callbacks via the constructor?
    def fit(self, X, y=None, *, groups=None, callback=None, **fit_params):
        """Run fit on the estimator with randomly drawn parameters.

        Parameters
        ----------
        X : array-like or sparse matrix, shape = [n_samples, n_features]
            The training input samples.

        y : array-like, shape = [n_samples] or [n_samples, n_output]
            Target relative to X for classification or regression (class
            labels should be integers or strings).

        groups : array-like, with shape (n_samples,), optional
            Group labels for the samples used while splitting the dataset into
            train/test set.

        callback: [callable, list of callables, optional]
            If callable then `callback(res)` is called after each parameter
            combination tested. If list of callables, then each callable in
            the list is called.
        """
        self._callbacks = check_callback(callback)

        if self.optimizer_kwargs is None:
            self.optimizer_kwargs_ = {}
        else:
            self.optimizer_kwargs_ = dict(self.optimizer_kwargs)

        super().fit(X=X, y=y, groups=groups, **fit_params)

        # BaseSearchCV never ranked train scores,
        # but apparently we used to ship this (back-compat)
        if self.return_train_score:
            self.cv_results_["rank_train_score"] = \
                rankdata(-np.array(self.cv_results_["mean_train_score"]),
                         method='min').astype(int)
        return self

    def _run_search(self, evaluate_candidates):
        # check if space is a single dict, convert to list if so
        search_spaces = self.search_spaces
        if isinstance(search_spaces, dict):
            search_spaces = [search_spaces]

        callbacks = self._callbacks

        random_state = check_random_state(self.random_state)
        self.optimizer_kwargs_['random_state'] = random_state

        # Instantiate optimizers for all the search spaces.
        optimizers = []
        for search_space in search_spaces:
            if isinstance(search_space, tuple):
                search_space = search_space[0]
            optimizers.append(self._make_optimizer(search_space))
        self.optimizers_ = optimizers  # will save the states of the optimizers

        self._optim_results = []

        n_points = self.n_points

        for search_space, optimizer in zip(search_spaces, optimizers):
            # if not provided with search subspace, n_iter is taken as
            # self.n_iter
            if isinstance(search_space, tuple):
                search_space, n_iter = search_space
            else:
                n_iter = self.n_iter

            # do the optimization for particular search space
            while n_iter > 0:
                # when n_iter < n_points points left for evaluation
                n_points_adjusted = min(n_iter, n_points)

                optim_result = self._step(
                    search_space, optimizer,
                    evaluate_candidates, n_points=n_points_adjusted
                )
                n_iter -= n_points

                if eval_callbacks(callbacks, optim_result):
                    break
            self._optim_results.append(optim_result)
