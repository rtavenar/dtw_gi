"""
Code provided in this file is widely inspired from tslearn's
implementation of the DBA algorithm.
"""

import numpy
import torch
import torch.optim as optim
import geoopt
import warnings
from scipy.interpolate import interp1d
from sklearn.exceptions import ConvergenceWarning
from tslearn.utils import to_time_series, ts_size

from dtw_gi.stiefel_utils import StiefelLinear
from dtw_gi.softdtw_metrics import SoftDTW
from dtw_gi.metrics import dtw_gi


EPSILON = 1e-6


def dtw_gi_barycenter_averaging(X, barycenter_size=None,
                                init_barycenter=None,
                                max_iter=30, tol=1e-5, weights=None,
                                metric_params=None,
                                keep_p_matrices=False,
                                verbose=False, n_init=3):
    """DTW-GI Barycenter Averaging (DBA) method estimated through
    Expectation-Maximization algorithm.

    DBA was originally presented in [1]_.
    This implementation is based on a idea from [2]_ (Majorize-Minimize Mean
    Algorithm).

    Parameters
    ----------
    X : array-like, shape=(n_ts, sz, d)
        Time series dataset.

    barycenter_size : int or None (default: None)
        Size of the barycenter to generate. If None, the size of the barycenter
        is that of the data provided at fit
        time or that of the initial barycenter if specified.

    init_barycenter : array or None (default: None)
        Initial barycenter to start from for the optimization process.

    max_iter : int (default: 30)
        Number of iterations of the Expectation-Maximization optimization
        procedure.

    tol : float (default: 1e-5)
        Tolerance to use for early stopping: if the decrease in cost is lower
        than this value, the
        Expectation-Maximization procedure stops.

    weights: None or array
        Weights of each X[i]. Must be the same size as len(X).
        If None, uniform weights are used.

    metric_params: dict or None (default: None)
        DTW constraint parameters to be used.
        See :ref:`tslearn.metrics.dtw_path <fun-tslearn.metrics.dtw_path>` for
        a list of accepted parameters
        If None, no constraint is used for DTW computations.

    keep_p_matrices: bool (default: False)
        Whether P matrices from previous iteration should be used to initialize
        P matrices for the current iteration

    verbose : boolean (default: False)
        Whether to print information about the cost at each iteration or not.

    n_init : int (default: 3)
        Number of different initializations to try

    Returns
    -------
    numpy.array of shape (barycenter_size, d) or (sz, d) if barycenter_size \
            is None
        DBA barycenter of the provided time series dataset.

    float
        Corresponding cost (weighted sum of alignement costs)

    References
    ----------
    .. [1] F. Petitjean, A. Ketterlin & P. Gancarski. A global averaging method
       for dynamic time warping, with applications to clustering. Pattern
       Recognition, Elsevier, 2011, Vol. 44, Num. 3, pp. 678-693

    .. [2] D. Schultz and B. Jain. Nonsmooth Analysis and Subgradient Methods
       for Averaging in Dynamic Time Warping Spaces.
       Pattern Recognition, 74, 340-358.
    """
    best_cost = numpy.inf
    best_barycenter = None
    for i in range(n_init):
        if verbose:
            print("Attempt {}".format(i + 1))
        bary, loss = dtw_gi_barycenter_averaging_one_init(
            X=X,
            barycenter_size=barycenter_size,
            init_barycenter=init_barycenter,
            max_iter=max_iter,
            tol=tol,
            weights=weights,
            metric_params=metric_params,
            keep_p_matrices=keep_p_matrices,
            verbose=verbose
        )
        if loss < best_cost:
            best_cost = loss
            best_barycenter = bary
    return best_barycenter, best_cost


def dtw_gi_barycenter_averaging_one_init(X, barycenter_size=None,
                                         init_barycenter=None,
                                         max_iter=30, tol=1e-5, weights=None,
                                         metric_params=None,
                                         keep_p_matrices=False,
                                         verbose=False):
    """DTW-GI Barycenter Averaging (DBA) method estimated through
    Expectation-Maximization algorithm.

    DBA was originally presented in [1]_.
    This implementation is based on a idea from [2]_ (Majorize-Minimize Mean
    Algorithm).

    Here, the transform is of the form x -> Ax + b where A lies on the Stiefel
    manifold.

    Parameters
    ----------
    X : array-like, shape=(n_ts, sz, d)
        Time series dataset.

    barycenter_size : int or None (default: None)
        Size of the barycenter to generate. If None, the size of the barycenter
        is that of the data provided at fit
        time or that of the initial barycenter if specified.

    init_barycenter : array or None (default: None)
        Initial barycenter to start from for the optimization process.

    max_iter : int (default: 30)
        Number of iterations of the Expectation-Maximization optimization
        procedure.

    tol : float (default: 1e-5)
        Tolerance to use for early stopping: if the decrease in cost is lower
        than this value, the
        Expectation-Maximization procedure stops.

    weights: None or array
        Weights of each X[i]. Must be the same size as len(X).
        If None, uniform weights are used.

    metric_params: dict or None (default: None)
        DTW constraint parameters to be used.
        See :ref:`tslearn.metrics.dtw_path <fun-tslearn.metrics.dtw_path>` for
        a list of accepted parameters
        If None, no constraint is used for DTW computations.

    keep_p_matrices: bool (default: False)
        Whether P matrices from previous iteration should be used to initialize
        P matrices for the current iteration

    verbose : boolean (default: False)
        Whether to print information about the cost at each iteration or not.

    Returns
    -------
    numpy.array of shape (barycenter_size, d) or (sz, d) if barycenter_size \
            is None
        DBA barycenter of the provided time series dataset.

    float
        Corresponding cost (weighted sum of alignement costs)

    References
    ----------
    .. [1] F. Petitjean, A. Ketterlin & P. Gancarski. A global averaging method
       for dynamic time warping, with applications to clustering. Pattern
       Recognition, Elsevier, 2011, Vol. 44, Num. 3, pp. 678-693

    .. [2] D. Schultz and B. Jain. Nonsmooth Analysis and Subgradient Methods
       for Averaging in Dynamic Time Warping Spaces.
       Pattern Recognition, 74, 340-358.
    """
    X_ = [to_time_series(Xi, remove_nans=True) for Xi in X]
    if barycenter_size is None:
        barycenter_size = ts_size(X_[0])
    weights = _set_weights(weights, len(X_))
    if init_barycenter is None:
        try:
            barycenter = _init_avg(X_, barycenter_size)
        except:
            barycenter_idx = numpy.random.choice(len(X_))
            barycenter = X_[barycenter_idx]
    else:
        barycenter = init_barycenter
    cost_prev, cost = numpy.inf, numpy.inf
    list_p = [numpy.eye(Xi.shape[1], barycenter.shape[1]).T for Xi in X_]
    for it in range(max_iter):
        if not keep_p_matrices:
            list_p = None
        list_w_pi, list_p, list_bias, cost = _mm_assignment(
            X_,
            barycenter,
            weights,
            list_init_p=list_p,
            metric_params=metric_params
        )
        diag_sum_v_k, list_w_k = _mm_valence_warping(list_w_pi, weights)
        if verbose:
            print("[DBA] epoch %d, cost: %.3f" % (it + 1, cost))
        rotated_x = [numpy.empty(Xi.shape) for Xi in X_]
        for i in range(len(X_)):
            rotated_x[i] = X_[i].dot(list_p[i].T) + list_bias[i].T
        barycenter = _mm_update_barycenter(rotated_x, diag_sum_v_k, list_w_k)
        if abs(cost_prev - cost) < tol:
            break
        elif cost_prev < cost:
            warnings.warn("DBA loss is increasing while it should not be. "
                          "Stopping optimization.", ConvergenceWarning)
            break
        else:
            cost_prev = cost
    return barycenter, cost


def _set_weights(w, n):
    """Return w if it is a valid weight vector of size n, and a vector of n 1s
    otherwise.
    """
    if w is None or len(w) != n:
        w = numpy.ones((n,))
    return w


def _init_avg(X, barycenter_size):
    if ts_size(X[0]) == barycenter_size:
        return numpy.nanmean(X, axis=0)
    else:
        X_avg = numpy.nanmean(X, axis=0)
        xnew = numpy.linspace(0, 1, barycenter_size)
        f = interp1d(numpy.linspace(0, 1, X_avg.shape[0]), X_avg,
                     kind="linear", axis=0)
        return f(xnew)


def _mm_assignment(X, barycenter, weights, list_init_p=None,
                   metric_params=None):
    """Computes item assignement based on DTW alignments and return cost as a
    bonus.

    Parameters
    ----------
    X : numpy.array of shape (n, sz, d)
        Time-series to be averaged

    barycenter : numpy.array of shape (barycenter_size, d)
        Barycenter as computed at the current step of the algorithm.

    weights: array
        Weights of each X[i]. Must be the same size as len(X).

    list_init_p: list of arrays of shape (d, d)
        Initial P matrices

    metric_params: dict or None (default: None)
        Key-value parameters for f_dtw

    Returns
    -------
    list_W_pi
        List of warping matrices

    list_p
        List of mapping matrices

    float
        Current alignment cost
    """
    if metric_params is None:
        metric_params = {}
    n = len(X)
    if list_init_p is None:
        list_init_p = [numpy.eye(Xi.shape[1], barycenter.shape[1]).T
                       for Xi in X]
    cost = 0.
    list_w_pi = []
    list_p = []
    list_bias = []
    for i in range(n):
        w_pi, p_i, bias_i, dist_i = dtw_gi(barycenter, X[i],
                                           init_p=list_init_p[i],
                                           return_matrix=True,
                                           use_bias=True,
                                           **metric_params)
        cost += dist_i ** 2 * weights[i]
        list_w_pi.append(w_pi)
        list_p.append(p_i)
        list_bias.append(bias_i)
    cost /= weights.sum()
    return list_w_pi, list_p, list_bias, cost


def _subgradient_valence_warping(list_w_pi, weights):
    """Compute Valence and Warping matrices from paths.

    Valence matrices are denoted :math:`V^{(k)}` and Warping matrices are
    :math:`W^{(k)}` in [1]_.

    This function returns a list of :math:`V^{(k)}` diagonals (as a vector)
    and a list of :math:`W^{(k)}` matrices.

    Parameters
    ----------
    list_w_pi : list of arrays of shape (sz_bar, sz_i)
        List of warping matrices

    weights: array
        Weights of each X[i]. Must be the same size as len(X).

    Returns
    -------
    list of numpy.array of shape (barycenter_size, )
        list of weighted :math:`V^{(k)}` diagonals (as a vector)

    list of numpy.array of shape (barycenter_size, sz_k)
        list of weighted :math:`W^{(k)}` matrices

    References
    ----------

    .. [1] D. Schultz and B. Jain. Nonsmooth Analysis and Subgradient Methods
       for Averaging in Dynamic Time Warping Spaces.
       Pattern Recognition, 74, 340-358.
    """
    list_v_k = []
    list_w_k = []
    for k, w_pi in enumerate(list_w_pi):
        list_w_k.append(w_pi * weights[k])
        list_v_k.append(w_pi.sum(axis=1) * weights[k])
    return list_v_k, list_w_k


def _mm_valence_warping(list_w_pi, weights):
    """Compute Valence and Warping matrices from paths.

    Valence matrices are denoted :math:`V^{(k)}` and Warping matrices are
    :math:`W^{(k)}` in [1]_.

    This function returns the sum of :math:`V^{(k)}` diagonals (as a vector)
    and a list of :math:`W^{(k)}` matrices.

    Parameters
    ----------
    list_w_pi : list of arrays of shape (sz_bar, sz_i)
        list of Warping matrices

    barycenter_size : int
        Size of the barycenter to generate.

    weights: array
        Weights of each X[i]. Must be the same size as len(X).

    Returns
    -------
    numpy.array of shape (barycenter_size, )
        sum of weighted :math:`V^{(k)}` diagonals (as a vector)

    list of numpy.array of shape (barycenter_size, sz_k)
        list of weighted :math:`W^{(k)}` matrices

    References
    ----------

    .. [1] D. Schultz and B. Jain. Nonsmooth Analysis and Subgradient Methods
       for Averaging in Dynamic Time Warping Spaces.
       Pattern Recognition, 74, 340-358.
    """
    list_v_k, list_w_k = _subgradient_valence_warping(
        list_w_pi=list_w_pi,
        weights=weights)
    diag_sum_v_k = numpy.zeros(list_v_k[0].shape)
    for v_k in list_v_k:
        diag_sum_v_k += v_k
    return diag_sum_v_k, list_w_k


def _mm_update_barycenter(X, diag_sum_v_k, list_w_k):
    """Update barycenters using the formula from Algorithm 2 in [1]_.

    Parameters
    ----------
    X : numpy.array of shape (n, sz, d)
        Time-series to be averaged

    diag_sum_v_k : numpy.array of shape (barycenter_size, )
        sum of weighted :math:`V^{(k)}` diagonals (as a vector)

    list_w_k : list of numpy.array of shape (barycenter_size, sz_k)
        list of weighted :math:`W^{(k)}` matrices

    Returns
    -------
    numpy.array of shape (barycenter_size, d)
        Updated barycenter

    References
    ----------

    .. [1] D. Schultz and B. Jain. Nonsmooth Analysis and Subgradient Methods
       for Averaging in Dynamic Time Warping Spaces.
       Pattern Recognition, 74, 340-358.
    """
    d = X[0].shape[-1]
    barycenter_size = diag_sum_v_k.shape[0]
    sum_w_x = numpy.zeros((barycenter_size, d))
    for k, (w_k, x_k) in enumerate(zip(list_w_k, X)):
        sum_w_x += w_k.dot(x_k[:ts_size(x_k)])
    barycenter = numpy.diag(1. / diag_sum_v_k).dot(sum_w_x)
    return barycenter


def normalize_tensor(w):
    return w / (torch.norm(w) + EPSILON)


def nanmean(losses):
    s = 0.
    n = 0
    for l in losses:
        if not torch.isnan(l):
            s += l
            n += 1
    return s / n


def softdtw_gi_barycenter(dataset, barycenter_size, barycenter_dim,
                         gamma=1.0, normalize_dist=False,
                         lr=0.001, max_iter=10,
                         normalize_map=False,
                         stiefel_opt=False,
                         verbose=False, step_verbose=10, n_init=3):
    r"""Compute soft-DTW with Global Invariances (softDTW-GI) barycenter for a
    dataset.

    :math:`f` is assumed to be an affine map and, if stiefel_opt is True, its
    linear part lies on the Stiefel manifold

    It is not required that all time series share the same size, nor the same
    dimension. softDTW was originally presented in [1]_.

    Parameters
    ----------
    dataset: array of shape (n, sz, d) or list of arrays of shape (sz_i, d_i)
        A dataset of time series.

    barycenter_size: int
        Size of the barycenter to be generated

    barycenter_dim: int
        Dimensionality of the barycenter to be generated

    gamma : float (default: 1.0)
        gamma parameter for softDTW

    normalize_dist : boolean (default: False)
        Whether normalized version of softDTW should be used

    lr : float (default: 0.001)
        Learning rate for the gradient descent optimization

    max_iter: int (default: 10)
        Number of epochs for the Gradient Descent aoptimization algorithm

    normalize_map: boolean (default: False)
        Whether barycenter should be normalized after each epoch
        (useful is stiefel_opt is False)

    stiefel_opt: boolean (default: False)
        Whether linear part of the affine map should lie on the Stiefel
        manifold or not

    verbose: boolean (default: False)
        Whether information should be printed during the optimization process

    step_verbose: int (default: 10)
        Number of epochs between two loss printouts (only used is verbose is
        True)

    n_init: int (default: 3)
        Number of different initializations to be tried (the one with lowest
        loss is kept in the end)

    Returns
    -------
    barycenter
        Obtained barycenter

    cost
        Inertia

    References
    ----------
    .. [1] M. Cuturi, M. Blondel "Soft-DTW: a Differentiable Loss Function for
       Time-Series," ICML 2017.
    """
    best_cost = numpy.inf
    best_barycenter = None
    for i in range(n_init):
        if verbose:
            print("Attempt {}".format(i + 1))
        bary, loss = softdtw_gi_barycenter_one_init(
            dataset=dataset,
            barycenter_size=barycenter_size,
            barycenter_dim=barycenter_dim,
            gamma=gamma,
            normalize_dist=normalize_dist,
            lr=lr, max_iter=max_iter,
            normalize_map=normalize_map,
            stiefel_opt=stiefel_opt,
            verbose=verbose,
            step_verbose=step_verbose
        )
        if loss < best_cost:
            best_cost = loss
            best_barycenter = bary
    return best_barycenter, best_cost


def softdtw_gi_barycenter_one_init(dataset, barycenter_size=None,
                                   barycenter_dim=None,
                                   init_barycenter=None,
                                   gamma=1.0, normalize_dist=False,
                                   lr=0.001, max_iter=10,
                                   normalize_map=False,
                                   stiefel_opt=False,
                                   verbose=False,
                                   step_verbose=10):
    """softDTW-GI barycenter computation.

    This implementation assumes that the dimension in which the barycenter lies
    is lower or equal to the smallest dimension of all time series in the
    dataset."""
    if init_barycenter is None:
        barycenter = torch.empty(barycenter_size, barycenter_dim,
                                 requires_grad=True)
        torch.nn.init.normal_(barycenter, mean=0.0, std=1.0)
    else:
        barycenter = torch.Tensor(init_barycenter)
        barycenter.requires_grad_(True)
        barycenter_dim = barycenter.size(-1)

    tensor_list_dataset = []
    affine_maps = []
    for time_series in dataset:
        tensor_list_dataset.append(
            torch.Tensor(time_series[:ts_size(time_series)])
        )
        if stiefel_opt:
            affine_maps.append(
                StiefelLinear(in_features=barycenter_dim,
                              out_features=tensor_list_dataset[-1].size(1))
            )
        else:
            affine_maps.append(
                torch.nn.Linear(in_features=barycenter_dim,
                                out_features=tensor_list_dataset[-1].size(1))
            )
    criterion = SoftDTW(gamma=gamma, normalize=normalize_dist)
    params = [barycenter]
    for f in affine_maps:
        params.extend(list(f.parameters()))
    if stiefel_opt:
        optimizer = geoopt.optim.RiemannianAdam(params, lr=lr)
    else:
        optimizer = optim.Adam(params, lr=lr)

    running_loss = 0.0
    for i in range(max_iter):
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        losses = [criterion(f(barycenter), x)
                  for f, x in zip(affine_maps, tensor_list_dataset)]
        # print(losses)
        loss = nanmean(losses)
        loss.backward()
        optimizer.step()

        if normalize_map:
            barycenter.data = normalize_tensor(barycenter).detach()

        # print statistics
        running_loss = loss.item()
        if verbose and (i + 1) % step_verbose == 0:
            print('Iteration {}: soft-dtw-barycenter loss: {:.5f}'.format(
                i + 1,
                running_loss
            )
            )
    return barycenter.detach().numpy(), running_loss
