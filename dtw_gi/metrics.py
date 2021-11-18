import numpy as np
import torch
import torch.optim as optim
import geoopt
import scipy
from tslearn.metrics import dtw_path
from tslearn.utils import to_time_series, ts_size

from dtw_gi.stiefel_utils import StiefelLinear, StiefelLinearPerGroup
from dtw_gi.softdtw_metrics import SoftDTWWithMap


def path2mat(path):
    max0, max1 = path[-1]
    w_pi = np.zeros((max0 + 1, max1 + 1))
    for i, j in path:
        w_pi[i, j] = 1.
    return w_pi


def dtw_gi(ts0, ts1, init_p=None, max_iter=20, return_matrix=False,
           verbose=False, use_bias=False):
    r"""Compute Dynamic Time Warping with Global Invariance (DTW-GI) similarity
    measure between (possibly multidimensional) time series and return it.
    DTW-GI is computed as the Euclidean distance between aligned+rotated time
    series, i.e.:

    .. math::
        DTW-GI(X, Y) = \min_{P \in V_{d_0, d_1}} \min_{\Pi}
                        \sqrt{\sum_{(i, j) \in \Pi} \|X_{i} - Y_{j} P^T \|^2}
        
    It is not required that both time series share the same size, nor the same 
    dimension. DTW was originally presented in [1]_.
    
    Parameters
    ----------
    ts0: array of shape (sz0, d0)
        A time series.
        
    ts1: array of shape (sz1, d1)
        A time series.
        
    init_p : array of shape (d0, d1) (default: None)
        Initial p matrix for the Stiefel linear map. If None, identity matrix
        is used.
        
    max_iter : int (default: 20)
        Number of iterations for the iterative optimization algorithm.
    
    return_matrix : boolean (default: False)
        Whether the warping matrix should be returned in place of the path.
    
    verbose: boolean (default: True)
        Whether information should be printed during optimization

    use_bias: boolean (default: False)
        If True, the feature space map is affine, otherwise it is linear.
        
    Returns
    -------
    w_pi or path
        Warping matrix (binary matrix of shape (sz0, sz1) or path (list of
        index pairs)
        
    p
        Stiefel matrix
        
    cost
        Similarity score
    
    References
    ----------
    .. [1] H. Sakoe, S. Chiba, "Dynamic programming algorithm optimization for
           spoken word recognition," IEEE Transactions on Acoustics, Speech and
           Signal Processing, vol. 26(1), pp. 43--49, 1978.
    """
    ts0_ = to_time_series(ts0, remove_nans=True)
    ts1_ = to_time_series(ts1, remove_nans=True)

    sz0, d0 = ts0_.shape
    sz1, d1 = ts1_.shape

    ts0_m = ts0_.mean(axis=0).reshape((-1, 1))
    ts1_m = ts1_.mean(axis=0).reshape((-1, 1))

    w_pi = np.zeros((sz0, sz1))
    if init_p is None:
        p = np.eye(d0, d1)
    else:
        p = init_p
    bias = np.zeros((d0, 1))

    # BCD loop
    for iter in range(1, max_iter + 1):
        w_pi_old = w_pi
        # Temporal alignment
        path, cost = dtw_path(ts0_, ts1_.dot(p.T) + bias.T)
        if verbose:
            print("Iteration {}: DTW cost: {:.3f}".format(iter, cost))
        w_pi = path2mat(path)
        if np.allclose(w_pi, w_pi_old):
            break
        # Feature space registration
        if use_bias:
            m = (ts0_.T - ts0_m).dot(w_pi).dot(ts1_ - ts1_m.T)
        else:
            m = (ts0_.T).dot(w_pi).dot(ts1_)
        u, sigma, vt = scipy.linalg.svd(m, full_matrices=False)
        p = u.dot(vt)
        if use_bias:
            bias = ts0_m - ts1_m.T.dot(p.T).T
    path, cost = dtw_path(ts0_, ts1_.dot(p.T) + bias.T)
    if verbose:
        print("After optimization: DTW cost: {:.3f}".format(cost))
    if use_bias:
        if return_matrix:
            return w_pi, p, bias, cost
        else:
            return path, p, bias, cost
    else:
        if return_matrix:
            return w_pi, p, cost
        else:
            return path, p, cost


def f_shift(time_series, circular_shift):
    """
    >>> X = np.arange(9).reshape((3, 3))
    >>> f_shift(X, 0)
    array([[0, 1, 2],
           [3, 4, 5],
           [6, 7, 8]])
    >>> f_shift(X, -1)
    array([[1, 2, 0],
           [4, 5, 3],
           [7, 8, 6]])
    >>> f_shift(X, 1)
    array([[2, 0, 1],
           [5, 3, 4],
           [8, 6, 7]])
    >>> f_shift(X, -2)
    array([[2, 0, 1],
           [5, 3, 4],
           [8, 6, 7]])
    >>> f_shift(X, 2)
    array([[1, 2, 0],
           [4, 5, 3],
           [7, 8, 6]])
    """
    d = time_series.shape[1]
    if circular_shift == 0:
        return time_series
    elif circular_shift > 0:
        return np.hstack((time_series[:, -circular_shift:],
                          time_series[:, :d-circular_shift]))
    else:
        return np.hstack((time_series[:, -(d+circular_shift):],
                          time_series[:, :-circular_shift]))


def best_shift(ts0, ts1):
    d = ts0.shape[1]
    argmin_dist = None
    min_dist = np.inf
    for shift in range(d):
        dist = scipy.linalg.norm(ts0 - f_shift(ts1, shift))
        if dist < min_dist:
            min_dist = dist
            argmin_dist = shift
    return argmin_dist


def dtw_gi_circular_feature_map(ts0, ts1, max_iter=20,
                                return_matrix=False, verbose=False):
    r"""Compute f-invariant Dynamic Time Warping (f-DTW) similarity measure
    between (possibly multidimensional) time series and return it.
    f-DTW is computed as the Euclidean distance between aligned+rotated time
    series, i.e.:

    .. math::
        DTW-GI(X, Y) = \min_{f} \min_{\Pi}
                        \sqrt{\sum_{(i, j) \in \Pi} \|X_{i} - f(Y_{j}) \|^2}

    It is not required that both time series share the same size.
    DTW was originally presented in [1]_.

    The feature map considered here is a circular permutation of the feature
    space dimensions (corresponds to Optimal Transposition Index for cover
    song identification).

    Parameters
    ----------
    ts0: array of shape (sz0, d)
        A time series.

    ts1: array of shape (sz1, d)
        A time series.

    max_iter : int (default: 20)
        Number of iterations for the iterative optimization algorithm.

    return_matrix : boolean (default: False)
        Whether the warping matrix should be returned in place of the path.

    verbose: boolean (default: True)
        Whether information should be printed during optimization

    Returns
    -------
    w_pi or path
        Warping matrix (binary matrix of shape (sz0, sz1) or path (list of
        index pairs)

    d_shift
        Index of dimension circular shift

    cost
        Similarity score

    References
    ----------
    .. [1] H. Sakoe, S. Chiba, "Dynamic programming algorithm optimization for
           spoken word recognition," IEEE Transactions on Acoustics, Speech and
           Signal Processing, vol. 26(1), pp. 43--49, 1978.
    """
    ts0_ = to_time_series(ts0, remove_nans=True)
    ts1_ = to_time_series(ts1, remove_nans=True)

    sz0, d0 = ts0_.shape
    sz1, d1 = ts1_.shape

    assert d0 == d1

    w_pi = np.zeros((sz0, sz1))
    d_shift = 0

    # BCD loop
    for iter in range(1, max_iter + 1):
        w_pi_old = w_pi
        # Temporal alignment
        path, cost = dtw_path(ts0_, f_shift(ts1_, d_shift))
        if verbose:
            print("Iteration {}: DTW cost: {:.3f}".format(iter, cost))
        w_pi = path2mat(path)
        if np.allclose(w_pi, w_pi_old):
            break
        # Feature space registration
        indices0 = [t0 for (t0, t1) in path]
        indices1 = [t1 for (t0, t1) in path]
        d_shift = best_shift(ts0_[indices0], ts1_[indices1])

    path, cost = dtw_path(ts0_, f_shift(ts1_, d_shift))
    if verbose:
        print("After optimization: DTW cost: {:.3f}".format(cost))
    if return_matrix:
        return w_pi, d_shift, cost
    else:
        return path, d_shift, cost


def softdtw_gi(ts0, ts1, gamma=1.0, normalize=False,
              lr=0.001, max_iter=10, stiefel_opt=False, init_eye=True,
              init_zero=True,
              verbose=False, step_verbose=10,
              early_stopping_patience=None):
    r"""Compute soft-DTW with Global Invariances (softDTW-GI) similarity
    measure between (possibly multidimensional) time series and return it.
    softDTW-GI is computed as the Euclidean distance between aligned+rotated
    time series,
    i.e.:
    .. math::
        softDTW-GI(X, Y) = \min_{f} softDTW(X, f(Y))

    :math:`f` is assumed to be an affine map and, if stiefel_opt is True, its
    linear part lies on the Stiefel manifold

    It is not required that both time series share the same size, nor the same
    dimension. softDTW was originally presented in [1]_.

    Parameters
    ----------
    ts0: array of shape (sz0, d0)
        A time series.

    ts1: array of shape (sz1, d1)
        A time series.

    gamma : float (default: 1.0)
        gamma parameter for softDTW

    normalize : boolean (default: False)
        Whether normalized version of softDTW should be used

    lr : float (default: 0.001)
        Learning rate for the gradient descent optimization

    max_iter: int (default: 10)
        Number of epochs for the Gradient Descent aoptimization algorithm

    stiefel_opt: boolean (default: False)
        Whether linear part of the affine map should lie on the Stiefel
        manifold or not

    init_eye: boolean (default: True)
        Whether map should be initialized with identity matrix as its linear
        part.

    init_zero: boolean (default: True)
        Whether bias should be initialized as zeros vector

    verbose: boolean (default: False)
        Whether information should be printed during the optimization process

    step_verbose: int (default: 10)
        Number of epochs between two loss printouts (only used is verbose is
        True)

    early_stopping_patience: int or None (default: None)
        Number of gradient steps to wait before triggering early stopping.
        If None, early topping is not used and max_iter iterations are
        performed regardless of loss values.

    Returns
    -------
    cost
        Similarity score

    References
    ----------
    .. [1] M. Cuturi, M. Blondel "Soft-DTW: a Differentiable Loss Function for
       Time-Series," ICML 2017.
    """
    if not isinstance(ts0, torch.Tensor):
        ts0 = torch.Tensor(ts0[:ts_size(ts0)])
    if not isinstance(ts1, torch.Tensor):
        ts1 = torch.Tensor(ts1[:ts_size(ts1)])

    if stiefel_opt:
        affine_map = StiefelLinear(in_features=ts1.size(1),
                                   out_features=ts0.size(1),
                                   init_eye=init_eye,
                                   init_zero=init_zero)
    else:
        affine_map = torch.nn.Linear(in_features=ts1.size(1),
                                     out_features=ts0.size(1))
    # Note that the criterion embeds the map
    criterion = SoftDTWWithMap(map=affine_map,
                               gamma=gamma,
                               normalize=normalize)
    if stiefel_opt:
        optimizer = geoopt.optim.RiemannianAdam(affine_map.parameters(), lr=lr)
    else:
        optimizer = optim.Adam(affine_map.parameters(), lr=lr)

    best_loss = np.inf
    iter_best_loss = -1
    running_loss = 0.0
    for i in range(max_iter):
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        loss = criterion(ts0, ts1)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss = loss.item()
        if verbose and (i + 1) % step_verbose == 0:
            print('Iteration {}: soft-dtw loss: {:.3f}'.format(i + 1,
                                                               running_loss))
        if running_loss < best_loss:
            best_loss = running_loss
            iter_best_loss = i
        elif early_stopping_patience is not None and \
                        (i - iter_best_loss) > early_stopping_patience:
            if verbose:
                print("Early stopping at iteration {}".format(i + 1))
            break
    return running_loss

def softdtw_gi_with_map(ts0, ts1, gamma=1.0, normalize=False,
                        lr=0.001, max_iter=10, stiefel_opt=False,
                        init_eye=True, init_zero=True, bias=True,
                        verbose=False, step_verbose=10,
                        early_stopping_patience=None, group_size=None):
    r"""Compute soft-DTW with Global Invariances(softDTW-GI) similarity measure
    between (possibly multidimensional) time series and return it, together
    with its associated map.
    softDTW-GI is computed as the Euclidean distance between aligned+rotated
    time series, i.e.:
    .. math::
        softDTW-GI(X, Y) = \min_{f} softDTW(X, f(Y))

    :math:`f` is assumed to be an affine map and, if stiefel_opt is True, its
    linear part lies on the Stiefel manifold

    It is not required that both time series share the same size, nor the same
    dimension. softDTW was originally presented in [1]_.

    Parameters
    ----------
    ts0: array of shape (sz0, d0)
        A time series.

    ts1: array of shape (sz1, d1)
        A time series.

    gamma : float (default: 1.0)
        gamma parameter for softDTW

    normalize : boolean (default: False)
        Whether normalized version of softDTW should be used

    lr : float (default: 0.001)
        Learning rate for the gradient descent optimization

    max_iter: int (default: 10)
        Number of epochs for the Gradient Descent aoptimization algorithm

    stiefel_opt: boolean (default: False)
        Whether linear part of the affine map should lie on the Stiefel
        manifold or not

    init_eye: boolean (default: True)
        Whether map should be initialized with identity matrix as its linear
        part.

    init_zero: boolean (default: True)
        Whether bias should be initialized as zeros vector

    verbose: boolean (default: False)
        Whether information should be printed during the optimization process

    step_verbose: int (default: 10)
        Number of epochs between two loss printouts (only used is verbose is
        True)

    early_stopping_patience: int or None (default: None)
        Number of gradient steps to wait before triggering early stopping.
        If None, early topping is not used and max_iter iterations are
        performed regardless of loss values.

    Returns
    -------
    cost
        Similarity score

    map: torch.nn.Module
        Affine map

    References
    ----------
    .. [1] M. Cuturi, M. Blondel "Soft-DTW: a Differentiable Loss Function for
       Time-Series," ICML 2017.
    """
    if not isinstance(ts0, torch.Tensor):
        ts0 = torch.Tensor(ts0[:ts_size(ts0)])
    if not isinstance(ts1, torch.Tensor):
        ts1 = torch.Tensor(ts1[:ts_size(ts1)])

    if stiefel_opt:
        if group_size is None:
            affine_map = StiefelLinear(in_features=ts1.size(1),
                                       out_features=ts0.size(1),
                                       init_eye=init_eye,
                                       init_zero=init_zero,
                                       bias=bias)
        else:
            affine_map = StiefelLinearPerGroup(group_size=group_size,
                                               init_eye = init_eye,
                                               init_zero=init_zero,
                                               bias=bias)
    else:
        affine_map = torch.nn.Linear(in_features=ts1.size(1),
                                     out_features=ts0.size(1))
    criterion = SoftDTWWithMap(map=affine_map,
                               gamma=gamma,
                               normalize=normalize)
    if stiefel_opt:
        optimizer = geoopt.optim.RiemannianAdam(affine_map.parameters(), lr=lr)
    else:
        optimizer = optim.Adam(affine_map.parameters(), lr=lr)

    best_loss = np.inf
    iter_best_loss = -1
    running_loss = 0.0
    for i in range(max_iter):
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        loss = criterion(ts0, ts1)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss = loss.item()
        if verbose and (i + 1) % step_verbose == 0:
            print('Iteration {}: soft-dtw loss: {:.3f}'.format(i + 1,
                                                               running_loss))
        if running_loss < best_loss:
            best_loss = running_loss
            iter_best_loss = i
        elif early_stopping_patience is not None and \
            (i - iter_best_loss) > early_stopping_patience:
            if verbose:
                print("Early stopping at iteration {}".format(i + 1))
            break

    return running_loss, affine_map
