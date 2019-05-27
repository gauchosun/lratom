import numpy as np


def trim(ymat, xmat):
    y = ymat.copy()
    x = xmat.copy()
    assert len(y) == len(x)
    y = y.reshape(len(y), -1)
    x = x.reshape(len(x), -1)
    xyall = np.concatenate([y, x], axis=1)
    xyall = xyall[np.all(np.isfinite(xyall), axis=1).flatten()]
    assert np.isfinite(xyall).all()
    y_trimed = xyall[:, :y.shape[1]].copy()
    x_trimed = xyall[:, y.shape[1]:].copy()
    return y_trimed, x_trimed


def normer(array):
    mu = np.nanmean(array)
    sigma = np.nanstd(array)
    return mu, sigma, (array - mu) / sigma


def normalize(ndarray):
    xmat = ndarray.reshape(len(ndarray), -1).copy()
    x_normed = np.empty(xmat.shape)
    for i in range(xmat.shape[1]):
        array = xmat[:, i]
        *_, normed_array = normer(array)
        x_normed[:, i] = normed_array
    return x_normed.reshape(ndarray.shape)


def fisher(target, factor):
    assert len(target) == len(factor)
    y, x = trim(target, factor)
    n = len(y)
    if n < 3:
        return 0, 0, 0
    r = np.corrcoef(y.flatten(), x.flatten())[0, 1]
    return r, n, np.log((1+r)/(1-r)) * np.sqrt(n-3)


def regress(y, xmat):
    y_reg, x_reg = trim(y, xmat)
    beta = np.empty((x_reg.shape[1], y_reg.shape[1]))
    if y_reg.shape[0] < 3 or x_reg.shape[1] == 0:
        # beta = np.zeros((x_reg.shape[1], 1))
        beta[:] = np.nan
        return beta
    beta = np.linalg.pinv(x_reg.T @ x_reg) @ x_reg.T @ y_reg
    return beta


def iregress(y, xmat):
    y_reg, x_reg = trim(y, xmat)
    beta = np.empty((x_reg.shape[1]+1, y_reg.shape[1]))
    if y_reg.shape[0] < 3 or x_reg.shape[1] == 0:
        # beta = np.zeros((x_reg.shape[1]+1, 1))
        beta[:] = np.nan
        return beta
    x_reg = np.concatenate([np.ones((x_reg.shape[0], 1)), x_reg], axis=1)
    beta = np.linalg.pinv(x_reg.T @ x_reg) @ x_reg.T @ y_reg
    return beta


def pca(xmat):
    assert np.isfinite(xmat).all()
    m, n = xmat.shape
    sigma = (1/m) * xmat.T @ xmat
    vals, vecs = np.linalg.eig(sigma)
    idx = vals.argsort()[::-1]
    eigsys = tuple((vals[i], vecs[:, i]) for i in idx)
    return eigsys


def pcreg(y, xmat, n_components=3):
    y_reg, x_reg = trim(y, xmat)
    beta = np.empty((x_reg.shape[1], y_reg.shape[1]))
    if y_reg.shape[0] < 3 or x_reg.shape[1] == 0:
        beta[:] = np.nan
        return beta
    eigsys = pca(x_reg)[:n_components]
    proj_space = np.array([vec for _, vec in eigsys]).T
    x_proj = x_reg @ proj_space
    beta = regress(y_reg, x_proj)
    beta = proj_space @ beta
    return beta


def ipcreg(y, xmat, n_components=3):
    y_reg, x_reg = trim(y, xmat)
    beta = np.empty((x_reg.shape[1]+1, y_reg.shape[1]))
    if y_reg.shape[0] < 3:
        beta[:] = np.nan
        return beta
    eigsys = pca(x_reg)[:n_components]
    proj_space = np.array([vec for _, vec in eigsys]).T
    x_proj = x_reg @ proj_space
    beta = iregress(y_reg, x_proj)
    beta0 = np.array(beta[0]).reshape(-1, 1)
    beta1 = (proj_space @ beta[1:]).reshape(-1, 1)
    beta = np.concatenate([beta0, beta1], axis=0).reshape(-1, 1)
    return beta
