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
    f = np.log((1+r)/(1-r)) * np.sqrt(n-3)
    return r, n, f


def pca(xmat):
    assert np.isfinite(xmat).all()
    m, n = xmat.shape
    sigma = (1/m) * xmat.T @ xmat
    vals, vecs = np.linalg.eig(sigma)
    idx = vals.argsort()[::-1]
    eigsys = tuple((vals[i], vecs[:, i]) for i in idx)
    return eigsys


def prepareg(y, xmat):
    y_reg, x_reg = trim(y, xmat)
    flag = 1
    if y_reg.shape[0] < 3 or x_reg.shape[1] == 0:
        flag = 0
    return flag, y_reg, x_reg


def calcbeta(y_reg, x_reg):
    beta = np.linalg.pinv(x_reg.T @ x_reg) @ x_reg.T @ y_reg
    return beta


def regress(y, xmat):
    flag, y_reg, x_reg = prepareg(y, xmat)
    if flag == 0:
        beta = np.array([[np.nan]] * x_reg.shape[1])
        return beta
    elif flag == 1:
        beta = calcbeta(y_reg, x_reg)
        return beta


def iregress(y, xmat):
    flag, y_reg, x_reg = prepareg(y, xmat)
    if flag == 0:
        beta = np.array([[np.nan]] * (x_reg.shape[1]+1))
        return beta
    elif flag == 1:
        x_reg = np.concatenate([np.ones((x_reg.shape[0], 1)), x_reg], axis=1)
        beta = calcbeta(y_reg, x_reg)
        return beta


def pcreg(y, xmat, n_components=3):
    flag, y_reg, x_reg = prepareg(y, xmat)
    if flag == 0:
        beta = np.array([[np.nan]] * x_reg.shape[1])
        return beta
    elif flag == 1:
        eigsys = pca(x_reg)[:n_components]
        proj_space = np.array([vec for _, vec in eigsys]).T
        x_proj = x_reg @ proj_space
        beta = calcbeta(y_reg, x_proj)
        beta = proj_space @ beta
        return beta


def ipcreg(y, xmat, n_components=3):
    flag, y_reg, x_reg = prepareg(y, xmat)
    if flag == 0:
        beta = np.array([[np.nan]] * (x_reg.shape[1]+1))
        return beta
    elif flag == 1:
        eigsys = pca(x_reg)[:n_components]
        proj_space = np.array([vec for _, vec in eigsys]).T
        x_proj = x_reg @ proj_space
        x_proj = np.concatenate([np.ones((x_proj.shape[0], 1)), x_proj], axis=1)
        beta = calcbeta(y_reg, x_proj)
        beta0 = np.array(beta[0]).reshape(-1, 1)
        beta1 = (proj_space @ beta[1:]).reshape(-1, 1)
        beta = np.concatenate([beta0, beta1], axis=0).reshape(-1, 1)
        return beta


def pred_reg(y, xmat):
    new = xmat[-1].copy()
    beta = regress(y, xmat[:-1]).flatten()
    pred = new @ beta
    return pred


def pred_ireg(y, xmat):
    new = xmat[-1].copy()
    beta = iregress(y, xmat[:-1]).flatten()
    pred = new @ beta[1:] + beta[0]
    return pred


def pred_pcreg(y, xmat, n_components=3):
    xmat = normalize(xmat)
    xmat = np.clip(xmat, -5, 5)
    new = xmat[-1].copy()
    beta = pcreg(y, xmat[:-1], n_components).flatten()
    pred = new @ beta
    return pred


def pred_ipcreg(y, xmat, n_components=3):
    xmat = normalize(xmat)
    xmat = np.clip(xmat, -5, 5)
    new = xmat[-1].copy()
    beta = ipcreg(y, xmat[:-1], n_components).flatten()
    pred = new @ beta[1:] + beta[0]
    return pred
