"""week08 학습 실험에서 같이 쓰는 작은 함수들"""
import numpy as np
import lmi_common as C


def ref_(X):
    return X - X.mean(-1, keepdims=True)


def unit(X):
    return X / np.maximum(np.linalg.norm(X, axis=-1, keepdims=True), 1e-30)


def slor_kernel(Gn, lam2=1.0 / 9.0):
    """Gn (k, V) 한 방향 리드필드 → sLORETA 커널 (V, k), 평균 기준"""
    k = Gn.shape[0]
    R = np.eye(k) - 1.0 / k
    L = R @ Gn
    e, V = np.linalg.eigh(R @ R.T)
    keep = e > 1e-10 * e.max()
    W = (V[:, keep] / np.sqrt(e[keep])).T
    Lw = W @ L
    Lw /= np.sqrt(np.trace(Lw @ Lw.T) / keep.sum())
    A = np.linalg.solve(Lw @ Lw.T + lam2 * np.eye(keep.sum()), np.eye(keep.sum()))
    K = Lw.T @ A @ W
    return K / np.sqrt(np.einsum("ij,ji->i", Lw.T @ A, Lw))[:, None]


def dirs_rng(rng, n):
    return unit(rng.normal(size=(n, 3)))


def signal(Y, good):
    Yg = ref_(Y[good].T).T[:, C.tidx(*C.WIN)]
    u = Yg[:, np.argmax((Yg**2).sum(0))]
    return u / np.linalg.norm(u)
