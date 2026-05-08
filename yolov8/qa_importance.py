"""
QA(Quantum Annealing) 기반 채널 Importance — Torch-Pruning 호환.

QUBO 정식화:
    H(x) = -α·Σ(I_i·x_i) + β·Σ(R_ij·x_i·x_j) + λ·(Σx_i - k)²

  I_i  : 채널 i 의 importance (weight L1 norm)
  R_ij : 채널 i,j 의 redundancy (weight cosine similarity, [0,1])
  k    : 살릴 채널 수
  α,β,λ: 가중치 (λ는 평균 |w|에 자동 스케일)

Solver:
    1순위: dwave-neal (C++ SA, 빠름)
    2순위: pure-numpy SA (neal 미설치 시 자동 fallback)
    → 알고리즘은 동일 (Metropolis Simulated Annealing). 속도만 차이.

설치 (선택, 빠른 SA 원하면):
    pip install dwave-neal dimod

사용:
    prune_and_finetune.py 의 IMPORTANCE_FN 자리에서:
        from qa_importance import QAImportance
        IMPORTANCE_FN = lambda: QAImportance(prune_ratio=ratio)
"""

import numpy as np
import torch
import torch.nn as nn

import torch_pruning as tp


# neal/dimod 가용 여부 (있으면 사용, 없으면 numpy fallback)
try:
    import dimod
    import neal
    _HAS_NEAL = True
except ImportError:
    _HAS_NEAL = False


# =========================================================
# QUBO 구성 (형 코드 그대로의 수식)
# =========================================================
def compute_importance_from_weights(weight: torch.Tensor) -> np.ndarray:
    """채널별 L1 norm. shape (out_channels,)."""
    with torch.no_grad():
        if weight.dim() == 4:
            imp = torch.sum(torch.abs(weight), dim=(1, 2, 3))
        else:
            imp = torch.norm(
                weight.reshape(weight.shape[0], -1), p=1, dim=1)
    return imp.detach().cpu().numpy()


def compute_redundancy_from_weights(weight: torch.Tensor) -> np.ndarray:
    """채널 간 cosine similarity. [-1,1] → [0,1] 정규화, 대각 0."""
    with torch.no_grad():
        N = weight.shape[0]
        flat = weight.reshape(N, -1).float()
        flat = flat / (flat.norm(dim=1, keepdim=True) + 1e-12)
        S = (flat @ flat.T).detach().cpu().numpy()
    S = (S + 1.0) / 2.0
    np.fill_diagonal(S, 0.0)
    return S


def build_qubo_matrix(w, r, k, lam, beta):
    """
    H(x) = Σ w_i x_i + λ(Σx_i - k)² + β Σ_{i<j} r_ij x_i x_j

    대각:   w_i + λ(1 - 2k)
    비대각: 2λ + β·r_ij  (i<j 만 채움 → upper-triangular)
    """
    n = len(w)
    Q = np.zeros((n, n), dtype=np.float64)
    np.fill_diagonal(Q, w + lam * (1 - 2 * k))
    iu = np.triu_indices(n, k=1)
    Q[iu] = 2 * lam + beta * r[iu]
    return Q


# =========================================================
# Simulated Annealing — neal (1순위) / numpy (fallback)
# =========================================================
def _solve_sa_neal(Q: np.ndarray, num_reads: int, seed: int) -> np.ndarray:
    """dwave-neal 의 SimulatedAnnealingSampler 로 풀기 (빠름)."""
    n = Q.shape[0]
    qubo_dict = {}
    iu = np.triu_indices(n)   # diag + upper
    for i, j in zip(*iu):
        v = float(Q[i, j])
        if v != 0.0:
            qubo_dict[(int(i), int(j))] = v
    bqm = dimod.BinaryQuadraticModel.from_qubo(qubo_dict)
    sampler = neal.SimulatedAnnealingSampler()
    response = sampler.sample(bqm, num_reads=num_reads, seed=seed)
    best = response.first.sample
    return np.array([best[i] for i in range(n)], dtype=np.int32)


def _solve_sa_numpy(Q: np.ndarray, num_reads: int, num_sweeps: int,
                     seed: int) -> np.ndarray:
    """Pure-numpy Metropolis SA (neal 동일 알고리즘, 느림)."""
    rng = np.random.default_rng(seed)
    n = Q.shape[0]

    a = np.diag(Q).copy().astype(np.float64)
    Q_off = Q + Q.T
    np.fill_diagonal(Q_off, 0.0)

    scale = float(np.abs(Q).max() + 1e-12)
    T_start, T_end = scale, scale * 1e-3

    best_x, best_E = None, np.inf
    for _ in range(num_reads):
        x = rng.integers(0, 2, n).astype(np.int8)
        h = a + Q_off @ x.astype(np.float64)

        for sweep in range(num_sweeps):
            T = T_start * (T_end / T_start) ** (sweep / max(1, num_sweeps - 1))
            order = rng.permutation(n)
            rand_u = rng.random(n)
            for idx, i in enumerate(order):
                dE = (1 - 2 * x[i]) * h[i]
                if dE < 0 or rand_u[idx] < np.exp(-dE / max(T, 1e-12)):
                    delta = 1 - 2 * x[i]
                    x[i] = 1 - x[i]
                    h += Q_off[:, i] * delta

        x_f = x.astype(np.float64)
        E = float(a @ x_f + 0.5 * x_f @ Q_off @ x_f)
        if E < best_E:
            best_E = E
            best_x = x.copy()

    return best_x.astype(np.int32)


def solve_qubo_sa(Q: np.ndarray,
                   num_reads: int = 300,
                   num_sweeps: int = 500,
                   seed: int = 42) -> np.ndarray:
    """neal 있으면 그걸로, 없으면 numpy SA. 결과 형식 동일."""
    if _HAS_NEAL:
        return _solve_sa_neal(Q, num_reads=num_reads, seed=seed)
    return _solve_sa_numpy(Q, num_reads=num_reads,
                            num_sweeps=num_sweeps, seed=seed)


# =========================================================
# Torch-Pruning Importance 클래스
# =========================================================
class QAImportance(tp.importance.Importance):
    """GroupNormPruner 가 매 group 마다 호출.

    return: torch.Tensor, shape=(n_channels,) — 값 클수록 keep.
    """

    def __init__(self, prune_ratio: float = 0.5,
                 lam_scale: float = 2.0,
                 beta_scale: float = 0.2,
                 num_reads: int = 300,
                 num_sweeps: int = 500,
                 seed: int = 42,
                 verbose: bool = False):
        self.prune_ratio = float(prune_ratio)
        self.lam_scale = float(lam_scale)
        self.beta_scale = float(beta_scale)
        self.num_reads = int(num_reads)
        self.num_sweeps = int(num_sweeps)
        self.seed = int(seed)
        self.verbose = verbose
        self._call_idx = 0

        if not _HAS_NEAL and verbose:
            print("[QA] dwave-neal 미설치 → numpy SA fallback 사용 "
                  "(설치하면 더 빠름: pip install dwave-neal dimod)")

    def __call__(self, group, ch_groups=1, **kwargs):
        # 1. group 내 representative Conv
        rep_weight = None
        for dep, _idxs in group:
            m = dep.target.module
            if isinstance(m, nn.Conv2d) and m.weight.dim() == 4:
                rep_weight = m.weight
                break
        if rep_weight is None:
            return self._fallback_l2(group)

        n = rep_weight.shape[0]
        if n < 4:
            return self._fallback_l2(group)

        # 2. importance: -L1
        I = compute_importance_from_weights(rep_weight)
        I_norm = I / (I.max() + 1e-12)
        w = -I_norm

        # 3. redundancy: weight cosine
        r = compute_redundancy_from_weights(rep_weight)

        # 4. hyperparams
        k = max(1, int(round(n * (1 - self.prune_ratio))))
        lam = float(np.mean(np.abs(w)) * self.lam_scale)
        beta = lam * self.beta_scale

        # 5. QUBO 행렬 + SA
        Q = build_qubo_matrix(w, r, k, lam, beta)
        x = solve_qubo_sa(Q,
                          num_reads=self.num_reads,
                          num_sweeps=self.num_sweeps,
                          seed=self.seed + self._call_idx)

        if self.verbose:
            backend = "neal" if _HAS_NEAL else "numpy"
            print(f"  [QA-{backend}] group #{self._call_idx}: "
                  f"n={n}, k_target={k}, kept={int(x.sum())}, "
                  f"lam={lam:.4f}, beta={beta:.4f}")
        self._call_idx += 1

        # 6. 점수 텐서 (값 클수록 살림)
        scores = torch.tensor(
            x.astype(np.float32) + 0.001 * I_norm.astype(np.float32))
        return scores

    def _fallback_l2(self, group):
        for dep, _idxs in group:
            m = dep.target.module
            if hasattr(m, 'weight') and m.weight.dim() >= 2:
                w = m.weight
                scores = torch.norm(
                    w.reshape(w.shape[0], -1), p=2, dim=1)
                return scores.detach().cpu()
        return None
