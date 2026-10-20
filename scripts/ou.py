"""News-driven OU: x_{t+Δt} = A^{Δt} x_t + ε_t,  y_t ≈ g0 + ⟨g, x_t⟩.

Default timing is next-day: x_t uses news through the previous close, so the
forecast of day-t returns is known after the close of t−1 (tradable at the next open).
A is contracting: scalar, diagonal (per-component decay), or exp(−K) with K ≽ 0.
Δt is calendar days between trading sessions (weekends / holidays).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LOGABS_FLOOR = 1e-4
COVID = (pd.Timestamp("2020-02-15"), pd.Timestamp("2020-06-30"))


# --------------------------------------------------------------------------- calendar / splits
def split_masks(dates, train_end: str, val_end: str) -> dict[str, np.ndarray]:
    idx = pd.DatetimeIndex(dates)
    train_end_ts, val_end_ts = pd.Timestamp(train_end), pd.Timestamp(val_end)
    if not train_end_ts < val_end_ts:
        raise ValueError("need TRAIN_END < VAL_END")
    train = np.asarray(idx <= train_end_ts, dtype=bool)
    val = np.asarray((idx > train_end_ts) & (idx <= val_end_ts), dtype=bool)
    test = np.asarray(idx > val_end_ts, dtype=bool)
    if not (train.any() and val.any() and test.any()):
        raise ValueError("empty train/val/test")
    return {"train": train, "val": val, "test": test}


def calendar_dt(dates) -> np.ndarray:
    """dt[t] = calendar days from t−1 to t (dt[0]=1)."""
    d = pd.DatetimeIndex(dates)
    dt = np.ones(len(d), dtype=np.float64)
    if len(d) > 1:
        dt[1:] = np.maximum((d[1:] - d[:-1]).days.astype(np.float64), 1.0)
    return dt


def covid_mask(dates) -> np.ndarray:
    d = pd.DatetimeIndex(dates)
    return np.asarray((d >= COVID[0]) & (d <= COVID[1]), dtype=bool)


# --------------------------------------------------------------------------- targets
def _raw_series(prices_wide: pd.DataFrame, ticker: str) -> np.ndarray:
    if ticker not in prices_wide.columns:
        raise KeyError(f"{ticker!r} not in prices_wide")
    return prices_wide[ticker].astype(float).to_numpy()


def target(prices_wide: pd.DataFrame, ticker: str, kind: str = "logabs",
           vol_window: int = 21, ma_window: int = 63) -> np.ndarray:
    """Next-day-ready series (still needs to be paired with x_t of the same index).

    kind:
      logret   Δ log p
      abs      |Δ log p|
      logabs   log(|Δ log p| + ε)          — default, dampens 2020 tails
      volnorm  |Δ log p| / σ_{t−1}(21d)
      diff     Δ p  (VIX, yields)
      logp_ma  log p − rolling mean(log p) — locally stationary stand-in for p_t
    """
    p = _raw_series(prices_wide, ticker)
    r = np.full_like(p, np.nan, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        r[1:] = np.log(p[1:]) - np.log(p[:-1])
    dlt = np.full_like(p, np.nan, dtype=np.float64)
    dlt[1:] = p[1:] - p[:-1]
    if kind == "logret":
        return r
    if kind == "abs":
        return np.abs(r)
    if kind == "logabs":
        return np.log(np.abs(r) + LOGABS_FLOOR)
    if kind == "diff":
        return dlt
    if kind == "volnorm":
        sig = pd.Series(r).shift(1).rolling(vol_window, min_periods=vol_window).std().to_numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.abs(r) / np.where(sig > 1e-12, sig, np.nan)
    if kind == "logp_ma":
        lp = np.log(np.clip(p, 1e-12, None))
        ma = pd.Series(lp).rolling(ma_window, min_periods=ma_window).mean().to_numpy()
        return lp - ma
    raise ValueError(f"unknown kind {kind!r}")


def target_panel(prices_wide: pd.DataFrame, specs: list[tuple[str, str]]) -> tuple[np.ndarray, list[str]]:
    cols, names = [], []
    for ticker, kind in specs:
        if ticker not in prices_wide.columns:
            continue
        cols.append(target(prices_wide, ticker, kind))
        names.append(f"{ticker} {kind}")
    if not cols:
        raise ValueError("empty panel")
    return np.column_stack(cols), names


# --------------------------------------------------------------------------- A^{Δt} x + ε
def _lam_from_theta(theta: np.ndarray) -> np.ndarray:
    return 0.01 + 0.98 * 1.0 / (1.0 + np.exp(-theta))


def _K_from_lvec(lvec: np.ndarray, d: int) -> np.ndarray:
    L = np.zeros((d, d), dtype=np.float64)
    L[np.tril_indices(d)] = lvec
    return L @ L.T + 1e-4 * np.eye(d)


def _A_full_from_vec(z: np.ndarray, d: int, rho_max: float = 0.99) -> np.ndarray:
    """A = ρ · M / ||M||_2  with ρ ∈ (0, ρ_max), so the spectral radius is < 1."""
    M = np.asarray(z[: d * d], dtype=np.float64).reshape(d, d)
    theta = float(z[d * d]) if len(z) > d * d else 0.0
    rho = rho_max / (1.0 + np.exp(-theta))
    s = np.linalg.svd(M, compute_uv=False)
    nrm = float(s[0]) if len(s) else 1.0
    if nrm < 1e-12:
        return np.zeros((d, d), dtype=np.float64)
    return (rho / nrm) * M


def states(eps: np.ndarray, A, dt: np.ndarray, kind: str = "diag",
           include_today: bool = False) -> np.ndarray:
    """x_0=0, x_{t+1}=A^{dt} x_t+ε_t. include_today: x_t = A^{dt} x_{t-1}+ε_t (overnight at the open)."""
    e = np.asarray(eps, dtype=np.float64)
    x = np.zeros_like(e)
    d = e.shape[1]
    if kind == "scalar":
        lam = float(np.asarray(A).reshape(-1)[0])

        def step(acc, eps_t, dti):
            return (lam ** dti) * acc + eps_t
    elif kind == "diag":
        lam = np.asarray(A, dtype=np.float64).reshape(d)

        def step(acc, eps_t, dti):
            return (lam ** dti) * acc + eps_t
    elif kind == "spd":
        K = np.asarray(A, dtype=np.float64)
        cache = {}

        def step(acc, eps_t, dti):
            key = float(dti)
            Adt = cache.get(key)
            if Adt is None:
                Adt = cache[key] = _expm_spd(-key * K)
            return Adt @ acc + eps_t
    elif kind == "full":
        Amat = np.asarray(A, dtype=np.float64)
        cache = {}

        def step(acc, eps_t, dti):
            n = int(max(round(float(dti)), 1))
            Adt = cache.get(n)
            if Adt is None:
                Adt = cache[n] = np.linalg.matrix_power(Amat, n)
            return Adt @ acc + eps_t
    else:
        raise ValueError(kind)

    acc = np.zeros(d)
    if include_today:
        for t in range(len(e)):
            acc = step(acc, e[t], dt[t] if t else 1.0)
            x[t] = acc
        return x
    for t in range(len(e) - 1):
        acc = step(acc, e[t], dt[t + 1])
        x[t + 1] = acc
    return x


def _expm_spd(M: np.ndarray) -> np.ndarray:
    """exp(M) for symmetric M via eigendecomposition."""
    w, v = np.linalg.eigh(0.5 * (M + M.T))
    return (v * np.exp(w)) @ v.T


def half_lives(lam: np.ndarray) -> np.ndarray:
    lam = np.asarray(lam, dtype=np.float64)
    out = np.full_like(lam, np.inf)
    ok = (lam > 0) & (lam < 1)
    out[ok] = np.log(2.0) / -np.log(lam[ok])
    out[lam <= 0] = 0.0
    return out


# --------------------------------------------------------------------------- observation
def fit_observation(x: np.ndarray, y: np.ndarray, ridge: float = 0.0):
    """Ridge y ~ g0 + x g. y may be (T,) or (T, m). Returns g0, G."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    one = y.ndim == 1
    if one:
        y = y[:, None]
    ok = np.isfinite(x).all(axis=1) & np.isfinite(y).all(axis=1)
    x, y = x[ok], y[ok]
    design = np.column_stack([np.ones(len(y)), x])
    xtx = design.T @ design
    if ridge:
        pen = ridge * np.eye(xtx.shape[0])
        pen[0, 0] = 0.0
        xtx = xtx + pen
    coef = np.linalg.solve(xtx, design.T @ y)
    g0, G = coef[0], coef[1:]
    if one:
        return float(g0[0]), G[:, 0]
    return g0, G


def predict(x: np.ndarray, g0, G) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    G = np.asarray(G, dtype=np.float64)
    g0 = np.asarray(g0, dtype=np.float64)
    return g0 + x @ G


# --------------------------------------------------------------------------- fit A + g
def residual(y: np.ndarray, control: np.ndarray, train: np.ndarray) -> np.ndarray:
    """y − â − b̂ control, coefficients from train (e.g. sector minus β·market)."""
    X = np.column_stack([np.ones(len(control)), np.asarray(control, float)])
    coef = _ols_coef(X, y, train)
    return np.asarray(y, float) - X @ coef


def fit_dynamics(eps, y, dt, train, val, *, kind="diag", ridge=1e-2, seed=0,
                 include_today: bool = False, maxiter: int = 40, rho_max: float = 0.99):
    """Jointly fit contracting A and (g0, g) on train; report val.

    kind: scalar | diag | spd | full
      full — unrestricted matrix with spectral radius < ρ_max via spectral-norm scaling.
    """
    from scipy.optimize import minimize

    e = np.asarray(eps, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if y.ndim > 1:
        y = y[:, 0]
    d = e.shape[1]
    n_l = d * (d + 1) // 2
    n_full = d * d + 1
    rng = np.random.default_rng(seed)

    def unpack(z):
        if kind == "scalar":
            A = _lam_from_theta(z[:1])[0]
            g0, g = float(z[1]), z[2:]
        elif kind == "diag":
            A = _lam_from_theta(z[:d])
            g0, g = float(z[d]), z[d + 1:]
        elif kind == "spd":
            A = _K_from_lvec(z[:n_l], d)
            g0, g = float(z[n_l]), z[n_l + 1:]
        elif kind == "full":
            A = _A_full_from_vec(z[:n_full], d, rho_max=rho_max)
            g0, g = float(z[n_full]), z[n_full + 1:]
        else:
            raise ValueError(kind)
        return A, g0, g

    def loss(z, mask):
        A, g0, g = unpack(z)
        x = states(e, A, dt, kind=kind, include_today=include_today)
        yhat = g0 + x @ g
        ok = mask & np.isfinite(y) & np.isfinite(yhat)
        resid = y[ok] - yhat[ok]
        return float(resid @ resid / max(ok.sum(), 1) + ridge * (g @ g))

    if kind == "scalar":
        z0 = np.concatenate([[0.0], [0.0], rng.normal(0, 0.01, d)])
    elif kind == "diag":
        z0 = np.concatenate([np.zeros(d), [0.0], rng.normal(0, 0.01, d)])
    elif kind == "spd":
        z0 = np.concatenate([rng.normal(0, 0.05, n_l), [0.0], rng.normal(0, 0.01, d)])
    else:
        z0 = np.concatenate([rng.normal(0, 0.05, d * d), [0.0], [0.0], rng.normal(0, 0.01, d)])

    res = minimize(lambda z: loss(z, train), z0, method="L-BFGS-B",
                   options=dict(maxiter=maxiter, ftol=1e-8))
    A, g0, g = unpack(res.x)
    x = states(e, A, dt, kind=kind, include_today=include_today)
    # ridge OLS refresh of g at the fitted A (more stable)
    g0, g = fit_observation(x[train], y[train], ridge=ridge)
    yhat = predict(x, g0, g)
    out = {"kind": kind, "A": A, "g0": g0, "g": g, "x": x, "yhat": yhat,
           "success": bool(res.success), "nfev": int(res.nfev)}
    if kind == "scalar":
        out["lam"] = np.array([A])
        out["half_life"] = half_lives(out["lam"])
    elif kind == "diag":
        out["lam"] = np.asarray(A)
        out["half_life"] = half_lives(out["lam"])
    elif kind == "spd":
        w = np.linalg.eigvalsh(A)
        out["lam"] = np.exp(-w)  # one-day eigenvalues of exp(-K)
        out["half_life"] = half_lives(out["lam"])
    else:
        ev = np.linalg.eigvals(A)
        out["lam"] = np.sort(np.abs(ev))[::-1]
        out["spectral_radius"] = float(np.max(np.abs(ev)))
        out["half_life"] = half_lives(np.clip(out["lam"].real, 0.0, 0.999999))
    for name, mask in (("train", train), ("val", val)):
        out[name] = metrics(y[mask], yhat[mask])
    return out


def select_ridge(eps, y, dt, train, val, *, kind="diag", ridges=(1e-3, 1e-2, 1e-1, 1.0, 10.0),
                 include_today: bool = False):
    best, best_r, best_s = None, None, -np.inf
    for r in ridges:
        fit = fit_dynamics(eps, y, dt, train, val, kind=kind, ridge=float(r),
                           include_today=include_today)
        score = fit["val"]["r2"]
        if np.isfinite(score) and score > best_s:
            best, best_r, best_s = fit, float(r), score
    best["ridge"] = best_r
    return best


# --------------------------------------------------------------------------- metrics / inference
def metrics(y, yhat) -> dict[str, float]:
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    ok = np.isfinite(y) & np.isfinite(yhat)
    y, yhat = y[ok], yhat[ok]
    n = int(len(y))
    if n < 3:
        return {"n": n, "mse": np.nan, "r2": np.nan, "r2_vs0": np.nan, "ic": np.nan}
    resid = y - yhat
    ss_res = float(resid @ resid)
    ss_mean = float(((y - y.mean()) ** 2).sum())
    ss_0 = float(y @ y)
    ic = float(np.corrcoef(y, yhat)[0, 1]) if yhat.std() > 0 else np.nan
    return {
        "n": n, "mse": ss_res / n,
        "r2": (1.0 - ss_res / ss_mean) if ss_mean > 0 else np.nan,
        "r2_vs0": (1.0 - ss_res / ss_0) if ss_0 > 0 else np.nan,
        "ic": ic,
    }


def newey_west_se(x: np.ndarray, lags: int | None = None) -> float:
    x = np.asarray(x, float)
    x = x - x.mean()
    n = len(x)
    if n < 4:
        return np.nan
    if lags is None:
        lags = int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0)))
    lags = max(lags, 1)
    gamma0 = float(x @ x) / n
    s = gamma0
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1)
        s += 2.0 * w * float(x[k:] @ x[:-k]) / n
    return float(np.sqrt(max(s, 0.0) / n))


def ic_hac(y, yhat) -> dict:
    from math import erfc, sqrt
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    ok = np.isfinite(y) & np.isfinite(yhat)
    y, yhat = y[ok], yhat[ok]
    yd, hd = y - y.mean(), yhat - yhat.mean()
    ic = float(np.corrcoef(y, yhat)[0, 1]) if hd.std() > 0 else np.nan
    s = yd * hd
    se = newey_west_se(s)
    t = float(s.mean() / se) if se and se > 0 else np.nan
    p = float(erfc(abs(t) / sqrt(2.0))) if np.isfinite(t) else np.nan  # two-sided
    return {"ic": ic, "t_hac": t, "p_hac": p, "n": int(len(y))}


def clark_west(y, yhat_restricted, yhat_unrestricted) -> dict:
    """One-sided CW test: H0 extra variables don't help OOS. Positive t → unrestricted better."""
    from math import erfc, sqrt
    y = np.asarray(y, float)
    e1 = y - np.asarray(yhat_restricted, float)
    e2 = y - np.asarray(yhat_unrestricted, float)
    adj = np.asarray(yhat_unrestricted, float) - np.asarray(yhat_restricted, float)
    f = e1 ** 2 - e2 ** 2 + adj ** 2
    ok = np.isfinite(f)
    f = f[ok]
    se = newey_west_se(f)
    t = float(f.mean() / se) if se and se > 0 else np.nan
    p = float(0.5 * erfc(t / sqrt(2.0))) if np.isfinite(t) else np.nan  # P(Z>t)
    return {"t": t, "p": p, "mean_f": float(f.mean()), "n": int(len(f))}


def block_bootstrap_ic(y, yhat, block: int = 10, n_boot: int = 400, seed: int = 0):
    rng = np.random.default_rng(seed)
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    ok = np.isfinite(y) & np.isfinite(yhat)
    y, yhat = y[ok], yhat[ok]
    n = len(y)
    if n < block * 2:
        return (np.nan, np.nan)
    n_blocks = int(np.ceil(n / block))
    ics = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n - block + 1, size=n_blocks)
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
        yh, yy = yhat[idx], y[idx]
        ics[b] = np.corrcoef(yy, yh)[0, 1] if yh.std() > 0 and yy.std() > 0 else np.nan
    return (float(np.nanquantile(ics, 0.025)), float(np.nanquantile(ics, 0.975)))


def calibration_deciles(y, yhat, n: int = 10) -> pd.DataFrame:
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    ok = np.isfinite(y) & np.isfinite(yhat)
    y, yhat = y[ok], yhat[ok]
    q = pd.qcut(yhat, n, labels=False, duplicates="drop")
    rows = []
    for i in sorted(set(q)):
        m = q == i
        rows.append({"decile": int(i + 1), "n": int(m.sum()),
                     "yhat": float(yhat[m].mean()), "y": float(y[m].mean())})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- baselines (all use info through t−1)
def _lag(a: np.ndarray, k: int = 1) -> np.ndarray:
    out = np.full_like(a, np.nan, dtype=np.float64)
    if k < len(a):
        out[k:] = a[:-k]
    return out


def _roll_mean_past(a: np.ndarray, window: int) -> np.ndarray:
    """mean of a_{t-window} … a_{t-1}."""
    s = pd.Series(a).shift(1).rolling(window, min_periods=window).mean()
    return s.to_numpy(dtype=np.float64)


def har_design(y: np.ndarray) -> np.ndarray:
    """Columns: 1, y_{t-1}, mean_{5}(past), mean_{22}(past)."""
    return np.column_stack([
        np.ones(len(y)),
        _lag(y, 1),
        _roll_mean_past(y, 5),
        _roll_mean_past(y, 22),
    ])


def garch11_sigma(r: np.ndarray, train: np.ndarray) -> np.ndarray:
    """σ_{t|t−1} from GARCH(1,1) fit on train returns, filtered on the full sample."""
    from scipy.optimize import minimize

    r = np.asarray(r, float)
    rt = r[train]
    rt = rt[np.isfinite(rt)]
    if len(rt) < 50:
        return np.full_like(r, np.nan)
    var0 = float(np.var(rt))

    def sig(z):
        omega = np.exp(z[0])
        alpha = 0.001 + 0.3 / (1 + np.exp(-z[1]))
        beta = 0.5 / (1 + np.exp(-z[2]))
        return omega, alpha, min(beta, 0.999 - alpha)

    def nll(z):
        omega, alpha, beta = sig(z)
        h = var0
        acc = 0.0
        for x in rt:
            acc += np.log(h) + x * x / h
            h = omega + alpha * x * x + beta * h
            h = max(h, 1e-12)
        return 0.5 * acc / len(rt)

    res = minimize(nll, np.array([np.log(var0 * 0.05 + 1e-12), -1.0, 2.0]),
                   method="L-BFGS-B", options=dict(maxiter=60))
    omega, alpha, beta = sig(res.x)
    h = np.full_like(r, np.nan)
    prev = var0
    for t in range(len(r)):
        h[t] = prev  # σ^2 of day t using r_{t-1}
        xt = 0.0 if not np.isfinite(r[t]) else r[t]
        prev = omega + alpha * xt * xt + beta * prev
        prev = max(prev, 1e-12)
    return np.sqrt(h)


def baseline_forecasts(y, r, vix, train, dt=None) -> dict[str, np.ndarray]:
    """Each series is a forecast of y_t using only t−1 information.

    Extra no-news baselines requested for the coursework:
      abs_lag — yesterday's |r|
      ewma    — same OU bucket as the news model, but the inflow is |r| itself (λ=0.94)
    """
    y = np.asarray(y, float)
    n = len(y)
    mu = float(np.nanmean(y[train]))
    yhat = {"mean": np.full(n, mu)}
    yhat["lag"] = _lag(y, 1)

    H = har_design(y)
    yhat["har"] = H @ _ols_coef(H, y, train)

    sig = garch11_sigma(r, train)
    yhat["garch"] = _affine(np.log(sig + LOGABS_FLOOR), y, train)

    vlag = _lag(np.asarray(vix, float), 1)
    daily_vol = (vlag / 100.0) / np.sqrt(252.0)
    yhat["vix"] = _affine(np.log(np.clip(daily_vol, 0, None) + LOGABS_FLOOR), y, train)
    yhat["lag"] = _affine(yhat["lag"], y, train)

    abs_r = np.abs(np.asarray(r, float))
    yhat["abs_lag"] = _affine(_lag(abs_r, 1), y, train)
    if dt is None:
        dt = np.ones(n, dtype=np.float64)
    x_ewma = states(np.nan_to_num(abs_r, nan=0.0)[:, None], 0.94, dt, kind="scalar")
    yhat["ewma"] = _affine(x_ewma[:, 0], y, train)
    return yhat


def _affine(pred, y, train):
    """Map a raw predictor to y via intercept + slope fit on train (fair MSE)."""
    X = np.column_stack([np.ones(len(pred)), np.asarray(pred, float)])
    return X @ _ols_coef(X, y, train)


def _ols_coef(X, y, mask, ridge=0.0):
    ok = mask & np.isfinite(y) & np.isfinite(X).all(1)
    Xk, yk = X[ok], y[ok]
    xtx = Xk.T @ Xk
    if ridge:
        xtx = xtx + ridge * np.eye(xtx.shape[0])
        xtx[0, 0] -= ridge
    return np.linalg.solve(xtx, Xk.T @ yk)


def nested_vol_vs_news(y, har_X, vix_lag, news_x, train, test, ridge=1e-2):
    controls = np.asarray(vix_lag, float)[:, None]
    return nested_news(y, news_x, train, test, controls=controls, ridge=ridge)


def nested_news(y, news_x, train, test, controls=None, ridge=1e-2):
    """Restricted: HAR(y) [+ controls]. Unrestricted: + news. Clark–West on test."""
    y = np.asarray(y, float)
    R = har_design(y)
    if controls is not None:
        C = np.asarray(controls, float)
        if C.ndim == 1:
            C = C[:, None]
        R = np.column_stack([R, C])
    U = np.column_stack([R, np.asarray(news_x, float)])

    def fit_pred(X):
        ok = train & np.isfinite(y) & np.isfinite(X).all(1)
        xtx = X[ok].T @ X[ok] + ridge * np.eye(X.shape[1])
        xtx[0, 0] -= ridge
        b = np.linalg.solve(xtx, X[ok].T @ y[ok])
        return X @ b

    y_r, y_u = fit_pred(R), fit_pred(U)
    return {
        "restricted": metrics(y[test], y_r[test]),
        "unrestricted": metrics(y[test], y_u[test]),
        "clark_west": clark_west(y[test], y_r[test], y_u[test]),
        "yhat_r": y_r, "yhat_u": y_u,
    }


def score_news(eps, y, dt, train, test, *, lam=0.0, include_today=False, ridge=1e-2, controls=None):
    """Fast: A=λI, OLS g. Returns news-only test metrics + nested Clark–West."""
    x = states(eps, lam, dt, kind="scalar", include_today=include_today)
    g0, g = fit_observation(x[train], y[train], ridge=ridge)
    yhat = predict(x, g0, g)
    news = metrics(y[test], yhat[test])
    nest = nested_news(y, x, train, test, controls=controls, ridge=ridge)
    return {"news": news, "nested": nest, "x": x, "yhat": yhat}


# --------------------------------------------------------------------------- walk-forward
def walk_forward(dates, eps, y, dt, *, kind="diag", ridge=1e-2, start_year=2013):
    """Expanding window, refit A,g each January, predict that year."""
    dates = pd.DatetimeIndex(dates)
    yhat = np.full(len(y), np.nan)
    years = []
    for year in range(start_year, int(dates[-1].year) + 1):
        train = np.asarray(dates < pd.Timestamp(f"{year}-01-01"), dtype=bool)
        test = np.asarray(dates.year == year, dtype=bool)
        if train.sum() < 400 or test.sum() < 20:
            continue
        fit = fit_dynamics(eps, y, dt, train, test, kind=kind, ridge=ridge)
        yhat[test] = fit["yhat"][test]
        years.append({"year": year, **{f"test_{k}": v for k, v in metrics(y[test], yhat[test]).items()}})
    return yhat, pd.DataFrame(years)


# --------------------------------------------------------------------------- helpers used by the coursework notebook
DEFAULT_LAMS = np.round(np.linspace(0.0, 0.99, 34), 4)


def standardize_train(X, train) -> np.ndarray:
    """Column-wise z-score using train moments only."""
    X = np.asarray(X, dtype=np.float64)
    one = X.ndim == 1
    if one:
        X = X[:, None]
    mu = np.nanmean(X[train], axis=0)
    sd = np.nanstd(X[train], axis=0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    Z = (X - mu) / sd
    return Z[:, 0] if one else Z


def cosine(a, b) -> float:
    a = np.asarray(a, float).ravel()
    b = np.asarray(b, float).ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return np.nan
    return float(a @ b / (na * nb))


def eval_row(name, y, yhat, train, val, test, extra=None) -> dict:
    row = {"model": name}
    for split, mask in (("train", train), ("val", val), ("test", test)):
        m = metrics(y[mask], yhat[mask])
        row[f"{split}_r2"] = m["r2"]
        row[f"{split}_ic"] = m["ic"]
        row[f"{split}_mse"] = m["mse"]
    if extra:
        row.update(extra)
    return row


def weekend_dt(n: int) -> np.ndarray:
    """Mon–Fri calendar: four 1-day steps, then Friday→Monday = 3."""
    dt = np.ones(n, dtype=np.float64)
    for t in range(1, n):
        dt[t] = 3.0 if (t % 5 == 0) else 1.0
    return dt


def simulate(T, d, lam, g, *, r2=0.5, seed=0, dt=None, weekends=False,
             include_today: bool = False) -> dict:
    """Draw ε_t, run the OU, observe y = ⟨g, x⟩ + noise calibrated to a target R²."""
    rng = np.random.default_rng(seed)
    g = np.asarray(g, dtype=np.float64).reshape(-1)
    if g.shape[0] != d:
        raise ValueError(f"g has length {g.shape[0]}, expected d={d}")
    if dt is None:
        dt = weekend_dt(T) if weekends else np.ones(T, dtype=np.float64)
    else:
        dt = np.asarray(dt, dtype=np.float64)
    eps = rng.normal(size=(T, d))
    x = states(eps, lam, dt, kind="scalar", include_today=include_today)
    signal = x @ g
    vs = float(np.var(signal))
    if r2 >= 1.0 - 1e-15:
        sigma, noise = 0.0, np.zeros(T)
    else:
        r2 = min(max(float(r2), 1e-12), 1.0 - 1e-12)
        sigma = np.sqrt(vs * (1.0 - r2) / r2) if vs > 0 else 1.0
        noise = sigma * rng.normal(size=T)
    y = signal + noise
    vy = float(np.var(y))
    return {
        "eps": eps, "x": x, "y": y, "signal": signal, "dt": dt,
        "lam": float(lam), "g": g, "sigma": float(sigma),
        "r2": (vs / vy) if vy > 0 else 1.0,
    }


def fit_scalar_grid(eps, y, dt, train, val=None, test=None, *,
                    lams=None, ridge=0.0, include_today: bool = False) -> dict:
    """Transparent 1-d search over λ: for each λ, x_t is known and g is OLS."""
    lams = DEFAULT_LAMS if lams is None else np.asarray(lams, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if y.ndim > 1:
        y = y[:, 0]
    rows, best, best_s = [], None, -np.inf
    for lam in lams:
        x = states(eps, lam, dt, kind="scalar", include_today=include_today)
        g0, g = fit_observation(x[train], y[train], ridge=ridge)
        yhat = predict(x, g0, g)
        rec = {"lam": float(lam), "g0": g0, "g": g, "x": x, "yhat": yhat,
               "train": metrics(y[train], yhat[train])}
        score = rec["train"]["r2"]
        if val is not None:
            rec["val"] = metrics(y[val], yhat[val])
            score = rec["val"]["r2"]
        if test is not None:
            rec["test"] = metrics(y[test], yhat[test])
        row = {"lam": float(lam), "train_r2": rec["train"]["r2"], "train_mse": rec["train"]["mse"]}
        if val is not None:
            row["val_r2"], row["val_mse"] = rec["val"]["r2"], rec["val"]["mse"]
        if test is not None:
            row["test_r2"], row["test_mse"] = rec["test"]["r2"], rec["test"]["mse"]
        rows.append(row)
        if np.isfinite(score) and score > best_s:
            best, best_s = rec, score
    if best is None:
        raise RuntimeError("grid produced no finite scores")
    best["grid"] = pd.DataFrame(rows)
    return best


def recovery_by_noise(T=2000, d=4, lam=0.9, r2s=(1.0, 0.5, 0.1, 0.01, 0.005, 0.001),
                      seed=0, weekends=True) -> pd.DataFrame:
    """Recover λ and g at several signal-to-noise levels (known ε)."""
    rng = np.random.default_rng(seed)
    g = rng.normal(size=d)
    g = g / np.linalg.norm(g)
    rows = []
    for r2 in r2s:
        sim = simulate(int(T), d, lam, g, r2=float(r2), seed=seed + int(round(1000 * float(r2))),
                       weekends=weekends)
        train = np.ones(len(sim["y"]), dtype=bool)
        fit = fit_scalar_grid(sim["eps"], sim["y"], sim["dt"], train)
        rows.append({
            "r2_target": float(r2), "r2_realized": sim["r2"],
            "lam_true": lam, "lam_hat": fit["lam"],
            "abs_err_lam": abs(fit["lam"] - lam),
            "cos_g": cosine(g, fit["g"]),
            "fit_r2": fit["train"]["r2"],
        })
    return pd.DataFrame(rows)


def power_lambda(lam_true=0.9, lam_alt=0.5, r2=0.005,
                 Ts=(250, 500, 1000, 2000, 4000), n_rep=12, d=4, seed=0,
                 weekends=True) -> pd.DataFrame:
    """Share of replications whose grid-λ is closer to lam_true than to lam_alt."""
    rng = np.random.default_rng(seed)
    g = rng.normal(size=d)
    g = g / np.linalg.norm(g)
    lams = np.unique(np.concatenate([DEFAULT_LAMS, [lam_true, lam_alt]]))
    rows = []
    for T in Ts:
        hats = np.empty(n_rep)
        closer = 0
        for i in range(n_rep):
            sim = simulate(int(T), d, lam_true, g, r2=r2,
                           seed=int(rng.integers(1_000_000_000)), weekends=weekends)
            train = np.ones(len(sim["y"]), dtype=bool)
            fit = fit_scalar_grid(sim["eps"], sim["y"], sim["dt"], train, lams=lams)
            hats[i] = fit["lam"]
            if abs(fit["lam"] - lam_true) < abs(fit["lam"] - lam_alt):
                closer += 1
        rows.append({
            "T": int(T), "years": round(int(T) / 252.0, 1),
            "mean_lam": float(hats.mean()), "std_lam": float(hats.std(ddof=1) if n_rep > 1 else 0.0),
            "p_closer_to_true": closer / n_rep,
            "p_above_mid": float((hats > 0.5 * (lam_true + lam_alt)).mean()),
        })
    return pd.DataFrame(rows)


def fit_full_torch(eps, y, dt, train, val, *, ridge=1e-2, rho_max=0.99,
                   steps=200, lr=0.05, include_today: bool = False, seed=0):
    """Full A with spectral-norm scaling, trained by Adam. Falls back is not our job — import torch."""
    import torch

    torch.manual_seed(int(seed))
    device = torch.device("cpu")
    e = torch.as_tensor(np.asarray(eps, np.float64), dtype=torch.float64, device=device)
    yt = torch.as_tensor(np.asarray(y, np.float64).reshape(-1), dtype=torch.float64, device=device)
    dt_i = np.asarray(dt, dtype=np.float64)
    T, d = int(e.shape[0]), int(e.shape[1])
    M = torch.nn.Parameter(0.05 * torch.randn(d, d, dtype=torch.float64, device=device))
    theta = torch.nn.Parameter(torch.zeros((), dtype=torch.float64, device=device))
    g0_p = torch.nn.Parameter(torch.zeros((), dtype=torch.float64, device=device))
    g_p = torch.nn.Parameter(0.01 * torch.randn(d, dtype=torch.float64, device=device))
    opt = torch.optim.Adam([M, theta, g0_p, g_p], lr=lr)
    tr = torch.as_tensor(np.asarray(train, dtype=bool), device=device) & torch.isfinite(yt)
    unique_n = sorted({int(max(round(float(v)), 1)) for v in dt_i})

    def make_A():
        rho = rho_max * torch.sigmoid(theta)
        nrm = torch.linalg.matrix_norm(M, ord=2)
        return (rho / (nrm + 1e-8)) * M

    def forward():
        A = make_A()
        powers = {n: torch.linalg.matrix_power(A, n) for n in unique_n}
        acc = torch.zeros(d, dtype=torch.float64, device=device)
        xs = []
        if include_today:
            for t in range(T):
                n = 1 if t == 0 else int(max(round(float(dt_i[t])), 1))
                acc = powers[n] @ acc + e[t]
                xs.append(acc)
        else:
            xs.append(acc)
            for t in range(T - 1):
                n = int(max(round(float(dt_i[t + 1])), 1))
                acc = powers[n] @ acc + e[t]
                xs.append(acc)
        x = torch.stack(xs, 0)
        return x, g0_p + x @ g_p, A

    last = None
    for _ in range(int(steps)):
        opt.zero_grad()
        x, yhat, A = forward()
        resid = yt[tr] - yhat[tr]
        loss = resid.square().mean() + ridge * (g_p @ g_p)
        loss.backward()
        opt.step()
        last = float(loss.detach())

    with torch.no_grad():
        x, _, A = forward()
    A_np = np.asarray(A.detach().cpu().numpy(), dtype=np.float64)
    x_np = np.asarray(x.detach().cpu().numpy(), dtype=np.float64)
    y_np = np.asarray(yt.detach().cpu().numpy(), dtype=np.float64)
    g0, g = fit_observation(x_np[np.asarray(train)], y_np[np.asarray(train)], ridge=ridge)
    yhat = predict(x_np, g0, g)
    ev = np.linalg.eigvals(A_np)
    out = {"kind": "full", "A": A_np, "g0": g0, "g": g, "x": x_np, "yhat": yhat,
           "loss": last, "steps": int(steps),
           "lam": np.sort(np.abs(ev))[::-1],
           "spectral_radius": float(np.max(np.abs(ev)))}
    out["half_life"] = half_lives(np.clip(out["lam"].real, 0.0, 0.999999))
    for name, mask in (("train", train), ("val", val)):
        out[name] = metrics(y_np[mask], yhat[mask])
    return out
