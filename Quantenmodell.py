import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.linalg import eigh
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize
import math

MARKET = "^GDAXI"
PERIOD = "5y"
INTERVAL = "1d"
K_STATES = 3
ROLL = 30

VALIDATION_WINDOW = 500       
LOOKBACK_WINDOW = 1000       

DATA_DIR = "HybridExperiment"
os.makedirs(DATA_DIR, exist_ok=True)

SEED = 42
np.random.seed(SEED)

# — tuned for realism —
LAMBDA_SCALE = 0.05
MU_CLAMP = 0.001

SIGMA_PRICE_BOUNDS = (0.005, 0.03)    # Tägliche Volatilitäten
SIGMA_RHO_BOUNDS   = (1e-4, 5e-3)     # quantum noise



def save_json(obj, path):
    def clean(x):
        if isinstance(x, complex):
            return float(x.real)
        if isinstance(x, np.ndarray):
            return clean(x.tolist())
        if isinstance(x, list):
            return [clean(v) for v in x]
        if isinstance(x, dict):
            return {k: clean(v) for k, v in x.items()}
        return x

    with open(path, "w") as f:
        json.dump(clean(obj), f, indent=2)


# Maktdaten laden

def download_market_data(symbol=MARKET, period=PERIOD, interval=INTERVAL):
    import yfinance as yf
    df = yf.download(symbol, period=period, interval=interval, progress=False)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join([str(c) for c in col if c]) for col in df.columns]

    colmap = {}
    for c in df.columns:
        l = c.lower()
        if "open" in l and "adj" not in l:
            colmap["Open"] = c
        if "high" in l:
            colmap["High"] = c
        if "low" in l:
            colmap["Low"] = c
        if "close" in l and "adj" not in l:
            colmap["Close"] = c
        if "adj close" in l or "adjclose" in l:
            colmap["Close"] = c
        if "volume" in l:
            colmap["Volume"] = c

    df2 = pd.DataFrame({
        "Date": df.index,
        "Open": df[colmap["Open"]],
        "High": df[colmap["High"]],
        "Low": df[colmap["Low"]],
        "Close": df[colmap["Close"]],
        "Volume": df[colmap["Volume"]]
    }).reset_index(drop=True)

    df2 = df2.sort_values("Date").reset_index(drop=True)

    return df2


# Eigenschaften der Zustände berechnen

def feature_engineering(df):
    df = df.copy()
    df["Return"] = np.log(df["Close"]).diff().fillna(0)

    high_low = df["High"] - df["Low"]
    high_pc  = (df["High"] - df["Close"].shift(1)).abs()
    low_pc   = (df["Low"]  - df["Close"].shift(1)).abs()
    TR = pd.concat([high_low, high_pc, low_pc], axis=1).max(axis=1)
    df["ATR_30"] = TR.rolling(ROLL).mean().fillna(method="bfill")
    df["ATR_rel"] = df["ATR_30"] / df["Close"]

    df["Vol_30"] = df["Return"].rolling(ROLL).std().fillna(0)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(method="bfill", inplace=True)
    df.fillna(method="ffill", inplace=True)

    return df[["Date", "Close", "Return", "ATR_rel", "Vol_30"]]


# 3. Zustände (k-Means)

def cluster_states(feat):
    X = feat[["Return", "ATR_rel", "Vol_30"]].values.astype(float)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    km = KMeans(n_clusters=K_STATES, random_state=SEED, n_init=20)
    labels = km.fit_predict(Xs)
    centers = scaler.inverse_transform(km.cluster_centers_)
    return labels, centers


# 4. Preisoperator

def compute_lambdas(feat, labels):
    df = feat.copy()
    df["State"] = labels
    lambdas = []
    for i in range(K_STATES):
        s = df.loc[df["State"] == i, "Return"]
        lambdas.append(s.mean() if len(s) else 0)

    lam = np.array(lambdas)
    lam_centered = lam - lam.mean()
    lam_scaled = lam_centered * LAMBDA_SCALE

    return lam, lam_centered, lam_scaled


# 5. Uhlenbeck & Ornstein Prozess - Parameter Schätzen 

def estimate_ou_params(feat, labels):
    df = feat.copy()
    df["State"] = labels
    out = {}
    for i in range(K_STATES):
        r = df.loc[df["State"] == i, "Return"].dropna().values
        if len(r) < 10:
            out[i] = [0.3, 0.0, 0.01]
            continue

        X = r[:-1]
        Y = r[1:]
        A = np.vstack([X, np.ones_like(X)]).T
        sol, _, _, _ = np.linalg.lstsq(A, Y, rcond=None)
        phi, c = sol
        a = max(1e-6, 1 - phi)
        mu = c / a
        mu = float(np.clip(mu, -MU_CLAMP, MU_CLAMP))
        sigma = np.std(Y - (phi*X + c))

        out[i] = [float(a), float(mu), float(sigma)]
    return out


# 6. Dichtematrix

def initial_rho(labels):
    counts = np.bincount(labels, minlength=K_STATES).astype(float)
    probs = counts / counts.sum()
    return np.diag(probs)


# 7. Dynamische Umsetzung (Lindbald Gleichung)

def comm(A, B):
    return A @ B - B @ A

def lindblad_diss(rho, Ls):
    D = np.zeros_like(rho, dtype=complex)
    for L in Ls:
        LdL = L.conj().T @ L
        D += L @ rho @ L.conj().T - 0.5*(LdL@rho + rho@LdL)
    return D

def enforce_density(rho):
    rho = 0.5*(rho + rho.conj().T)
    w,v = eigh(rho)
    w = np.clip(w, 0, None)
    rho = (v*w) @ v.conj().T
    rho /= np.trace(rho)
    return rho

def rk4_step(rho, dt, H, Ls):
    def f(r):
        return -1j*comm(H, r) + lindblad_diss(r, Ls)
    k1 = f(rho)
    k2 = f(rho + 0.5*dt*k1)
    k3 = f(rho + 0.5*dt*k2)
    k4 = f(rho + dt*k3)
    return rho + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)


# 8. Hybrid Simulation

def simulate(rho0, I_hat, P0, Hs, Ls,
             T=VALIDATION_WINDOW, dt=1.0,
             sigma_price=0.01, sigma_rho=1e-3,
             seed=SEED):

    rng = np.random.default_rng(seed)
    k = rho0.shape[0]

    rhos = np.zeros((T+1, k, k), dtype=complex)
    expectations = np.zeros(T+1)
    prices = np.zeros(T+1)

    rho = rho0.copy()
    prices[0] = P0
    expectations[0] = np.real(np.trace(rho @ I_hat))
    rhos[0] = rho

    for t in range(T):
        pops = np.real(np.diag(rho))
        H = sum(pops[i]*Hs[i] for i in range(k))

        rho_det = rk4_step(rho, dt, H, Ls)

        # stochastic quantum perturbation
        A = rng.normal(size=(k,k)) + 1j*rng.normal(size=(k,k))
        A = 0.5*(A + A.conj().T)
        rho_stoch = rho_det + sigma_rho*(A@rho_det - rho_det@A)

        rho = enforce_density(rho_stoch)
        rhos[t+1] = rho

        exp_val = np.real(np.trace(rho @ I_hat))
        expectations[t+1] = exp_val

        eps = rng.normal()
        prices[t+1] = prices[t] * math.exp(exp_val + sigma_price*eps)

    return rhos, expectations, prices


# 9. Kalibrierung

def loss_rmse(real, sim):
    L = min(len(real), len(sim))
    return float(np.sqrt(np.mean((np.log(real[:L]) - np.log(sim[:L]))**2)))

def calibrate_sigma(rho0, I_hat, P0, Hs, Ls, real_prices):

    def loss(x):
        sigma_p, sigma_r = x
        _,_,sim_prices = simulate(rho0, I_hat, P0, Hs, Ls,
                                  T=len(real_prices),
                                  sigma_price=sigma_p,
                                  sigma_rho=sigma_r,
                                  seed=SEED+5)
        return loss_rmse(real_prices, sim_prices)

    x0 = np.array([0.01, 1e-3])
    bounds = (SIGMA_PRICE_BOUNDS, SIGMA_RHO_BOUNDS)

    res = minimize(loss, x0, bounds=bounds, method="L-BFGS-B", options={"maxiter":20})
    print("Kalibrierung:", res)

    sigma_p = float(res.x[0])
    sigma_r = float(res.x[1])

    # HYBRID correction: inject realized daily volatility
    realized_vol = np.std(np.log(real_prices[1:] / real_prices[:-1]))
    sigma_p = 0.6*sigma_p + 0.4*realized_vol

    return sigma_p, sigma_r


# 10. Geasmmtausführung

def main():

    # Daten Laden
    df_raw = download_market_data()
    feat = feature_engineering(df_raw)

    # Trainings und Validierungsfenster aus Daten extrahieren
    feat = feat.iloc[-LOOKBACK_WINDOW:].reset_index(drop=True)
    df_raw = df_raw.iloc[-LOOKBACK_WINDOW:].reset_index(drop=True)


    # Einteilung in Zustände(Clustering)
    labels, centers = cluster_states(feat)

    # log-Return berchenen und Preisoperator konstruieren
    lam_raw, lam_centered, lam_scaled = compute_lambdas(feat, labels)
    I_hat = np.diag(lam_scaled)


    # Ornstein & Uhlenbeck Prozess schätzen
    ou_params = estimate_ou_params(feat, labels)

    # Dichtematrix zum Zeitpunkt t=0
    rho0 = initial_rho(labels)

    # Operatoren der Lindablad gleichung (H und L) 
    Hs = [0.01*(np.random.randn(K_STATES,K_STATES)) for _ in range(K_STATES)]
    Hs = [(H + H.T)/2 for H in Hs]
    Ls = []
    for i in range(K_STATES):
        P = np.zeros((K_STATES,K_STATES), dtype=complex)
        P[i,i] = 1
        Ls.append(np.sqrt(0.002)*P)

    # Validierungsfenster
    real_prices = df_raw["Close"].iloc[-VALIDATION_WINDOW:].values
    P0 = float(real_prices[0])   

    # Kalibrierung
    sigma_price, sigma_rho = calibrate_sigma(
        rho0, I_hat, P0, Hs, Ls, real_prices
    )


    # Finale Simulation
    rhos, expectations, prices = simulate(
        rho0, I_hat, P0, Hs, Ls,
        T=VALIDATION_WINDOW,
        sigma_price=sigma_price,
        sigma_rho=sigma_rho,
        seed=SEED+22
    )

    # Simulationsdaten speichern

    # Preis
    plt.figure(figsize=(10,4))
    plt.plot(real_prices, label="Real")
    plt.plot(prices, label="Simulation")
    plt.title("Preisvergleich (500 Tage)")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, "price_compare.png"))
    plt.close()

    # Erwartungswert
    plt.figure(figsize=(10,4))
    plt.plot(expectations)
    plt.title("Erwartungswert")
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, "expectation.png"))
    plt.close()

    # Kohärenz
    coh = np.array([np.sum(np.abs(rhos[i] - np.diag(np.diag(rhos[i]))))
                    for i in range(len(rhos))])
    plt.figure(figsize=(10,4))
    plt.plot(coh)
    plt.title("Kohärenz L1")
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, "coherence.png"))
    plt.close()

    # Zusamenfassung
    summary = {
        "lambda_raw": lam_raw.tolist(),
        "lambda_scaled": lam_scaled.tolist(),
        "ou_params": ou_params,
        "sigma_price": sigma_price,
        "sigma_rho": sigma_rho
    }
    save_json(summary, os.path.join(DATA_DIR, "summary.json"))


if __name__ == "__main__":
    main()
