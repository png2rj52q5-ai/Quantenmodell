import os
import json
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import pearsonr

# Import des bestehenden Quantenmodells
import Quantenmodell as QM


AUSGABE_ORDNER = "HybridExperiment"
os.makedirs(AUSGABE_ORDNER, exist_ok=True)

ANZAHL_SIMULATIONEN = 100
SEED_START = QM.SEED + 2000

SIMULATIONS_DAUER = QM.VALIDATION_WINDOW
LOOKBACK = QM.LOOKBACK_WINDOW
ANZAHL_ZUSTAENDE = QM.K_STATES

ALPHA_GRENZEN = (-1.0, 1.0)
SIGMA_PREIS_GRENZEN = QM.SIGMA_PRICE_BOUNDS
SIGMA_RHO_GRENZEN = QM.SIGMA_RHO_BOUNDS

# Hilfsfunktionen

def speichere_json(objekt, pfad):
    with open(pfad, "w", encoding="utf-8") as f:
        json.dump(objekt, f, indent=2, ensure_ascii=False)

def log_rmse(a, b):
    a, b = np.array(a), np.array(b)
    L = min(len(a), len(b))
    return float(np.sqrt(np.mean((np.log(a[:L] + 1e-12) - np.log(b[:L] + 1e-12))**2)))

def pearson_sicher(a, b):
    try:
        r, _ = pearsonr(a, b)
        return float(r)
    except Exception:
        return float("nan")

def abdeckung_real_in_band(real, simulationen, unteres=10, oberes=90):
    simulationen = np.array(simulationen)
    unter = np.percentile(simulationen, unteres, axis=0)
    ober = np.percentile(simulationen, oberes, axis=0)
    real = np.array(real)[:len(unter)]
    return float(np.mean((real >= unter) & (real <= ober)))

# Ornstein & Uhlenbeck Prozess simulieren

def ou_simuliere_renditen(T, a, mu, sigma, seed=None):
    rng = np.random.default_rng(seed)
    r = np.zeros(T)
    r[0] = mu
    for t in range(1, T):
        eps = rng.normal()
        r[t] = (1 - a) * r[t-1] + a * mu + sigma * eps
    return r

def renditen_zu_preisen(P0, renditen):
    logS = math.log(P0) + np.cumsum(renditen)
    preise = np.exp(logS)
    preise[0] = P0
    return preise

# Hybridmodell erster Durchlauf

def hybrid_einzellauf(P0, ou_parameter, quanten_erwartung, alpha, sigma_preis, seed=None):
    T = len(quanten_erwartung)
    a, mu, sigma = ou_parameter["a"], ou_parameter["mu"], ou_parameter["sigma"]

    r_ou = ou_simuliere_renditen(T, a, mu, sigma, seed)
    r_kombiniert = r_ou + alpha * np.array(quanten_erwartung)

    rng = np.random.default_rng(seed)
    r_stoch = r_kombiniert + sigma_preis * rng.normal(size=T)

    preise = renditen_zu_preisen(P0, r_stoch)
    return preise, r_stoch, r_ou

# Ducrhlauf von n Simulationen

def quantum_ensemble(rho0, I_hat, P0, Hs, Ls, n, T, sigma_rho):
    preise, erwartungen = [], []
    for i in range(n):
        rhos, exp, preis = QM.simulate(
            rho0, I_hat, P0, Hs, Ls,
            T=T, sigma_price=0.0, sigma_rho=sigma_rho,
            seed=SEED_START + i
        )
        preise.append(preis[:T])
        erwartungen.append(exp[:T])
    return np.array(preise), np.array(erwartungen)

def ou_ensemble(P0, T, a, mu, sigma, n):
    ensemble = []
    for i in range(n):
        r = ou_simuliere_renditen(T, a, mu, sigma, SEED_START + 10000 + i)
        ensemble.append(renditen_zu_preisen(P0, r))
    return np.array(ensemble)

# Kalibrierung

def kalibriere_hybrid(realpreise, P0, ou_parameter, quanten_erwartungen):
    template = np.median(quanten_erwartungen, axis=0)

    def verlust(x):
        alpha, sigma_preis = x
        preise, _, _ = hybrid_einzellauf(
            P0, ou_parameter, template, alpha, sigma_preis, SEED_START + 999
        )
        return log_rmse(realpreise, preise)

    res = minimize(
        verlust,
        x0=[0.05, 0.01],
        bounds=[ALPHA_GRENZEN, SIGMA_PREIS_GRENZEN],
        method="L-BFGS-B",
        options={"maxiter": 40}
    )
    return res

# Hauptprogramm

def main():
    print(">>> Starte Hybridexperiment")

    # Daten vorbereiten
    roh = QM.download_market_data()
    features = QM.feature_engineering(roh)
    features = features.iloc[-LOOKBACK:].reset_index(drop=True)
    roh = roh.iloc[-LOOKBACK:].reset_index(drop=True)

    labels, _ = QM.cluster_states(features)
    rho0 = QM.initial_rho(labels)
    _, _, lambdas = QM.compute_lambdas(features, labels)
    I_hat = np.diag(lambdas)

    ou_params = QM.estimate_ou_params(features, labels)
    a_med = float(np.median([ou_params[k][0] for k in ou_params]))
    mu_med = float(np.median([ou_params[k][1] for k in ou_params]))
    sigma_med = float(np.median([ou_params[k][2] for k in ou_params]))
    ou_basis = {"a": a_med, "mu": mu_med, "sigma": sigma_med}

    # Operatoren berechnen
    Hs = [(np.random.randn(ANZAHL_ZUSTAENDE, ANZAHL_ZUSTAENDE)) * 0.01 for _ in range(ANZAHL_ZUSTAENDE)]
    Hs = [(H + H.T)/2 for H in Hs]

    Ls = []
    for i in range(ANZAHL_ZUSTAENDE):
        P = np.zeros((ANZAHL_ZUSTAENDE, ANZAHL_ZUSTAENDE))
        P[i, i] = 1
        Ls.append(np.sqrt(0.002) * P)

    realpreise = roh["Close"].iloc[-SIMULATIONS_DAUER:].values
    P0 = float(realpreise[0])

    # Kalibrierung Quantenparameter
    sigma_preis, sigma_rho = QM.calibrate_sigma(rho0, I_hat, P0, Hs, Ls, realpreise)

    # n Durchläufe
    preise_q, erwartungen_q = quantum_ensemble(
        rho0, I_hat, P0, Hs, Ls,
        ANZAHL_SIMULATIONEN, SIMULATIONS_DAUER, sigma_rho
    )

    # Hybrid-Kalibrierung
    res = kalibriere_hybrid(realpreise, P0, ou_basis, erwartungen_q)
    alpha_opt, sigma_preis_opt = res.x

    # Ornstein & Uhlenbeck Prozess n Durchläufe
    sims_ou = ou_ensemble(P0, SIMULATIONS_DAUER, **ou_basis, n=ANZAHL_SIMULATIONEN)

    sims_hybrid = []
    for i in range(ANZAHL_SIMULATIONEN):
        preise, _, _ = hybrid_einzellauf(
            P0, ou_basis, erwartungen_q[i], alpha_opt, sigma_preis_opt, SEED_START + 20000 + i
        )
        sims_hybrid.append(preise)

    sims_hybrid = np.array(sims_hybrid)

    # Mediane
    med_ou = np.median(sims_ou, axis=0)
    med_q = np.median(preise_q, axis=0)
    med_h = np.median(sims_hybrid, axis=0)

    #Metriken: log-RMSE & Pearson-r

    # auf gleiche Länge bringen
    L = min(len(realpreise), len(med_h))
    real_cut = realpreise[:L]
    sim_cut  = med_h[:L]

    # log-RMSE
    log_rmse_wert = np.sqrt(
        np.mean(
        (np.log(real_cut + 1e-12) - np.log(sim_cut + 1e-12))**2
    )
    )

    # Pearson-r
    r_pearson = np.corrcoef(real_cut, sim_cut)[0, 1]

    
    metrik_text = (
        f"Pearson r = {r_pearson:.3f}\n"
        f"log-RMSE = {log_rmse_wert:.4f}"
    )   

    # In Plot einzeichnen
    plt.text(
        0.05, 0.95,
        metrik_text,
        transform=plt.gca().transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor="white",
            edgecolor="gray",
            alpha=0.85
        )
    )

    # Metriken
    metriken = {
        "OU_logRMSE": log_rmse(realpreise, med_ou),
        "Quantum_logRMSE": log_rmse(realpreise, med_q),
        "Hybrid_logRMSE": log_rmse(realpreise, med_h),
        "OU_Pearson": pearson_sicher(realpreise, med_ou),
        "Quantum_Pearson": pearson_sicher(realpreise, med_q),
        "Hybrid_Pearson": pearson_sicher(realpreise, med_h),
        "alpha": float(alpha_opt),
        "sigma_preis": float(sigma_preis_opt),
        "sigma_rho": float(sigma_rho)
    }

    speichere_json(metriken, os.path.join(AUSGABE_ORDNER, "metriken.json"))

    # Scatter-Plot

    # Längenanpassung
    L = min(len(realpreise), len(med_h))
    real_cut = realpreise[:L]
    sim_cut  = med_h[:L]

    # Metriken berechnen
    log_rmse_wert = np.sqrt(
        np.mean((np.log(real_cut + 1e-12) - np.log(sim_cut + 1e-12))**2)
    )
    pearson_wert = pearson_sicher(real_cut, sim_cut)

    # Plot
    plt.figure(figsize=(6,6))
    plt.scatter(real_cut, sim_cut, alpha=0.6, s=15)
    plt.plot(
        [real_cut.min(), real_cut.max()],
        [real_cut.min(), real_cut.max()],
        "r--",
        label="Ideal: y = x"
    )

    plt.xlabel("Reale Preise")
    plt.ylabel("Hybrid-Modell (Median)")
    plt.title("Scatter: Real vs. Hybrid")

    # Textbox
    plt.text(
        0.05, 0.95,
        f"Pearson r = {pearson_wert:.3f}\nlog-RMSE = {log_rmse_wert:.4f}",
        transform=plt.gca().transAxes,
        fontsize=11,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85)
    )

    plt.legend()
    plt.grid()
    plt.tight_layout()

    # Speichern
    plt.savefig(
        os.path.join(AUSGABE_ORDNER, "scatter_real_vs_hybrid.png"),
        dpi=150
    )
    plt.close()

    #Scatter Plot Ornstein & Uhlenbeck Prozess

    # Längenanpassung
    L = min(len(realpreise), len(med_ou))
    real_cut_ou = realpreise[:L]
    ou_cut      = med_ou[:L]

    # Metriken berechnen
    log_rmse_ou = np.sqrt(
        np.mean((np.log(real_cut_ou + 1e-12) - np.log(ou_cut + 1e-12))**2)
    )
    pearson_ou = pearson_sicher(real_cut_ou, ou_cut)

    # Plot
    plt.figure(figsize=(6,6))
    plt.scatter(real_cut_ou, ou_cut, alpha=0.6, s=15)
    plt.plot(
        [real_cut_ou.min(), real_cut_ou.max()],
        [real_cut_ou.min(), real_cut_ou.max()],
        "r--",
        label="Ideal: y = x"
    )

    plt.xlabel("Reale Preise")
    plt.ylabel("OU-Modell (Median)")
    plt.title("Scatter: Real vs. OU")

    # Textbox
    plt.text(
        0.05, 0.95,
        f"Pearson r = {pearson_ou:.3f}\nlog-RMSE = {log_rmse_ou:.4f}",
        transform=plt.gca().transAxes,
        fontsize=11,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85)
    )

    plt.legend()
    plt.grid()
    plt.tight_layout()

    # Speichern
    plt.savefig(
        os.path.join(AUSGABE_ORDNER, "scatter_real_vs_ou.png"),
        dpi=150
    )
    plt.close()

    # Plot
    t = np.arange(len(realpreise))
    plt.figure(figsize=(12,6))
    plt.plot(t, realpreise, label="Realer DAX", lw=2)
    plt.plot(t, med_ou, label="OU-Median")
    plt.plot(t, med_h, label="Hybrid-Median")
    plt.legend()
    plt.grid()
    plt.xlabel("Tage")
    plt.ylabel("Preis")
    plt.title("Hybridexperiment: OU + quanteninspirierte Schwankungen")
    plt.tight_layout()
    plt.savefig(os.path.join(AUSGABE_ORDNER, "preisvergleich.png"), dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
