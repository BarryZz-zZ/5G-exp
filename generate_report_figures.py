import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from scipy.signal import correlate, welch


def estimate_psd(x, fs, nperseg=1024, nfft=4096):
    f, p = welch(
        x,
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=nperseg // 2,
        nfft=nfft,
        return_onesided=False,
        scaling="density",
    )
    return np.fft.fftshift(f), 10 * np.log10(np.fft.fftshift(p) + 1e-12)


def detect_toa_pss(rx, refs, fs):
    peak_abs = []
    peak_lag = []
    peak_metric = []
    corr_all = []
    lags_all = []
    for k in range(refs.shape[1]):
        ref = refs[:, k]
        c = correlate(rx, np.conj(ref), mode="full")
        lags = np.arange(-(len(ref) - 1), len(rx))
        a = np.abs(c)
        idx = int(np.argmax(a))
        peak_abs.append(float(a[idx]))
        peak_lag.append(int(lags[idx]))
        peak_metric.append(float(a[idx] / (np.median(a) + 1e-12)))
        corr_all.append(c)
        lags_all.append(lags)

    best = int(np.argmax(peak_abs))
    toa_sample = int(np.clip(peak_lag[best] + 1, 1, len(rx)))
    return {
        "best_template_idx_1based": best + 1,
        "assumed_npss": best,
        "toa_lag": int(peak_lag[best]),
        "toa_sample": toa_sample,
        "toa_time_us": float(toa_sample / fs * 1e6),
        "peak_abs": peak_abs,
        "peak_metric": peak_metric,
        "peak_lag_all": peak_lag,
        "corr_all": corr_all,
        "lags_all": lags_all,
    }


def solve_large_cfo(rx_fo, refs, fs):
    cfg = {
        "coarse_min_hz": -500_000.0,
        "coarse_max_hz": 500_000.0,
        "coarse_step_hz": 2_000.0,
        "fine_span_hz": 4_000.0,
        "fine_step_hz": 100.0,
    }

    n_all = np.arange(len(rx_fo))
    best = {
        "score": -1e30,
        "template_idx_1based": 1,
        "cfo_hz": 0.0,
        "raw_peak_metric": 0.0,
        "comp_peak_metric": 0.0,
        "rough_lag": 0,
        "df_grid_hz": None,
        "metric_grid": None,
        "rx_comp": rx_fo,
    }

    for k in range(refs.shape[1]):
        ref = refs[:, k]
        c = correlate(rx_fo, np.conj(ref), mode="full")
        lags = np.arange(-(len(ref) - 1), len(rx_fo))
        a = np.abs(c)
        i = int(np.argmax(a))
        lag = int(lags[i])
        raw_metric = float(a[i] / (np.median(a) + 1e-12))

        st = int(np.clip(lag + 1, 1, len(rx_fo) - len(ref) + 1) - 1)
        seg = rx_fo[st : st + len(ref)]
        n = np.arange(len(ref))

        coarse = np.arange(
            cfg["coarse_min_hz"],
            cfg["coarse_max_hz"] + cfg["coarse_step_hz"],
            cfg["coarse_step_hz"],
        )
        metric_coarse = np.empty_like(coarse)
        for ii, df in enumerate(coarse):
            seg_comp = seg * np.exp(-1j * 2 * np.pi * df * n / fs)
            metric_coarse[ii] = np.abs(np.vdot(ref, seg_comp))
        df0 = coarse[int(np.argmax(metric_coarse))]

        fine = np.arange(
            df0 - cfg["fine_span_hz"],
            df0 + cfg["fine_span_hz"] + cfg["fine_step_hz"],
            cfg["fine_step_hz"],
        )
        metric_fine = np.empty_like(fine)
        for ii, df in enumerate(fine):
            seg_comp = seg * np.exp(-1j * 2 * np.pi * df * n / fs)
            metric_fine[ii] = np.abs(np.vdot(ref, seg_comp))

        df_est = float(fine[int(np.argmax(metric_fine))])
        rx_comp = rx_fo * np.exp(-1j * 2 * np.pi * df_est * n_all / fs)
        c_comp = correlate(rx_comp, np.conj(ref), mode="full")
        a_comp = np.abs(c_comp)
        comp_metric = float(np.max(a_comp) / (np.median(a_comp) + 1e-12))

        if comp_metric > best["score"]:
            best.update(
                {
                    "score": comp_metric,
                    "template_idx_1based": k + 1,
                    "cfo_hz": df_est,
                    "raw_peak_metric": raw_metric,
                    "comp_peak_metric": comp_metric,
                    "rough_lag": lag,
                    "df_grid_hz": fine,
                    "metric_grid": metric_fine,
                    "rx_comp": rx_comp,
                }
            )

    return best


def main():
    root = Path(__file__).resolve().parent
    out_dir = root / "report_figures"
    out_dir.mkdir(exist_ok=True)

    data = sio.loadmat(root / "data_5G.mat")
    fs = 7.68e6
    ts = 1 / fs

    rx = data["recv_signal"].ravel()
    rx_fo = data["recv_signal_freq_offset"].ravel()
    refs = data["ref_signal_grid"]

    res_normal = detect_toa_pss(rx, refs, fs)
    res_fo_raw = detect_toa_pss(rx_fo, refs, fs)
    cfo_best = solve_large_cfo(rx_fo, refs, fs)
    rx_fo_comp = cfo_best["rx_comp"]
    res_fo_comp = detect_toa_pss(rx_fo_comp, refs, fs)

    plt.style.use("seaborn-v0_8-whitegrid")

    # Figure 1: time-domain signals
    n_show = min(3000, len(rx))
    t_ms = np.arange(n_show) * ts * 1e3
    fig = plt.figure(figsize=(11, 6))
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(t_ms, np.real(rx[:n_show]), lw=1, label="Real")
    ax1.plot(t_ms, np.imag(rx[:n_show]), lw=1, label="Imag")
    ax1.set_title("recv_signal (first 3000 samples)")
    ax1.set_xlabel("Time (ms)")
    ax1.set_ylabel("Amplitude")
    ax1.legend()
    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(t_ms, np.real(rx_fo[:n_show]), lw=1, label="Real")
    ax2.plot(t_ms, np.imag(rx_fo[:n_show]), lw=1, label="Imag")
    ax2.set_title("recv_signal_freq_offset (first 3000 samples)")
    ax2.set_xlabel("Time (ms)")
    ax2.set_ylabel("Amplitude")
    ax2.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "fig1_time_domain.png", dpi=150)
    plt.close(fig)

    # Figure 2: PSD
    f1, p1 = estimate_psd(rx, fs)
    f2, p2 = estimate_psd(rx_fo, fs)
    fig = plt.figure(figsize=(11, 4.5))
    plt.plot(f1 / 1e6, p1, lw=1.1, label="recv_signal")
    plt.plot(f2 / 1e6, p2, lw=1.1, label="recv_signal_freq_offset")
    plt.title("PSD comparison")
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("PSD (dB)")
    plt.legend()
    plt.tight_layout()
    fig.savefig(out_dir / "fig2_psd.png", dpi=150)
    plt.close(fig)

    # Figure 3: correlation overlay on recv_signal
    fig = plt.figure(figsize=(11, 4.8))
    for k in range(3):
        c = np.abs(res_normal["corr_all"][k])
        l = res_normal["lags_all"][k]
        plt.plot(l, c / (np.max(c) + 1e-12), lw=1.0, label=f"ref col {k+1}")
    plt.title("Cross-correlation on recv_signal")
    plt.xlabel("Lag (samples)")
    plt.ylabel("Normalized |xcorr|")
    plt.legend()
    plt.tight_layout()
    fig.savefig(out_dir / "fig3_corr_recv_signal.png", dpi=150)
    plt.close(fig)

    # Figure 4: peak comparison on recv_signal
    fig = plt.figure(figsize=(8, 4.5))
    x = np.arange(1, 4)
    plt.bar(x - 0.15, res_normal["peak_abs"], width=0.3, label="Peak abs")
    plt.bar(x + 0.15, res_normal["peak_metric"], width=0.3, label="Peak metric")
    plt.xticks(x, ["Ref1", "Ref2", "Ref3"])
    plt.title("PSS decision features on recv_signal")
    plt.xlabel("Reference template")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    fig.savefig(out_dir / "fig4_pss_features_recv_signal.png", dpi=150)
    plt.close(fig)

    # Figure 5: raw correlation on freq-offset signal
    fig = plt.figure(figsize=(11, 4.8))
    for k in range(3):
        c = np.abs(res_fo_raw["corr_all"][k])
        l = res_fo_raw["lags_all"][k]
        plt.plot(l, c / (np.max(c) + 1e-12), lw=1.0, label=f"ref col {k+1}")
    plt.title("Cross-correlation on recv_signal_freq_offset (raw)")
    plt.xlabel("Lag (samples)")
    plt.ylabel("Normalized |xcorr|")
    plt.legend()
    plt.tight_layout()
    fig.savefig(out_dir / "fig5_corr_freq_offset_raw.png", dpi=150)
    plt.close(fig)

    # Figure 6: CFO search curve
    fig = plt.figure(figsize=(10, 4.5))
    plt.plot(cfo_best["df_grid_hz"] / 1e3, cfo_best["metric_grid"], lw=1.1)
    plt.title(
        f"CFO fine search metric (selected template col {cfo_best['template_idx_1based']})"
    )
    plt.xlabel("Candidate CFO (kHz)")
    plt.ylabel("Coherent metric")
    plt.tight_layout()
    fig.savefig(out_dir / "fig6_cfo_search.png", dpi=150)
    plt.close(fig)

    # Figure 7: best-template correlation before/after compensation
    k = cfo_best["template_idx_1based"] - 1
    ref_best = refs[:, k]
    c_raw = correlate(rx_fo, np.conj(ref_best), mode="full")
    l_raw = np.arange(-(len(ref_best) - 1), len(rx_fo))
    c_cmp = correlate(rx_fo_comp, np.conj(ref_best), mode="full")
    l_cmp = np.arange(-(len(ref_best) - 1), len(rx_fo_comp))
    fig = plt.figure(figsize=(11, 4.8))
    plt.plot(l_raw, np.abs(c_raw) / (np.max(np.abs(c_raw)) + 1e-12), lw=1.0, label="Before")
    plt.plot(l_cmp, np.abs(c_cmp) / (np.max(np.abs(c_cmp)) + 1e-12), lw=1.0, label="After")
    plt.title(f"Best-template correlation before/after compensation (col {k+1})")
    plt.xlabel("Lag (samples)")
    plt.ylabel("Normalized |xcorr|")
    plt.legend()
    plt.tight_layout()
    fig.savefig(out_dir / "fig7_corr_before_after_comp.png", dpi=150)
    plt.close(fig)

    # Figure 8: phase evolution in matched segment
    lag = cfo_best["rough_lag"]
    st = int(np.clip(lag + 1, 1, len(rx_fo) - len(ref_best) + 1) - 1)
    seg_raw = rx_fo[st : st + len(ref_best)]
    seg_cmp = rx_fo_comp[st : st + len(ref_best)]
    phi_raw = np.unwrap(np.angle(seg_raw * np.conj(ref_best)))
    phi_cmp = np.unwrap(np.angle(seg_cmp * np.conj(ref_best)))
    fig = plt.figure(figsize=(10, 4.5))
    plt.plot(phi_raw, lw=1.0, label="Before")
    plt.plot(phi_cmp, lw=1.0, label="After")
    plt.title("Phase drift before/after compensation")
    plt.xlabel("Sample index in PSS window")
    plt.ylabel("Unwrapped phase (rad)")
    plt.legend()
    plt.tight_layout()
    fig.savefig(out_dir / "fig8_phase_before_after_comp.png", dpi=150)
    plt.close(fig)

    # Figure 9: correlation overlay after compensation
    fig = plt.figure(figsize=(11, 4.8))
    for k in range(3):
        c = np.abs(res_fo_comp["corr_all"][k])
        l = res_fo_comp["lags_all"][k]
        plt.plot(l, c / (np.max(c) + 1e-12), lw=1.0, label=f"ref col {k+1}")
    plt.title("Cross-correlation on recv_signal_freq_offset (compensated)")
    plt.xlabel("Lag (samples)")
    plt.ylabel("Normalized |xcorr|")
    plt.legend()
    plt.tight_layout()
    fig.savefig(out_dir / "fig9_corr_freq_offset_comp.png", dpi=150)
    plt.close(fig)

    summary = {
        "fs_hz": fs,
        "normal": {
            "best_template_idx_1based": res_normal["best_template_idx_1based"],
            "assumed_npss": res_normal["assumed_npss"],
            "toa_lag": res_normal["toa_lag"],
            "toa_sample": res_normal["toa_sample"],
            "toa_time_us": res_normal["toa_time_us"],
            "peak_abs": res_normal["peak_abs"],
            "peak_metric": res_normal["peak_metric"],
        },
        "freq_offset_raw": {
            "best_template_idx_1based": res_fo_raw["best_template_idx_1based"],
            "assumed_npss": res_fo_raw["assumed_npss"],
            "toa_lag": res_fo_raw["toa_lag"],
            "toa_sample": res_fo_raw["toa_sample"],
            "toa_time_us": res_fo_raw["toa_time_us"],
            "peak_abs": res_fo_raw["peak_abs"],
            "peak_metric": res_fo_raw["peak_metric"],
        },
        "cfo_solution": {
            "selected_template_idx_1based": cfo_best["template_idx_1based"],
            "estimated_cfo_hz": cfo_best["cfo_hz"],
            "raw_peak_metric": cfo_best["raw_peak_metric"],
            "comp_peak_metric": cfo_best["comp_peak_metric"],
        },
        "freq_offset_compensated": {
            "best_template_idx_1based": res_fo_comp["best_template_idx_1based"],
            "assumed_npss": res_fo_comp["assumed_npss"],
            "toa_lag": res_fo_comp["toa_lag"],
            "toa_sample": res_fo_comp["toa_sample"],
            "toa_time_us": res_fo_comp["toa_time_us"],
            "peak_abs": res_fo_comp["peak_abs"],
            "peak_metric": res_fo_comp["peak_metric"],
        },
    }

    with open(root / "results_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Figures saved to: {out_dir}")
    print(f"Summary saved to: {root / 'results_summary.json'}")


if __name__ == "__main__":
    main()
