%% 5G Lab: Spectrum analysis, TOA estimation, PSS detection, and large-CFO solution
% Author: Codex
% Data:
%   - recv_signal              (460800x1 complex)
%   - recv_signal_freq_offset  (460800x1 complex)
%   - ref_signal_grid          (256x3 complex)
%
% This script completes:
% 1) Spectrum visualization
% 2) TOA estimation by xcorr peak
% 3) PSS decision by comparing 3 references
% 4) Further discussion: large frequency-offset compensation
%
% Note:
%   The script assumes reference column 1/2/3 maps to N_PSS = 0/1/2.

clear; clc; close all;

%% ----------------------------- Config -----------------------------
Fs = 7.68e6;       % sample rate (Hz)
Ts = 1 / Fs;
save_figures = true;
fig_dir = "figures_matlab";
if save_figures && ~exist(fig_dir, "dir")
    mkdir(fig_dir);
end

%% ----------------------------- Load data -----------------------------
data = load("data_5G.mat");
rx = data.recv_signal(:);
rx_fo = data.recv_signal_freq_offset(:);
ref_grid = data.ref_signal_grid;

assert(size(ref_grid, 2) == 3, "ref_signal_grid must be 256x3.");
assert(size(ref_grid, 1) == 256, "ref_signal_grid length must be 256.");

fprintf("============================================\n");
fprintf("5G Lab Processing Started\n");
fprintf("Signal length: %d samples\n", length(rx));
fprintf("Sampling rate: %.2f MHz\n", Fs / 1e6);
fprintf("============================================\n\n");

%% ----------------------------- Figure 1: Time-domain signal -----------------------------
N_show = min(3000, length(rx));
t_show = (0:N_show-1) * Ts * 1e3; % ms

f1 = figure("Name", "Time-domain waveform", "Color", "w");
tiledlayout(2, 1);
nexttile;
plot(t_show, real(rx(1:N_show)), "LineWidth", 1.0); hold on;
plot(t_show, imag(rx(1:N_show)), "LineWidth", 1.0); hold off;
grid on;
xlabel("Time (ms)");
ylabel("Amplitude");
title("recv\_signal (first 3000 samples)");
legend("Real", "Imag", "Location", "best");

nexttile;
plot(t_show, real(rx_fo(1:N_show)), "LineWidth", 1.0); hold on;
plot(t_show, imag(rx_fo(1:N_show)), "LineWidth", 1.0); hold off;
grid on;
xlabel("Time (ms)");
ylabel("Amplitude");
title("recv\_signal\_freq\_offset (first 3000 samples)");
legend("Real", "Imag", "Location", "best");
if save_figures
    save_figure(f1, fig_dir, "fig1_time_domain.png");
end

%% ----------------------------- Figure 2: PSD -----------------------------
[f_rx, p_rx_db] = estimate_psd(rx, Fs);
[f_fo, p_fo_db] = estimate_psd(rx_fo, Fs);

f2 = figure("Name", "Power Spectral Density", "Color", "w");
plot(f_rx / 1e6, p_rx_db, "LineWidth", 1.1); hold on;
plot(f_fo / 1e6, p_fo_db, "LineWidth", 1.1); hold off;
grid on;
xlabel("Frequency (MHz)");
ylabel("PSD (dB)");
title("PSD comparison");
legend("recv\_signal", "recv\_signal\_freq\_offset", "Location", "best");
if save_figures
    save_figure(f2, fig_dir, "fig2_psd.png");
end

%% ----------------------------- Part A: TOA + PSS on recv_signal -----------------------------
res_normal = detect_toa_pss(rx, ref_grid, Fs);
print_detection("recv_signal", res_normal, Ts);

f3 = figure("Name", "Correlation on recv_signal", "Color", "w");
plot_corr_overlay(res_normal.corr_all, res_normal.lags_all, "recv_signal");
if save_figures
    save_figure(f3, fig_dir, "fig3_corr_recv_signal.png");
end

%% ----------------------------- Part B: Direct detection on large-CFO signal -----------------------------
res_fo_raw = detect_toa_pss(rx_fo, ref_grid, Fs);
print_detection("recv_signal_freq_offset (raw)", res_fo_raw, Ts);

f4 = figure("Name", "Raw correlation on large-CFO signal", "Color", "w");
plot_corr_overlay(res_fo_raw.corr_all, res_fo_raw.lags_all, "recv_signal\_freq\_offset (raw)");
if save_figures
    save_figure(f4, fig_dir, "fig4_corr_freq_offset_raw.png");
end

%% ----------------------------- Part C: Further discussion solution -----------------------------
% Solve the large-frequency-offset problem:
% 1) rough lag from xcorr
% 2) CFO coarse/fine grid search
% 3) compensate exp(-j2piDf n/Fs)
% 4) re-run TOA + PSS

cfo_cfg = struct();
cfo_cfg.coarse_min_hz = -500e3;
cfo_cfg.coarse_max_hz = 500e3;
cfo_cfg.coarse_step_hz = 2e3;
cfo_cfg.fine_span_hz = 4e3;
cfo_cfg.fine_step_hz = 100;

[rx_fo_comp, cfo_result] = solve_large_cfo(rx_fo, ref_grid, Fs, cfo_cfg);
res_fo_comp = detect_toa_pss(rx_fo_comp, ref_grid, Fs);
print_detection("recv_signal_freq_offset (after CFO compensation)", res_fo_comp, Ts);

fprintf("CFO solution summary:\n");
fprintf("  Selected reference column for CFO estimation: %d\n", cfo_result.best_template_idx);
fprintf("  Estimated CFO: %.2f Hz\n", cfo_result.best_cfo_hz);
fprintf("  Raw peak metric: %.3f\n", cfo_result.raw_peak_metric);
fprintf("  Compensated peak metric: %.3f\n\n", cfo_result.comp_peak_metric);

% Figure 5: CFO search curve
f5 = figure("Name", "CFO search metric", "Color", "w");
plot(cfo_result.df_grid_hz / 1e3, cfo_result.metric_grid, "LineWidth", 1.1);
grid on;
xlabel("Candidate CFO (kHz)");
ylabel("Coherent correlation metric");
title(sprintf("CFO search curve (template col %d)", cfo_result.best_template_idx));
if save_figures
    save_figure(f5, fig_dir, "fig5_cfo_search_metric.png");
end

% Figure 6: Best-template correlation before/after compensation
f6 = figure("Name", "Best-template correlation comparison", "Color", "w");
[c_raw_best, l_raw_best] = xcorr(rx_fo, ref_grid(:, cfo_result.best_template_idx));
[c_cmp_best, l_cmp_best] = xcorr(rx_fo_comp, ref_grid(:, cfo_result.best_template_idx));
plot(l_raw_best, abs(c_raw_best) / (max(abs(c_raw_best)) + eps), "LineWidth", 1.0); hold on;
plot(l_cmp_best, abs(c_cmp_best) / (max(abs(c_cmp_best)) + eps), "LineWidth", 1.0); hold off;
grid on;
xlabel("Lag (samples)");
ylabel("Normalized |xcorr|");
title(sprintf("Best template col %d: before vs after compensation", cfo_result.best_template_idx));
legend("Before compensation", "After compensation", "Location", "best");
if save_figures
    save_figure(f6, fig_dir, "fig6_corr_before_after_comp.png");
end

% Figure 7: phase evolution before/after compensation
lag_raw = cfo_result.rough_lag;
seg_start = max(1, min(length(rx_fo) - size(ref_grid,1) + 1, lag_raw + 1));
seg_raw = rx_fo(seg_start:seg_start + size(ref_grid,1) - 1);
seg_cmp = rx_fo_comp(seg_start:seg_start + size(ref_grid,1) - 1);
ref_best = ref_grid(:, cfo_result.best_template_idx);
phi_raw = unwrap(angle(seg_raw .* conj(ref_best)));
phi_cmp = unwrap(angle(seg_cmp .* conj(ref_best)));

f7 = figure("Name", "Phase evolution comparison", "Color", "w");
n = 0:length(ref_best)-1;
plot(n, phi_raw, "LineWidth", 1.0); hold on;
plot(n, phi_cmp, "LineWidth", 1.0); hold off;
grid on;
xlabel("Sample index inside PSS window");
ylabel("Unwrapped phase (rad)");
title("Phase drift before/after compensation");
legend("Before compensation", "After compensation", "Location", "best");
if save_figures
    save_figure(f7, fig_dir, "fig7_phase_before_after_comp.png");
end

% Figure 8: final correlation overlay after compensation
f8 = figure("Name", "Correlation after CFO compensation", "Color", "w");
plot_corr_overlay(res_fo_comp.corr_all, res_fo_comp.lags_all, "recv\_signal\_freq\_offset (compensated)");
if save_figures
    save_figure(f8, fig_dir, "fig8_corr_freq_offset_comp.png");
end

fprintf("Done. Generated figures are saved in: %s\n", fig_dir);

%% ----------------------------- Local functions -----------------------------
function [f_axis, p_db] = estimate_psd(x, Fs)
    nfft = 4096;
    x = x(:);
    if exist("pwelch", "file")
        win = hann(1024);
        [p, f] = pwelch(x, win, round(0.5 * numel(win)), nfft, Fs, "twosided");
        p = fftshift(p);
        f = fftshift(f);
    else
        x2 = x(1:min(length(x), 8 * nfft));
        X = fftshift(fft(x2, nfft));
        p = (abs(X).^2) / max(1, length(x2));
        f = linspace(-Fs/2, Fs/2, nfft).';
    end
    p_db = 10 * log10(p + eps);
    f_axis = f;
end

function res = detect_toa_pss(rx, ref_grid, Fs)
    num_ref = size(ref_grid, 2);
    peak_abs = zeros(num_ref, 1);
    peak_lag = zeros(num_ref, 1);
    peak_metric = zeros(num_ref, 1);
    corr_all = cell(num_ref, 1);
    lags_all = cell(num_ref, 1);

    for k = 1:num_ref
        ref = ref_grid(:, k);
        [c, lags] = xcorr(rx, ref);
        a = abs(c);
        [peak_abs(k), idx] = max(a);
        peak_lag(k) = lags(idx);
        peak_metric(k) = peak_abs(k) / (median(a) + eps);
        corr_all{k} = c;
        lags_all{k} = lags;
    end

    [~, best_idx] = max(peak_abs);
    toa_sample = peak_lag(best_idx) + 1;
    toa_sample = max(1, min(length(rx), toa_sample));

    res = struct();
    res.best_template_idx = best_idx;
    res.assumed_npss = best_idx - 1; % col1/2/3 -> N_PSS=0/1/2
    res.toa_lag = peak_lag(best_idx);
    res.toa_sample = toa_sample;
    res.toa_time_s = toa_sample / Fs;
    res.peak_abs = peak_abs;
    res.peak_metric = peak_metric;
    res.peak_lag_all = peak_lag;
    res.corr_all = corr_all;
    res.lags_all = lags_all;
end

function plot_corr_overlay(corr_all, lags_all, figure_name)
    colors = lines(length(corr_all));
    hold on;
    for k = 1:length(corr_all)
        c = corr_all{k};
        l = lags_all{k};
        a = abs(c);
        plot(l, a / (max(a) + eps), "Color", colors(k, :), "LineWidth", 1.0, ...
            "DisplayName", sprintf("ref col %d", k));
    end
    hold off;
    grid on;
    xlabel("Lag (samples)");
    ylabel("Normalized |xcorr|");
    title(["Cross-correlation magnitude - " figure_name]);
    legend("Location", "best");
end

function print_detection(tag, res, Ts)
    fprintf("---- %s ----\n", tag);
    fprintf("Best template column: %d\n", res.best_template_idx);
    fprintf("Assumed N_PSS: %d\n", res.assumed_npss);
    fprintf("TOA lag: %d samples\n", res.toa_lag);
    fprintf("TOA sample index (1-based): %d\n", res.toa_sample);
    fprintf("TOA time: %.6f us\n", res.toa_sample * Ts * 1e6);
    fprintf("Peak abs (col1..3): [%.3f, %.3f, %.3f]\n", ...
        res.peak_abs(1), res.peak_abs(2), res.peak_abs(3));
    fprintf("Peak metric (col1..3): [%.3f, %.3f, %.3f]\n\n", ...
        res.peak_metric(1), res.peak_metric(2), res.peak_metric(3));
end

function [rx_comp_best, out] = solve_large_cfo(rx_fo, ref_grid, Fs, cfg)
    num_ref = size(ref_grid, 2);
    n_all = (0:length(rx_fo)-1).';

    best_score = -inf;
    rx_comp_best = rx_fo;

    out = struct();
    out.best_template_idx = 1;
    out.best_cfo_hz = 0;
    out.raw_peak_metric = 0;
    out.comp_peak_metric = 0;
    out.df_grid_hz = [];
    out.metric_grid = [];
    out.rough_lag = 0;

    for k = 1:num_ref
        ref = ref_grid(:, k);

        % Rough lag from raw correlation
        [c_raw, l_raw] = xcorr(rx_fo, ref);
        a_raw = abs(c_raw);
        [peak_raw, idx_raw] = max(a_raw);
        lag_raw = l_raw(idx_raw);
        raw_metric = peak_raw / (median(a_raw) + eps);

        st = lag_raw + 1;
        st = max(1, min(length(rx_fo) - length(ref) + 1, st));
        seg = rx_fo(st:st + length(ref) - 1);
        n = (0:length(ref)-1).';

        % Coarse CFO grid search
        df_coarse = (cfg.coarse_min_hz:cfg.coarse_step_hz:cfg.coarse_max_hz).';
        metric_coarse = zeros(length(df_coarse), 1);
        for ii = 1:length(df_coarse)
            df = df_coarse(ii);
            seg_comp = seg .* exp(-1j * 2 * pi * df * n / Fs);
            metric_coarse(ii) = abs(sum(seg_comp .* conj(ref)));
        end
        [~, i_coarse] = max(metric_coarse);
        df0 = df_coarse(i_coarse);

        % Fine CFO search around the coarse maximum
        df_fine = ((df0 - cfg.fine_span_hz):cfg.fine_step_hz:(df0 + cfg.fine_span_hz)).';
        metric_fine = zeros(length(df_fine), 1);
        for ii = 1:length(df_fine)
            df = df_fine(ii);
            seg_comp = seg .* exp(-1j * 2 * pi * df * n / Fs);
            metric_fine(ii) = abs(sum(seg_comp .* conj(ref)));
        end
        [~, i_fine] = max(metric_fine);
        df_est = df_fine(i_fine);

        % Compensate entire signal and evaluate
        rx_comp = rx_fo .* exp(-1j * 2 * pi * df_est * n_all / Fs);
        [c_cmp, ~] = xcorr(rx_comp, ref);
        a_cmp = abs(c_cmp);
        cmp_metric = max(a_cmp) / (median(a_cmp) + eps);

        if cmp_metric > best_score
            best_score = cmp_metric;
            rx_comp_best = rx_comp;
            out.best_template_idx = k;
            out.best_cfo_hz = df_est;
            out.raw_peak_metric = raw_metric;
            out.comp_peak_metric = cmp_metric;
            out.df_grid_hz = df_fine;
            out.metric_grid = metric_fine;
            out.rough_lag = lag_raw;
        end
    end
end

function save_figure(fig_handle, fig_dir, file_name)
    drawnow;
    out_path = fullfile(fig_dir, file_name);
    if exist("exportgraphics", "file")
        exportgraphics(fig_handle, out_path, "Resolution", 150);
    else
        saveas(fig_handle, out_path);
    end
end
