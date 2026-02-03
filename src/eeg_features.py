# -*- coding: utf-8 -*-
"""EEG Feature Extraction (46 features across 3 spatial scales)

Note: The original paper reported 45 EEG features. Following reviewer feedback,
theta-gamma phase-amplitude coupling (PAC) was added to the Appendix.
"""

import os
import logging
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import scipy.signal
import scipy.stats
import pywt
import antropy
import mne
from mne.time_frequency import psd_array_multitaper
from scipy.signal import hilbert, butter, filtfilt

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

# Parameters
FREQ_BANDS = {'delta': (1, 4), 'theta': (4, 8), 'alpha': (8, 13), 
              'beta': (13, 30), 'gamma': (30, 45)}
ERP_WINDOWS = {'N100': (0.08, 0.12), 'P100': (0.09, 0.13), 'P200': (0.15, 0.25),
               'P300': (0.25, 0.45), 'N400': (0.30, 0.50)}
CHANNEL_GROUPS = {
    'F': ['Fp1', 'Fp2', 'F3', 'F4', 'F7', 'F8', 'Fz', 'FC1', 'FC2', 'FC5', 'FC6'],
    'C': ['C3', 'C4', 'Cz', 'CP1', 'CP2', 'CP5', 'CP6'],
    'T': ['FT9', 'FT10', 'T7', 'T8', 'TP9', 'TP10'],
    'P': ['P3', 'P4', 'P7', 'P8', 'Pz'],
    'O': ['O1', 'O2', 'Oz'],
}


def bandpass(signal, low, high, sfreq, order=4):
    nyq = sfreq / 2
    b, a = butter(order, [max(low/nyq, 0.001), min(high/nyq, 0.999)], btype='band')
    return filtfilt(b, a, signal)


def pac_mi(signal, sfreq, phase_band, amp_band, n_bins=18):
    """Phase-Amplitude Coupling Modulation Index."""
    if len(signal) < 50:
        return np.nan
    try:
        phase_sig = bandpass(signal, phase_band[0], phase_band[1], sfreq)
        amp_sig = bandpass(signal, amp_band[0], amp_band[1], sfreq)
        theta = np.angle(hilbert(phase_sig))
        gamma = np.abs(hilbert(amp_sig))
        edges = np.linspace(-np.pi, np.pi, n_bins + 1)
        amp_per_bin = np.array([np.mean(gamma[(theta >= edges[i]) & (theta < edges[i+1])]) 
                                for i in range(n_bins)])
        if np.sum(amp_per_bin) == 0:
            return np.nan
        P = amp_per_bin / np.sum(amp_per_bin)
        P = np.clip(P, 1e-10, None)
        return np.sum(P * np.log(P * n_bins)) / np.log(n_bins)
    except:
        return np.nan


def extract_features(signal, sfreq, times):
    """Extract 46 features from a signal."""
    f = {}
    
    # Time domain (9)
    f['time_mean'] = np.mean(signal)
    f['time_median'] = np.median(signal)
    f['time_std'] = np.std(signal)
    f['time_variance'] = np.var(signal)
    f['time_mean_square'] = np.mean(signal**2)
    f['time_skewness'] = scipy.stats.skew(signal)
    f['time_kurtosis'] = scipy.stats.kurtosis(signal)
    try:
        mob, comp = antropy.hjorth_params(signal)
    except:
        mob, comp = np.nan, np.nan
    f['time_hjorth_mobility'] = mob
    f['time_hjorth_complexity'] = comp
    
    # Frequency domain (12)
    try:
        psds, freqs = psd_array_multitaper(
            signal.reshape(1, 1, -1), sfreq=sfreq, fmin=1.0, fmax=45.0,
            bandwidth=2.0*(sfreq/len(signal)), adaptive=True, n_jobs=1, verbose=False)
        psd = psds.squeeze()
    except:
        psd, freqs = np.array([0]), np.array([0])
    
    total = 0
    for band, (lo, hi) in FREQ_BANDS.items():
        mask = (freqs >= lo) & (freqs <= hi)
        power = np.trapezoid(psd[mask], freqs[mask]) if mask.sum() > 1 else 0
        f[f'freq_abs_{band}'] = power
        total += power
    for band in FREQ_BANDS:
        f[f'freq_rel_{band}'] = f[f'freq_abs_{band}'] / max(total, 1e-10)
    f['freq_peak'] = freqs[np.argmax(psd)] if len(psd) > 0 else np.nan
    cumsum = np.cumsum(psd)
    f['freq_median'] = freqs[np.searchsorted(cumsum, cumsum[-1]/2)] if len(psd) > 0 else np.nan
    
    # ERP (15)
    for name, (tmin, tmax) in ERP_WINDOWS.items():
        mask = (times >= tmin) & (times <= tmax)
        if mask.sum() > 0:
            win = signal[mask]
            f[f'erp_mean_{name}'] = np.mean(win)
            idx = np.argmin(win) if 'N' in name else np.argmax(win)
            f[f'erp_peak_amp_{name}'] = win[idx]
            f[f'erp_peak_lat_{name}'] = times[mask][idx]
        else:
            f[f'erp_mean_{name}'] = f[f'erp_peak_amp_{name}'] = f[f'erp_peak_lat_{name}'] = np.nan
    
    # Wavelet (5)
    try:
        max_lv = min(5, pywt.dwt_max_level(len(signal), 'db4'))
        coeffs = pywt.wavedec(signal, 'db4', level=max_lv)
        for i, c in enumerate(coeffs[1:], 1):
            f[f'wavelet_var_d{max_lv-i+1}'] = np.var(c)
    except:
        for i in range(1, 6):
            f[f'wavelet_var_d{i}'] = np.nan
    
    # Nonlinear (4)
    try:
        f['nonlinear_sampen'] = antropy.sample_entropy(signal)
    except:
        f['nonlinear_sampen'] = np.nan
    try:
        f['nonlinear_petrosian_fd'] = antropy.petrosian_fd(signal)
    except:
        f['nonlinear_petrosian_fd'] = np.nan
    try:
        f['nonlinear_katz_fd'] = antropy.katz_fd(signal)
    except:
        f['nonlinear_katz_fd'] = np.nan
    try:
        f['nonlinear_higuchi_fd'] = antropy.higuchi_fd(signal, kmax=min(10, len(signal)//2))
    except:
        f['nonlinear_higuchi_fd'] = np.nan
    
    # PAC (1)
    f['pac_theta_gamma_mi'] = pac_mi(signal, sfreq, (4, 8), (30, 45))
    
    return f


def extract_multi_scale(epoch_data, sfreq, times, ch_names):
    """Extract features at global, region, and channel levels."""
    ch_idx = {n: i for i, n in enumerate(ch_names)}
    
    # Global (WB)
    wb = extract_features(np.mean(epoch_data, axis=0), sfreq, times)
    global_f = [{'location': 'WB', 'features': wb}]
    
    # Region
    region_f = []
    for region, chs in CHANNEL_GROUPS.items():
        idx = [ch_idx[c] for c in chs if c in ch_idx]
        if idx:
            rf = extract_features(np.mean(epoch_data[idx], axis=0), sfreq, times)
            region_f.append({'location': region, 'features': rf})
    
    # Channel
    channel_f = []
    for i, ch in enumerate(ch_names):
        cf = extract_features(epoch_data[i], sfreq, times)
        channel_f.append({'location': ch, 'features': cf})
    
    return global_f, region_f, channel_f


def extract_eeg_features(epochs_path: str, output_dir: str, 
                         tmin: float = 0.0, tmax: float = 0.5) -> bool:
    """Extract EEG features from MNE Epochs file."""
    if not os.path.exists(epochs_path):
        return False
    
    try:
        mne.set_log_level('ERROR')
        epochs = mne.read_epochs(epochs_path, preload=True, verbose=False)
        epochs.crop(tmin=tmin, tmax=tmax)
        sfreq, times, ch_names = epochs.info['sfreq'], epochs.times, epochs.ch_names
        
        all_g, all_r, all_c = [], [], []
        for i in range(len(epochs)):
            token_id = int(epochs.events[i, 2])
            data = epochs[i].get_data(copy=False).squeeze()
            g, r, c = extract_multi_scale(data, sfreq, times, ch_names)
            
            for item in g:
                for k, v in item['features'].items():
                    all_g.append({'token_id': token_id, 'location': item['location'], 
                                 'feature_name': k, 'value': v})
            for item in r:
                for k, v in item['features'].items():
                    all_r.append({'token_id': token_id, 'location': item['location'],
                                 'feature_name': k, 'value': v})
            for item in c:
                for k, v in item['features'].items():
                    all_c.append({'token_id': token_id, 'location': item['location'],
                                 'feature_name': k, 'value': v})
        
        os.makedirs(output_dir, exist_ok=True)
        pd.DataFrame(all_g).to_csv(os.path.join(output_dir, "global_features.csv"), index=False)
        pd.DataFrame(all_r).to_csv(os.path.join(output_dir, "region_features.csv"), index=False)
        pd.DataFrame(all_c).to_csv(os.path.join(output_dir, "channel_features.csv"), index=False)
        return True
    except Exception as e:
        logger.error(f"EEG extraction failed: {e}")
        return False


def get_feature_names():
    """Return list of 46 feature names."""
    names = []
    names += ['time_mean', 'time_median', 'time_std', 'time_variance', 'time_mean_square',
              'time_skewness', 'time_kurtosis', 'time_hjorth_mobility', 'time_hjorth_complexity']
    for b in FREQ_BANDS:
        names += [f'freq_abs_{b}', f'freq_rel_{b}']
    names += ['freq_peak', 'freq_median']
    for e in ERP_WINDOWS:
        names += [f'erp_mean_{e}', f'erp_peak_amp_{e}', f'erp_peak_lat_{e}']
    names += [f'wavelet_var_d{i}' for i in range(1, 6)]
    names += ['nonlinear_sampen', 'nonlinear_petrosian_fd', 'nonlinear_katz_fd', 'nonlinear_higuchi_fd']
    names += ['pac_theta_gamma_mi']
    return names
