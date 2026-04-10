import xarray as xr
import xrft
import numpy as np
import logging
from dask.diagnostics import ProgressBar
import matplotlib.pyplot as plt
from . import transforms

# STANDARDS FROM INFERENCE, NEED TO REVISIT THIS
STANDARDS  = {'mean_ssh': 0,
             'std_ssh': 0.0453692672359483,
             'mean_sst': 15.956900367755182,
             'std_sst': 5.5,
             'extra_mean_tuning': 1.5}



def npfft_isotropic_psd(arr):
    """
    Parameters:
        arr (np.ndarray): 2D input array.

    Returns:
        radialprofile (np.ndarray): Radially averaged PSD.
    """
    # Compute 2D FFT and shift zero frequency to center
    fft2 = np.fft.fftshift(np.fft.fft2(arr))
    psd2D = np.abs(fft2)**2
    # Create indices grid
    y, x = np.indices(psd2D.shape)
    center = np.array([ (x.max()-x.min())/2.0, (y.max()-y.min())/2.0 ])
    # Compute radial distance from center for each pixel
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    # Convert radius to integer bins
    r = r.astype(int)
    # Compute the sum of PSD values in each radial bin and count pixels per bin
    tbin = np.bincount(r.ravel(), psd2D.ravel())
    nr = np.bincount(r.ravel())
    # Avoid division by zero; compute the radial profile (average)
    radialprofile = tbin / (nr + np.finfo(float).eps)
    return radialprofile[:int(psd2D.shape[-1]/2)]
    
'''
def get_cutoff_freq(ds):
    """
    Parameters
    ----------
    ds : xarray.Dataset
        Must contain:
            - WPSD_mean(time, freq_r)
            - freq_r coordinate
    Returns
    -------
    ds : xarray.Dataset
        With added variables:
            - freq_above_cutoff(time)
            - freq_cutoff_interp(time)
    """
    thr = 0.5
    # Prepare output arrays
    freq_above = xr.zeros_like(ds.WPSD_mean.mean(dim="freq_r"))
    freq_interp = xr.zeros_like(freq_above)
    freqs = ds.freq_r.values
    ntime = ds.sizes["time"]
    for t in range(ntime):
        vals = ds.WPSD_mean.isel(time=t).values
        # Index where threshold is first crossed
        idx = np.argmax(vals < thr)
        # Case 1 — threshold never crossed (all values >= 0.5)
        if vals[idx] >= thr:
            freq_above[t] = freqs[-1]
            freq_interp[t] = freqs[-1]
            continue
        # Otherwise, threshold is crossed at idx
        if idx > 0:
            # Above and below values
            f1, f2 = freqs[idx-1], freqs[idx]
            v1, v2 = vals[idx-1], vals[idx]
            # 1) Save above cutoff (original behavior)
            freq_above[t] = f1
            # 2) Linear interpolation for exact cutoff
            frac = (thr - v1) / (v2 - v1)
            freq_interp[t] = f1 + frac * (f2 - f1)
        else:
            # Threshold crossed at the first index
            freq_above[t] = freqs[0]
            freq_interp[t] = freqs[0]
    # Add to dataset as new variables
    ds["freq_above_cutoff_WPSD_mean"] = freq_above.assign_attrs({
        "description": "Highest freq_r where WPSD_mean >= 0.5",
        "units": ds.freq_r.attrs.get("units", "")
    })
    ds["freq_cutoff_interp_WPSD_mean"] = freq_interp.assign_attrs({
        "description": "Linearly interpolated freq_r where WPSD_mean = 0.5",
        "units": ds.freq_r.attrs.get("units", "")
    })

    return ds
'''
def _compute_cutoff(vals, freqs, thr=0.5):
    """
    vals: (..., freq) ndarray
    freqs: (freq,) ndarray
    Returns:
        above_cutoff: (...,) last freq where vals >= thr
        interp_cutoff: (...,) interpolated freq where vals crosses thr
    """
    # shape: (..., freq)
    below = vals < thr
    # index of first True along last axis
    idx = below.argmax(axis=-1)
    # case: threshold never crossed → idx==0 and vals[...,0]>=thr
    never_cross = (below.sum(axis=-1) == 0)
    # f_above = freqs[idx-1], but clip idx-1≥0
    idx_above = np.clip(idx - 1, 0, len(freqs)-1)
    f_above = freqs[idx_above]
    # get above/below values
    v1 = np.take_along_axis(vals, idx_above[..., None], axis=-1)[..., 0]
    v2 = np.take_along_axis(vals, idx[..., None], axis=-1)[..., 0]
    f2 = freqs[idx]
    # linear interpolation
    frac = np.where(
        v2 != v1,
        (thr - v1) / (v2 - v1),
        0.0,  # fallback (should not occur unless flat)
    )
    f_interp = f_above + frac * (f2 - f_above)
    # never-cross case → set to highest frequency
    f_above = np.where(never_cross, freqs[-1], f_above)
    f_interp = np.where(never_cross, freqs[-1], f_interp)
    return f_above, f_interp


def get_cutoff_freq(ds):
    """
    Adds:
        - freq_above_cutoff_mean(time)
        - freq_cutoff_interp_mean(time)
        - freq_above_cutoff_sample(time, sample)
        - freq_cutoff_interp_sample(time, sample)
    """
    freqs = ds.freq_r.values
    # --------------------------
    # Mean WPSD
    # --------------------------
    above_mean, interp_mean = xr.apply_ufunc(
        _compute_cutoff,
        ds.WPSD_mean,
        xr.DataArray(freqs, dims=["freq_r"]),
        input_core_dims=[["freq_r"], ["freq_r"]],
        output_core_dims=[[], []],
        vectorize=True,
        output_dtypes=[float, float],
    )
    ds["freq_above_cutoff_mean"] = above_mean.assign_attrs({
        "description": "last freq where WPSD_mean ≥ 0.5"
    })
    ds["freq_cutoff_interp_mean"] = interp_mean.assign_attrs({
        "description": "linearly interpolated cutoff freq for WPSD_mean"
    })
    # --------------------------
    # Sample-wise WPSD
    # --------------------------
    above_sample, interp_sample = xr.apply_ufunc(
        _compute_cutoff,
        ds.WPSD_sample,
        xr.DataArray(freqs, dims=["freq_r"]),
        input_core_dims=[["freq_r"], ["freq_r"]],
        output_core_dims=[[], []],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float, float],
    )
    ds["freq_above_cutoff_sample"] = above_sample.assign_attrs({
        "description": "last freq where WPSD_sample ≥ 0.5"
    })
    ds["freq_cutoff_interp_sample"] = interp_sample.assign_attrs({
        "description": "linearly interpolated cutoff freq for each sample"
    })
    return ds
    
############################################################################################################
# Scores from 2020 SSH mapping challenge (Le Guillou, 2021)
############################################################################################################

def rmse_based_scores(x_new, x_true):
    #logging.info('     Compute RMSE-based scores...')
    print('     Compute RMSE-based scores...')
    # RMSE(t) based score
    rmse_t = 1.0 - (((x_new - x_true)**2).mean(dim=('x', 'y')))**0.5/(((x_true)**2).mean(dim=('x', 'y')))**0.5
    # RMSE(x, y) based score
    rmse_xy = (((x_new - x_true)**2).mean(dim=('time')))**0.5
    rmse_t = rmse_t.rename('rmse_t')
    rmse_xy = rmse_xy.rename('rmse_xy')
    # Temporal stability of the error
    reconstruction_error_stability_metric = rmse_t.std().values
    # Show leaderboard SSH-RMSE metric (spatially and time averaged normalized RMSE)
    leaderboard_rmse = 1.0 - (((x_new - x_true) ** 2).mean()) ** 0.5 / (
        ((x_true) ** 2).mean()) ** 0.5
    logging.info('          => Leaderboard SSH RMSE score = %s', np.round(leaderboard_rmse.values, 4))
    logging.info('          Error variability = %s (temporal stability of the mapping error)', np.round(reconstruction_error_stability_metric, 4))
    print('          => Leaderboard SSH RMSE score = %s', np.round(leaderboard_rmse.values, 4))
    print('          Error variability = %s (temporal stability of the mapping error)', np.round(reconstruction_error_stability_metric, 4))
    return rmse_t, rmse_xy, np.round(leaderboard_rmse.values, 4), np.round(reconstruction_error_stability_metric, 4)


def psd_based_scores(x_new, x_true):
    #logging.info('     Compute PSD-based scores...')
    print('     Compute PSD-based scores...')
    with ProgressBar():
        # Compute error = SSH_reconstruction - SSH_true
        if "time_i" in x_new.coords:
            x_new = x_new.drop(['time_i'])
        if "time_i" in x_true.coords:
            x_true = x_true.drop(['time_i'])   
        err = (x_new - x_true)
        err = err.chunk({"y":1, 'time': err['time'].size, 'x': err['x'].size})
        # make time vector in days units 
        err['time'] = (err.time - err.time[0]) / np.timedelta64(1, 'D')
        # Rechunk SSH_true
        signal = x_true.chunk({"y":1, 'time': x_true['time'].size, 'x': x_true['x'].size})
        # make time vector in days units
        signal['time'] = (signal.time - signal.time[0]) / np.timedelta64(1, 'D')
        # Compute PSD_err and PSD_signal
        psd_err = xrft.power_spectrum(err, dim=['time', 'x'], detrend='constant', window=True).compute()
        psd_signal = xrft.power_spectrum(signal, dim=['time', 'x'], detrend='constant', window=True).compute()
        # Averaged over latitude
        mean_psd_signal = psd_signal.mean(dim='y').where((psd_signal.freq_x > 0.) & (psd_signal.freq_time > 0), drop=True)
        mean_psd_err = psd_err.mean(dim='y').where((psd_err.freq_x > 0.) & (psd_err.freq_time > 0), drop=True)
        # return PSD-based score
        psd_based_score = (1.0 - mean_psd_err/mean_psd_signal)
        # Find the key metrics: shortest temporal & spatial scales resolved based on the 0.5 contour criterion of the PSD_score
        level = [0.5]
        cs = plt.contour(1./psd_based_score.freq_x.values,1./psd_based_score.freq_time.values, psd_based_score, level)
        # --- Modern extraction of contour vertices ---
        paths = cs.get_paths()
        if len(paths) == 0:
            raise RuntimeError("No contour found at level 0.5")
        # Take the first path in the first (and only) contour level
        verts = paths[0].vertices
        x05, y05 = verts[:, 0], verts[:, 1]
        plt.close()
        
        shortest_spatial_wavelength_resolved = np.min(x05)
        shortest_temporal_wavelength_resolved = np.min(y05)
        logging.info('          => Leaderboard Spectral score = %s (kilometers)',
                     np.round(shortest_spatial_wavelength_resolved, 2))
        logging.info('          => shortest temporal wavelength resolved = %s (days)',
                     np.round(shortest_temporal_wavelength_resolved, 2))
        print('          => Leaderboard Spectral score = %s (kilometersn)',
                     np.round(shortest_spatial_wavelength_resolved, 2))
        print('          => shortest temporal wavelength resolved = %s (days)',
                     np.round(shortest_temporal_wavelength_resolved, 2))

        return (1.0 - mean_psd_err/mean_psd_signal), np.round(shortest_spatial_wavelength_resolved, 2), np.round(shortest_temporal_wavelength_resolved, 2)


def denorm_ssh(field, time_indices=None):
    return field*STANDARDS["std_ssh"] + STANDARDS["mean_ssh"]


def denorm_sst(field, time_indices):
    sst_normalizer = transforms.SeasonalSSTNormalizer(std=STANDARDS["std_sst"],extra_mean_tuning=STANDARDS["extra_mean_tuning"])
    return sst_normalizer.denormalize(field, time_indices)

    
def denorm_null(field,):
    return field


def PSD_WPSD_metrics(test,
                     score_type = "xrft",
                     # xrft isospec params, same as Le Guillou (2021)
                     xrft_isospec_args = {"detrend":'constant', #"detrend":"linear", 
                                          "window":True, #"window":"hann", 
                                          #"nfactor":2,
                                           },
                     denorm_field = "SSH",
                    ):    
    r""" Calcualte isotropic power spectral density scores for generated fields

        Lots of hardcoding going on here unfortunately..
    
    """
    # Containers for all timesteps
    all_true_isospec = []
    all_mean_field_isospec = []
    all_sample_isospecs_mean = []
    all_sample_isospecs_std = []
    all_sample_WPSDs_mean = []
    all_sample_WPSDs_std = []
    all_sample_RMSEs = []

    if denorm_field == "SSH":
        denorm_func = denorm_ssh
    elif denorm_field == "SST":
        denorm_func = denorm_sst
    else: 
        denorm_func = denorm_null
        
    ################################################
    # Calculate the isospec for every step
    ################################################
    n_timesteps = test.time.size
    for t in range(n_timesteps):
        true_x_denormed = denorm_func(
            test.x_true.isel(time=t).chunk({}).compute(),
            test.time_i
        )
        mean_sample_x_denormed =  denorm_func(
            test.x_sample_members.isel(time=t).mean(dim="sample").chunk({}).compute() * STANDARDS["std_ssh"] + STANDARDS["mean_ssh"],
            test.time_i
        )
        
        ################################################
        # --- Compute Isospectra for truth field and all sample members ---
        ################################################
        if score_type == "xrft":
            true_isospec = xrft.isotropic_power_spectrum(
                true_x_denormed,
                **xrft_isospec_args
            ).compute()
            mean_field_isospec = xrft.isotropic_power_spectrum(
                mean_sample_x_denormed,
                **xrft_isospec_args
            ).compute()
        # Try replacing with more ad-hoc fft calculation. Use xrft-computed dataset (which includes correct wavenumber units) as a template
        else: 
            true_isospec = true_isospec*0 + npfft_isotropic_psd(
                true_x_denormed,
            )
            mean_field_isospec = mean_field_isospec*0 + npfft_isotropic_psd(
                mean_sample_x_denormed,
            )
        # Calculate the sample-wise isospecs
        sample_isospecs = []
        for i_member in range(len(test.sample)):
            tmp_sample = test.x_sample_members.isel(sample=i_member, time=t).chunk({}).compute()
            tmp_sample_denormed = denorm_func(tmp_sample,test.time_i)
            sample_isospecs.append(xrft.isotropic_power_spectrum(tmp_sample_denormed.compute(), **xrft_isospec_args).compute())
            # Try replacing with more ad-hoc fft calculation
            if score_type != "xrft": 
                sample_isospecs[i_member] = sample_isospecs[i_member]*0 + npfft_isotropic_psd(tmp_sample_denormed.compute())
        # Aggregate samplewise PSDs     
        sample_isospecs = xr.concat(sample_isospecs, dim="sample")
        sample_isospecs_samplewise_std = sample_isospecs.std(dim="sample")
        sample_isospecs_samplewise_mean = sample_isospecs.mean(dim="sample")

        ################################################
        # --- Compute WPSDs and other metrics ---
        ################################################
        sample_WPSDs = []
        sample_RMSEs = []
        for i_member in range(len(test.sample)):
            tmp_sample_minus_truth = (test.x_sample_members.isel(sample=i_member, time=t) - test.x_true.isel(time=t)).chunk({}).compute()
            tmp_sample_minus_truth_denormed = denorm_func(tmp_sample_minus_truth,test.time_i)
            tmp_isospec = xrft.isotropic_power_spectrum(tmp_sample_minus_truth_denormed, 
                                                        **xrft_isospec_args
                                                       ).compute()
            if score_type != "xrft": 
                tmp_isospec = tmp_isospec*0 + npfft_isotropic_psd(tmp_sample_minus_truth_denormed)
            sample_WPSDs.append(1 - tmp_isospec / true_isospec)
            sample_RMSEs.append(np.sqrt(np.sum(tmp_sample_minus_truth_denormed**2) / (128 * 128))/np.sqrt(np.sum(true_isospec**2)))
        # Aggregate samplewise metrics
        sample_WPSDs = xr.concat(sample_WPSDs, dim="sample")
        sample_WPSDs_samplewise_std = sample_WPSDs.std(dim="sample")
        sample_WPSDs_samplewise_mean = sample_WPSDs.mean(dim="sample")
        sample_RMSEs = np.asarray(sample_RMSEs)
        # --- Append per timestep results ---
        all_true_isospec.append(true_isospec)
        all_mean_field_isospec.append(mean_field_isospec)
        all_sample_isospecs_mean.append(sample_isospecs_samplewise_mean)
        all_sample_isospecs_std.append(sample_isospecs_samplewise_std)
        all_sample_WPSDs_mean.append(sample_WPSDs_samplewise_mean)
        all_sample_WPSDs_std.append(sample_WPSDs_samplewise_std)
        all_sample_RMSEs.append(sample_RMSEs)

    ################################################
    # --- OSSE Data Challenge metrics ---
    ################################################
    sample_osse_leaderboard_nrmses = []
    sample_osse_leaderboard_nrmse_stds = []    
    sample_osse_leaderboard_psds_scores = []
    sample_osse_leaderboard_psdt_scores = []

    sample_rmse_t_scores = []
    sample_psd_scores = []
    
    true_x_denormed = denorm_func(test.x_true.chunk({}).compute(),test.time_i)
    for i_member in range(len(test.sample)):   
        rmse_t_scores, rmse_xy_oi1, leaderboard_nrmse, leaderboard_nrmse_std = rmse_based_scores(denorm_func(test.x_sample_members.isel(sample=i_member),test.time_i),true_x_denormed)
        psd_scores, leaderboard_psds_score, leaderboard_psdt_score = psd_based_scores(denorm_func(test.x_sample_members.isel(sample=i_member),test.time_i),true_x_denormed)
        sample_osse_leaderboard_nrmses.append(leaderboard_nrmse)
        sample_osse_leaderboard_nrmse_stds.append(leaderboard_nrmse_std)
        sample_osse_leaderboard_psds_scores.append(leaderboard_psds_score)
        sample_osse_leaderboard_psdt_scores.append(leaderboard_psdt_score)
        sample_rmse_t_scores.append(rmse_t_scores)
        sample_psd_scores.append(psd_scores)
        
    ################################################
    # --- Save everything in a unified dataset ---
    ################################################
    all_true_isospec = xr.concat(all_true_isospec, dim="time")
    all_mean_field_isospec = xr.concat(all_mean_field_isospec, dim="time")
    all_sample_isospecs_mean = xr.concat(all_sample_isospecs_mean, dim="time")
    all_sample_isospecs_std = xr.concat(all_sample_isospecs_std, dim="time")
    all_sample_WPSDs_mean = xr.concat(all_sample_WPSDs_mean, dim="time")
    all_sample_WPSDs_std = xr.concat(all_sample_WPSDs_std, dim="time")
    all_sample_RMSEs = np.stack(all_sample_RMSEs, axis=0)
    
    sample_osse_leaderboard_nrmses = np.asarray(sample_osse_leaderboard_nrmses)
    sample_osse_leaderboard_nrmse_stds = np.asarray(sample_osse_leaderboard_nrmse_stds)
    sample_osse_leaderboard_psds_scores = np.asarray(sample_osse_leaderboard_psds_scores)
    sample_osse_leaderboard_psdt_scores = np.asarray(sample_osse_leaderboard_psdt_scores)
    sample_rmse_t_scores = xr.concat(sample_rmse_t_scores,dim="sample")
    sample_psd_scores = xr.concat(sample_psd_scores,dim="sample")
    
    # Store in a single dataset
    result = xr.Dataset({
        "true_isospec": all_true_isospec,
        "mean_field_isospec": all_mean_field_isospec,
        "sample_isospecs_mean": all_sample_isospecs_mean,
        "sample_isospecs_std": all_sample_isospecs_std,
        "WPSD_sample": sample_WPSDs.drop_vars("time_i"),
        "WPSD_mean": all_sample_WPSDs_mean,
        "WPSD_std": all_sample_WPSDs_std,
        "RMSEs": (["time", "sample"], all_sample_RMSEs),

        "leaderboard_nrmses":(["sample"],sample_osse_leaderboard_nrmses),
        "leaderboard_nrmse_std":(["sample"],sample_osse_leaderboard_nrmse_stds),
        "leaderboard_psds_score":(["sample"],sample_osse_leaderboard_psds_scores),
        "leaderboard_psdt_score":(["sample"],sample_osse_leaderboard_psdt_scores),
        "sample_rmse_t_score":sample_rmse_t_scores,
        "sample_psd_score":sample_psd_scores,
    })
    
    ################################################
    # --- Compute cutoff_length per timestep ---
    ################################################
    result = get_cutoff_freq(result)    
    return result


