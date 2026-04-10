"""
Standalone seasonal SST normalization module.

This module provides functionality to normalize sea surface temperature (SST) data
using daily climatological means, removing seasonal variations.
"""

import numpy as np
import xarray as xr
import os


class SSHNormalizer:
    """
    Normalize SSH data given a mean and standard deviation.
    
    Parameters
    ----------
    std : float, default=5.0
        Standard deviation to use for normalization.
    mean : float, default=0.0
        Additional mean adjustment to apply during normalization.
    """

    def __init__(self, std=1, mean=0):
        self.std = std
        self.mean = mean

    def normalize(self, ssh_data, time_indices=None):
        return (ssh_data-self.mean)/self.std
        
    def denormalize(self, ssh_data, time_indices=None):
        return ssh_data*self.std + self.mean


        
class SeasonalSSTNormalizer:
    """
    Normalize SST data by removing daily climatological means.
    
    Parameters
    ----------
    climatology_path : str, optional
        Path to the SST climatology netCDF file. If None, will attempt to 
        auto-detect based on common project paths.
    std : float, default=5.0
        Standard deviation to use for normalization.
    extra_mean_tuning : float, default=0.0
        Additional mean adjustment to apply during normalization.
    """
    
    def __init__(self, climatology_path=None, std=5.0, extra_mean_tuning=0.0):
        self.std = std
        self.extra_mean_tuning = extra_mean_tuning
        
        # Auto-detect climatology path if not provided
        if climatology_path is None:
            climatology_path = self._find_climatology_path()
        
        # Load SST climatology
        self.SST_mean_climatology = xr.open_dataset(climatology_path)
        self.climatology_length = len(self.SST_mean_climatology.SST)
    
    def _find_climatology_path(self):
        """Attempt to auto-detect the climatology file path."""
        possible_paths = [
            '/home/tm3076/projects/NYU_SWOT_project/Inpainting_Pytorch_gen/SWOT-inpainting-DL/data/SST_NP_daily_climatology.nc',
            '/usr/home/tm3076/swot_SUM03/SWOT_project/SWOT-inpainting-DL/data/SST_NP_daily_climatology.nc',
            '/scratch/tm3076/project/SWOT-inpainting-DL/data/SST_NP_daily_climatology.nc',
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        raise FileNotFoundError(
            "Could not auto-detect climatology file. Please provide climatology_path explicitly."
        )
    
    def _match_climatology_by_dayofyear(self, data_times):
        """
        Match data times to climatology by day-of-year (ignoring year).
        
        Parameters
        ----------
        data_times : np.ndarray of datetime64
            Time coordinates from the data to normalize.
        
        Returns
        -------
        np.ndarray
            Indices into the climatology that match the day-of-year.
        """
        # Convert data times to day-of-year
        data_times_dt = data_times.astype('datetime64[D]')
        data_doy = (data_times_dt - data_times_dt.astype('datetime64[Y]')).astype(int) + 1
        
        # Convert climatology times to day-of-year
        clim_times = self.SST_mean_climatology.time.values
        clim_times_dt = clim_times.astype('datetime64[D]')
        clim_doy = (clim_times_dt - clim_times_dt.astype('datetime64[Y]')).astype(int) + 1
        
        # Match each data day-of-year to closest climatology day-of-year
        matched_indices = []
        for doy in data_doy:
            # Find the climatology index with matching day-of-year
            # Handle wraparound for leap years by finding closest match
            idx = np.argmin(np.abs(clim_doy - doy))
            matched_indices.append(idx)
        
        return np.array(matched_indices)
    
    def normalize_from_datetime(self, sst_data, data_times=None):
        """
        Normalize SST data using datetime coordinates to match climatology by day-of-year.
        
        This method is useful when normalizing data from different years using the same
        climatology, as it matches based on the day-of-year (month and day) rather than
        the absolute date.
        
        Parameters
        ----------
        sst_data : np.ndarray or xr.DataArray
            SST data to normalize. Can be:
            - 3D array: (time, y, x)
            - 2D array: (y, x) for single time step
            If xr.DataArray with time coordinate, data_times is extracted automatically.
        data_times : np.ndarray of datetime64, optional
            Time coordinates corresponding to the data. Required if sst_data is np.ndarray.
            Should be datetime64[ns] array matching the time dimension of sst_data.
        
        Returns
        -------
        np.ndarray
            Normalized SST data with same shape as input.
        
        Examples
        --------
        >>> # With xarray DataArray (time coordinate auto-detected)
        >>> normalized = normalizer.normalize_from_datetime(sst_dataarray)
        
        >>> # With numpy array (must provide times explicitly)
        >>> times = np.array(['2024-11-20', '2024-11-21'], dtype='datetime64[ns]')
        >>> normalized = normalizer.normalize_from_datetime(sst_array, data_times=times)
        """
        # Extract data and times from xarray if needed
        if isinstance(sst_data, xr.DataArray):
            if data_times is None and 'time' in sst_data.coords:
                data_times = sst_data.time.values
            arr = sst_data.values
        else:
            arr = sst_data
        
        if data_times is None:
            raise ValueError(
                "data_times must be provided when sst_data is a numpy array. "
                "Provide datetime64 array or use xr.DataArray with time coordinate."
            )
        
        # Ensure data_times is array
        data_times = np.atleast_1d(data_times)
        
        # Match climatology by day-of-year
        clim_indices = self._match_climatology_by_dayofyear(data_times)
        
        # Get climatological means for matched days
        clim_means = self.SST_mean_climatology.SST[:365].isel(time=clim_indices).values
        
        # Normalize based on array dimensionality
        if len(arr.shape) == 3:  # (time, y, x)
            clim_means_broadcast = clim_means[:, np.newaxis, np.newaxis]
            normalized_arr = (arr - clim_means_broadcast - self.extra_mean_tuning) / self.std
        elif len(arr.shape) == 2:  # (y, x) - single time step
            clim_mean = clim_means[0]
            normalized_arr = (arr - clim_mean - self.extra_mean_tuning) / self.std
        else:
            clim_mean = clim_means.mean()
            normalized_arr = (arr - clim_mean - self.extra_mean_tuning) / self.std
        
        return normalized_arr.astype(np.float32)
    
    def denormalize_from_datetime(self, normalized_data, data_times=None):
        """
        Reverse the normalization using datetime coordinates.
        
        Parameters
        ----------
        normalized_data : np.ndarray or xr.DataArray
            Normalized SST data to denormalize.
        data_times : np.ndarray of datetime64, optional
            Time coordinates corresponding to the data. Required if normalized_data is np.ndarray.
        
        Returns
        -------
        np.ndarray
            Denormalized SST data.
        """
        # Extract data and times from xarray if needed
        if isinstance(normalized_data, xr.DataArray):
            if data_times is None and 'time' in normalized_data.coords:
                data_times = normalized_data.time.values
            arr = normalized_data.values
        else:
            arr = normalized_data
        
        if data_times is None:
            raise ValueError(
                "data_times must be provided when normalized_data is a numpy array. "
                "Provide datetime64 array or use xr.DataArray with time coordinate."
            )
        
        # Ensure data_times is array
        data_times = np.atleast_1d(data_times)
        
        # Match climatology by day-of-year
        clim_indices = self._match_climatology_by_dayofyear(data_times)
        
        # Get climatological means
        clim_means = self.SST_mean_climatology.SST[:365].isel(time=clim_indices).values
        
        # Denormalize based on array dimensionality
        if len(arr.shape) == 3:  # (time, y, x)
            clim_means_broadcast = clim_means[:, np.newaxis, np.newaxis]
            denormalized = (arr * self.std) + clim_means_broadcast + self.extra_mean_tuning
        elif len(arr.shape) == 2:  # (y, x)
            clim_mean = clim_means[0]
            denormalized = (arr * self.std) + clim_mean + self.extra_mean_tuning
        else:
            clim_mean = clim_means.mean()
            denormalized = (arr * self.std) + clim_mean + self.extra_mean_tuning
        
        return denormalized.astype(np.float32)
    
    def normalize(self, sst_data, time_indices, dtime=12):
        """
        Normalize SST data using daily climatological means.
        
        Parameters
        ----------
        sst_data : np.ndarray or xr.DataArray
            SST data to normalize. Can be:
            - 3D array: (time, y, x)
            - 2D array: (y, x) for single time step
            Shape should match the time_indices provided.
        time_indices : np.ndarray or list
            Time step indices corresponding to the data. Should be:
            - Array of indices for multi-timestep data
            - Single index or list with one element for single timestep
        dtime : int, default=1
            Time discretization factor. If dtime > 1, converts from hourly 
            to daily timesteps by dividing by 24.
        
        Returns
        -------
        np.ndarray
            Normalized SST data with same shape as input.
        """
        # Convert xarray to numpy if needed
        if isinstance(sst_data, xr.DataArray):
            arr = sst_data.values
        else:
            arr = sst_data
        # Ensure time_indices is array
        time_indices = np.atleast_1d(time_indices)
        # Convert from hourly to daily timesteps if needed
        if dtime > 1:
            time_indices = time_indices // 24
        # Clip indices to valid range
        clipped_indices = np.clip(time_indices, 0, self.climatology_length - 1)
        # Get climatological means for this time window
        clim_means = self.SST_mean_climatology.SST.isel(time=clipped_indices).values
        # Normalize based on array dimensionality
        if len(arr.shape) == 3:  # (time, y, x)
            # Broadcast climatological means to match spatial dimensions
            clim_means_broadcast = clim_means[:, np.newaxis, np.newaxis]
            normalized_arr = (arr - clim_means_broadcast - self.extra_mean_tuning) / self.std
        elif len(arr.shape) == 2:  # (y, x) - single time step
            # Use single climatological value
            clim_mean = clim_means[len(clim_means)//2] if len(clim_means) > 1 else clim_means[0]
            normalized_arr = (arr - clim_mean - self.extra_mean_tuning) / self.std
        else:
            # Fallback: use mean of climatological values
            clim_mean = clim_means.mean()
            normalized_arr = (arr - clim_mean - self.extra_mean_tuning) / self.std
        return normalized_arr.astype(np.float32)
    
    def denormalize(self, normalized_data, time_indices, dtime=1):
        """
        Reverse the normalization to recover original SST values.
        
        Parameters
        ----------
        normalized_data : np.ndarray
            Normalized SST data to denormalize.
        time_indices : np.ndarray or list
            Time step indices corresponding to the data.
        dtime : int, default=1
            Time discretization factor.
        
        Returns
        -------
        np.ndarray
            Denormalized SST data.
        """
        # Ensure time_indices is array
        time_indices = np.atleast_1d(time_indices)
        # Convert from hourly to daily timesteps if needed
        if dtime > 1:
            time_indices = time_indices // 24
        # Clip indices to valid range
        clipped_indices = np.clip(time_indices, 0, self.climatology_length - 1)
        # Get climatological means
        clim_means = self.SST_mean_climatology.SST.isel(time=clipped_indices).values
        # Denormalize based on array dimensionality
        if len(normalized_data.shape) == 3:  # (time, y, x)
            clim_means_broadcast = clim_means[:, np.newaxis, np.newaxis]
            denormalized = (normalized_data * self.std) + clim_means_broadcast + self.extra_mean_tuning
        elif len(normalized_data.shape) == 2:  # (y, x)
            clim_mean = clim_means[len(clim_means)//2] if len(clim_means) > 1 else clim_means[0]
            denormalized = (normalized_data * self.std) + clim_mean + self.extra_mean_tuning
        else:
            clim_mean = clim_means.mean()
            denormalized = (normalized_data * self.std) + clim_mean + self.extra_mean_tuning
        return denormalized.astype(np.float32)



    
# Convenience function for quick usage
def normalize_sst_seasonal(sst_data, time_start, time_end, dtime=1, 
                          climatology_path=None, std=5.0, extra_mean_tuning=0.0):
    """
    Convenience function to normalize SST data in one call.
    
    Parameters
    ----------
    sst_data : np.ndarray or xr.DataArray
        SST data to normalize.
    time_start : int
        Starting time index.
    time_end : int
        Ending time index (exclusive).
    dtime : int, default=1
        Time step increment.
    climatology_path : str, optional
        Path to climatology file.
    std : float, default=5.0
        Standard deviation for normalization.
    extra_mean_tuning : float, default=0.0
        Additional mean adjustment.
    
    Returns
    -------
    np.ndarray
        Normalized SST data.
    
    Examples
    --------
    >>> # Normalize a 3D SST tile
    >>> sst_tile = np.random.randn(10, 128, 128)  # 10 timesteps
    >>> time_indices = np.arange(100, 110)  # corresponding time indices
    >>> normalized = normalize_sst_seasonal(sst_tile, 100, 110)
    
    >>> # Normalize a single timestep
    >>> sst_snapshot = np.random.randn(128, 128)
    >>> normalized = normalize_sst_seasonal(sst_snapshot, 50, 51)
    """
    normalizer = SeasonalSSTNormalizer(
        climatology_path=climatology_path,
        std=std,
        extra_mean_tuning=extra_mean_tuning
    )
    time_indices = np.arange(time_start, time_end, dtime)
    return normalizer.normalize(sst_data, time_indices, dtime)


if __name__ == "__main__":
    # Example usage
    print("Seasonal SST Normalizer - Example Usage\n")
    
    # Example 1: Normalize a 3D tile
    print("Example 1: 3D SST tile (10 timesteps)")
    sst_3d = np.random.randn(10, 128, 128) + 20  # Random SST data around 20°C
    time_start, time_end = 100, 110
    
    # Create normalizer instance for reuse
    normalizer = SeasonalSSTNormalizer(std=5.0)
    time_indices = np.arange(time_start, time_end)
    normalized_3d = normalizer.normalize(sst_3d, time_indices)
    
    print(f"  Input shape: {sst_3d.shape}")
    print(f"  Input mean: {sst_3d.mean():.2f}, std: {sst_3d.std():.2f}")
    print(f"  Normalized mean: {normalized_3d.mean():.2f}, std: {normalized_3d.std():.2f}")
    
    # Verify denormalization
    denormalized = normalizer.denormalize(normalized_3d, time_indices)
    print(f"  Denormalized matches original: {np.allclose(sst_3d, denormalized)}\n")
    
    # Example 2: Normalize a single timestep
    print("Example 2: Single SST snapshot")
    sst_2d = np.random.randn(128, 128) + 18
    normalized_2d = normalizer.normalize(sst_2d, [50])
    print(f"  Input shape: {sst_2d.shape}")
    print(f"  Normalized shape: {normalized_2d.shape}\n")
    
    # Example 3: Using convenience function
    print("Example 3: Using convenience function")
    normalized_quick = normalize_sst_seasonal(sst_3d, time_start, time_end, std=5.0)
    print(f"  Quick normalized shape: {normalized_quick.shape}")