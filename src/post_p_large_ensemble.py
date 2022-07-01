import xarray as xr
import numpy as np
from statsmodels.distributions.empirical_distribution import StepFunction

class MyECDF(StepFunction):
    def __init__(self, x, side='right'):
        x = np.sort(x)

        # count number of non-nan's instead of length
        nobs = np.count_nonzero(~np.isnan(x))

        # fill the y values corresponding to np.nan with np.nan
        y = np.full_like(x, np.nan)
        y[:nobs] = np.linspace(1./nobs, 1, nobs)
        super(MyECDF, self).__init__(x, y, side=side, sorted=True)


def prepare_data(data: xr.DataArray, threshold: float = 0.1) -> xr.DataArray:
    prep_data = data.where(data > threshold,
                           np.random.uniform(0, threshold, len(data)))
    return prep_data.where(~data.isnull())

def prepare_data_numpy(data: xr.DataArray, threshold: float = 0.1) -> xr.DataArray:
    random_array = np.random.uniform(0, threshold, len(data))
    prep_data = np.where(data > threshold, data, random_array)
    return np.where(np.isfinite(data), prep_data, np.nan)

def qdm_large_ensemble(hist, future, reanalysis):
    """QDM for daily data"""
    if 'model' in hist.dims:
        assert sorted(hist.model) == sorted(future.model)

        objects = []
        for model in list(hist.model.values):
            pp_dataset = qdm_large_ensemble(
                hist.sel(model=model),
                future.sel(model=model),
                reanalysis
            )
            objects.append(pp_dataset)
            print(f'{model} has been post-processed')
        return xr.concat(objects, dim=hist.model)

    else:
        return implement_quantile_delta_mapping(hist, future, reanalysis)


def implement_quantile_delta_mapping(
    X: xr.DataArray, hist: xr.DataArray, obs: xr.DataArray
) -> xr.DataArray:
    """Create the post processed dataset for the whole original time series.
    Using previously stored self.his and self.obs data, created from reanalysis
    and historical datasets. Implementation allows for both multiplicative and
    additive method. Original NaNs are kept. If necessary, data is prepared and
    returned to original values below threshold to maintain dry days.

    Parameters
    ----------
    X: testing dataset

    Returns
    -------
    y_hat: xr.Dataset or xr.DataArray
        post-processed data
    """
    obs_r = obs.where(obs.time==hist.time)
    hist_r = hist.where(obs.time==hist.time)

    hist_flat = hist_r.values.flatten()
    X_flat = X.values.flatten()

    if obs.name == 'tp':
        obs_r =  prepare_data(obs_r)
        hist_r = prepare_data_numpy(hist_r)
        X = prepare_data_numpy(X)

    quantiles = MyECDF(X_flat)(X_flat)

    historical_value = np.nanquantile(hist_flat, np.nan_to_num(quantiles))
    reanalysis_value = np.nanquantile(obs_r, np.nan_to_num(quantiles))

    if obs.name == 'tp':
        X_clean_post_p = (reanalysis_value * X_flat) /historical_value
    else:
        X_clean_post_p = reanalysis_value + X_flat - historical_value

    X_clean_post_p = X_clean_post_p.reshape(X.shape, order='C')

    data = xr.DataArray(X_clean_post_p, dims=X.dims, coords=X.coords)

    if obs.name == 'tp':
        data = data.where(X > 0.1, 0)

    return data.where(~X.isnull())
