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
    if isinstance(hist, dict):
        assert sorted(hist.keys()) == sorted(future.keys())

        qdm_le = {}
        for model in sorted(hist.keys()):
            pp_dataset = qdm_large_ensemble(
                hist[model].load(),
                future[model].load(),
                reanalysis
            )
            qdm_le[model] = pp_dataset
            print(f'{model} has been post-processed')
        return qdm_le

    else:
        return implement_quantile_delta_mapping(hist, future, reanalysis)


def create_rolling_window(X, time_window=10):
    # define time deltas for days (24 h), months (28, 29, 30, and 31 days ),
    # and year (regular and leap year)
    recurrence = 'D'

    years = X.time.dt.year.values
    start_date = np.datetime64(X.time.values[0], recurrence)
    end_date = np.datetime64(X.time.values[-1], recurrence)

    # create data to append in the beginning to ensure static time window
    X_1_slice = X.sel(time=slice(str(years[0] + int(time_window / 2)),
                                 str(years[0] + time_window - 1)))
    X1_dates = np.arange(start_date - np.timedelta64(len(X_1_slice.time), recurrence),
                         start_date,
                         dtype=f'datetime64[{recurrence}]')
    X_1 = X_1_slice.assign_coords(
        {'member_id': X.member_id.values, 'time': X1_dates})

    # create data to append in the end
    X_2_slice = X.sel(time=slice(str(years[-1] - time_window + 1),
                                 str(years[-1] - int(time_window / 2))))
    X2_dates = np.arange(end_date + np.timedelta64(1, recurrence),
                         end_date +
                         np.timedelta64(len(X_2_slice.time) + 1, recurrence),
                         dtype=f'datetime64[{recurrence}]')
    X_2 = X_2_slice.assign_coords(time=X2_dates, member_id=X.member_id.values)

    # append data, construct rolling and select original years
    X_final = xr.concat([X_1, X, X_2], dim='time')
    X_rolling = X_final.rolling(time=len(X1_dates) * 2,
                                center=True).construct('tm').sel(
        time=slice(X.time[0].values, X.time[-1].values))

    # select desired time first element of the year, every 'step' years
    X_rolling = X_rolling.sel(
        time=X_rolling.time.dt.day == X_rolling.time.dt.day.values[0])

    X_rolling = X_rolling.sel(
        time=X_rolling.time.dt.month == X_rolling.time.dt.month.values[0])

    return X_rolling


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
    obs_r = obs.where(obs.time == hist.time)
    hist_r = hist.where(obs.time == hist.time)

    hist_flat = hist_r.values.flatten()

    if obs.name == 'tp':
        obs_r = prepare_data(obs_r)
        hist_r = prepare_data_numpy(hist_r)
        X = prepare_data_numpy(X)

    years = []
    X_rolling = create_rolling_window(X)
    for i in X_rolling.time:
        X_flat = X_rolling.sel(time=i).values.flatten()
        X_cut = X.sel(time=str(i.dt.year.values))
        quantiles = MyECDF(X_flat)(X_cut.values.flatten())

        historical_value = np.nanquantile(hist_flat, np.nan_to_num(quantiles))
        reanalysis_value = np.nanquantile(obs_r, np.nan_to_num(quantiles))

        if obs.name == 'tp':
            X_clean_post_p = (reanalysis_value *
                              X_cut.values.flatten()) / historical_value
        else:
            X_clean_post_p = reanalysis_value + X_cut.values.flatten() - historical_value

        X_clean_post_p = X_clean_post_p.reshape(X_cut.shape, order='C')

        data = xr.DataArray(X_clean_post_p, dims=X_cut.dims,
                            coords=X_cut.coords)
        years.append(data)

    complete_data = xr.concat(years, dim='time')
    if obs.name == 'tp':
        complete_data = complete_data.where(X > 0.1, 0)

    return complete_data.where(~X.isnull())
