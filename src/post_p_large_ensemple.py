import xarray as xr
import numpy as np
from src.LE_LoadAndMerge import MultiModelLargeEnsemble
from statsmodels.distributions.empirical_distribution import ECDF
from statsmodels.distributions.empirical_distribution import StepFunction
from app.main.src.climate_projection import ClimateProjection

era2cmip = {
    't2m': 'tas',
    't2m_max': 'tasmax',
    't2m_min': 'tasmin',
    'tp': 'pr',
    'huss': 'huss',
    # 'd2m': 'tdps',
    'ws': 'sfcWind',
    'sm': 'mrsos'
}
cmip2era = {v: k for k, v in era2cmip.items()}

regions = {
    'USeast': {'lat': 41.3, 'lon': -72.5},
    'USwest':  {'lat': 37.7, 'lon': -122.4, },
    'iceland':  {'lat': 65, 'lon': -19},
    'europe':  {'lat': 51, 'lon': 10.5},
    'australia': {'lat': -25.2, 'lon': 133.7},
    'tropics': {'lat': 3.9, 'lon': -53.1}
}


class MyECDF(StepFunction):
    def __init__(self, x, side='right'):
        x = np.sort(x)

        # count number of non-nan's instead of length
        nobs = np.count_nonzero(~np.isnan(x))

        # fill the y values corresponding to np.nan with np.nan
        y = np.full_like(x, np.nan)
        y[:nobs] = np.linspace(1./nobs, 1, nobs)
        super(MyECDF, self).__init__(x, y, side=side, sorted=True)


def prepare_data(data: xr.DataArray) -> xr.DataArray:
    threshold = 0.1
    prep_data = data.where(data > threshold,
                           np.random.uniform(0, threshold, len(data)))
    return prep_data.where(~data.isnull())


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

    hist = hist.values.flatten()
    X_clean = X.values.flatten()

    quantiles = MyECDF(X_clean)(X_clean)

    historical_value = np.nanquantile(hist, np.nan_to_num(quantiles))
    reanalysis_value = np.nanquantile(obs, np.nan_to_num(quantiles))

    X_clean_post_p = reanalysis_value + X_clean - historical_value
    X_clean_post_p = X_clean_post_p.reshape(X.shape, order='C')

    data = xr.DataArray(X_clean_post_p, dims=X.dims, coords=X.coords)
    return data.where(~X.isnull())


variable = 'tasmax'
models = ['CanESM5', 'cesm_lens', 'MIROC6', 'MPI-ESM1-2-LR', 'EC-Earth3']

for region in ['USwest', 'europe', 'australia', 'tropics', 'USeast', 'iceland']:
    lat = regions[region]['lat']
    lon = regions[region]['lon']
    print(f'Processing {region} future for {variable}')

    MMLE = MultiModelLargeEnsemble(models=models,
                                   variable=variable, granularity='day',
                                   lat=lat, lon=lon,
                                   bucket='climateai_data_repository',
                                   path='tmp/internal_variability',
                                   load=True)

    path = f'gcs://climateai_data_repository/tmp/internal_variability/era_files/{region}/reanalysis_daily.zarr'
    reanalysis_daily = xr.open_zarr(path, consolidated=True).load()

    # print(f'Processing {region} historical for {variable}')
    # large_ens_hist = qdm_large_ensemble(
    #     MMLE.hist[variable],
    #     MMLE.hist[variable],
    #     reanalysis_daily[cmip2era[variable]]
    # ).to_dataset(name=variable)

    large_ens_future = qdm_large_ensemble(
        MMLE.future[variable],
        MMLE.hist[variable],
        reanalysis_daily[cmip2era[variable]]
    ).to_dataset(name=variable)

    cp = ClimateProjection(
        lat, lon, cmip2era[variable], 'ssp585', projection_name=region,
        gcs_bucket='climateai_data_repository', gcs_path='tmp/internal_variability/era_files'
    )
    cp._save_ds(large_ens_future, f'future_qdm_{variable}', chunks=None)
    # cp._save_ds(large_ens_hist, f'hist_qdm_{variable}', chunks=None)
    print(f'Done post-processing {region}!')
