import xarray as xr
from multi_model_large_ensemble import MultiModelLargeEnsemble
from post_p_large_ensemble import qdm_large_ensemble

era2cmip = {
    't2m': 'tas',
    't2m_max': 'tasmax',
    't2m_min': 'tasmin',
    'tp': 'pr',
    'huss': 'huss',  # Specific humidity does not exist in ERA, so just mapping CMIP to CMIP
    # 'd2m': 'tdps',
    'ws': 'sfcWind',
    'sm': 'mrsos'  # https://agupubs.onlinelibrary.wiley.com/doi/epdf/10.1029/2020GL089991
}

cmip2era = {v: k for k, v in era2cmip.items()}

regions_dict = {
    'USeast': {'lat': 41.3, 'lon': -72.5},
    'USwest':  {'lat': 37.7, 'lon': -122.4, },
    'iceland':  {'lat': 65, 'lon': -19},
    'europe':  {'lat': 51, 'lon': 10.5},
    'australia': {'lat': -25.2, 'lon': 133.7},
    'tropics': {'lat': 3.9, 'lon': -53.1}
}

variable = 'tasmax'

models_for_vars = {
    'tas': ['CanESM5', 'cesm_lens', 'MIROC6', 'MPI-ESM1-2-LR', 'EC-Earth3'],
    'pr': ['CanESM5', 'cesm_lens', 'MIROC6', 'MPI-ESM1-2-LR'],
    'tasmax': ['CanESM5', 'cesm_lens', 'MIROC6', 'MPI-ESM1-2-LR', 'EC-Earth3'],
}
# ,['USwest', 'europe', 'australia', 'tropics', 'USeast', 'iceland']:
for region in ['USwest', 'europe', 'tropics', 'USeast', 'iceland']:
    lat = regions_dict[region]['lat']
    lon = regions_dict[region]['lon']
    print(f'Processing {region} future for {variable}')

    MMLE = MultiModelLargeEnsemble(models=models_for_vars[variable],
                                   variable=variable, granularity='day',
                                   lat=lat, lon=lon,
                                   bucket='climateai_data_repository',
                                   path='tmp/global_cmip_2.5deg')

    path = f'gcs://climateai_data_repository/tmp/internal_variability/era_files/{region}/reanalysis_daily.zarr'
    reanalysis_daily = xr.open_zarr(path, consolidated=True).load()

    if variable in ('tasmax', 'tas'):
        reanalysis_daily[cmip2era[variable]] += 273.15

    print('Processing historical models')
    large_ens_hist = qdm_large_ensemble(
        MMLE.hist_dsets,
        MMLE.hist_dsets,
        reanalysis_daily[cmip2era[variable]]
    )

    print('Processing future models')
    large_ens_future = qdm_large_ensemble(
        MMLE.future_dsets,
        MMLE.hist_dsets,
        reanalysis_daily[cmip2era[variable]]
    )

    print(f'Done post-processing {region}!')
    print(f'Uploading {region} files to GCS')

    for model in large_ens_future.keys():
        future_path = f'gcs://{MMLE.bucket}/tmp/qdm_{region}/{model}/{MMLE.scenario}/{MMLE.granularity}/{MMLE.variable}.zarr'
        large_ens_future[model].to_dataset(name=variable).to_zarr(future_path, consolidated=True, mode='w')

        hist_path = f'gcs://{MMLE.bucket}/tmp/qdm_{region}/{model}/historical/{MMLE.granularity}/{MMLE.variable}.zarr'
        large_ens_hist[model].to_dataset(name=variable).to_zarr(hist_path, consolidated=True, mode='w')

    print(f'Done uploading {region}!')
