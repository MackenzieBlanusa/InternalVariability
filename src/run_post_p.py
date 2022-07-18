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

# regions_dict = {
#     'USeast': {'lat': 41.3, 'lon': -72.5},
#     'USwest':  {'lat': 37.7, 'lon': -122.4, },
#     'iceland':  {'lat': 65, 'lon': -19},
#     'europe':  {'lat': 51, 'lon': 10.5},
#     'australia': {'lat': -25.2, 'lon': 133.7},
#     'tropics': {'lat': 3.9, 'lon': 306.9}
# }

regions_dict = {
    'Seattle': {'lat': 47.6, 'lon': 237.7},
    'Sydney':  {'lat': -33.8, 'lon': 151.2},
    'Lagos':  {'lat': 6.5, 'lon': 3.4},
}

variable = 'pr'

models = ['MIROC6', 'CanESM5', 'MPI-ESM1-2-LR', 'EC-Earth3','cesm_lens']

# ,['USwest', 'europe', 'australia', 'tropics', 'USeast', 'iceland']:
for region, lat_lon in regions_dict.items():
    
    lat = lat_lon['lat']
    lon = lat_lon['lon']
    print(f'Processing {region} future for {variable}')

    MMLE = MultiModelLargeEnsemble(models=models, variable=variable, granularity='day',
                               lat=lat, lon=lon,
                               bucket='climateai_data_repository', path='tmp/global_cmip_2.5deg')

    path = f'gcs://climateai_data_repository/tmp/internal_variability/era_files/{region}/reanalysis_daily.zarr'
    reanalysis_daily = xr.open_zarr(path, consolidated=True).load()

    if variable in ('tasmax', 'tas'):
        reanalysis_daily[cmip2era[variable]] += 273.15
        print('Temperature era units converted')
    elif variable in ('pr'):
        reanalysis_daily[cmip2era[variable]] *= (997/(1000*24*60*60))
        print('Precipitation era units converted')

    print('Processing historical models')
    large_ens_hist = qdm_large_ensemble(
        MMLE.hist_dsets,
        MMLE.hist_dsets,
        reanalysis_daily[cmip2era[variable]],
        monthly_w = 3
    )

    print('Processing future models')
    large_ens_future = qdm_large_ensemble(
        MMLE.future_dsets,
        MMLE.hist_dsets,
        reanalysis_daily[cmip2era[variable]],
        monthly_w = 3
    )

    print(f'Done post-processing {region}!')
    print(f'Uploading {region} files to GCS')

    for model in large_ens_future.keys():
        future_path = f'gcs://{MMLE.bucket}/tmp/qdm_{region}/{model}/{MMLE.scenario}/{MMLE.granularity}/{MMLE.variable}.zarr'
        large_ens_future[model].to_dataset(name=variable).to_zarr(future_path, consolidated=True, mode='w')

        hist_path = f'gcs://{MMLE.bucket}/tmp/qdm_{region}/{model}/historical/{MMLE.granularity}/{MMLE.variable}.zarr'
        large_ens_hist[model].to_dataset(name=variable).to_zarr(hist_path, consolidated=True, mode='w')

    print(f'Done uploading {region}!')
