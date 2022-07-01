import xarray as xr
from LE_LoadAndMerge import MultiModelLargeEnsemble
from post_p_large_ensemble import qdm_large_ensemble

from app.main.src.climate_projection import ClimateProjection
from app.main.src.utils import cmip2era

regions = {
    'USeast': {'lat': 41.3, 'lon': -72.5},
    'USwest':  {'lat': 37.7, 'lon': -122.4, },
    'iceland':  {'lat': 65, 'lon': -19},
    'europe':  {'lat': 51, 'lon': 10.5},
    'australia': {'lat': -25.2, 'lon': 133.7},
    'tropics': {'lat': 3.9, 'lon': -53.1}
}

variable = 'tasmax'

models_for_vars = {
    'tas': ['CanESM5','cesm_lens','MIROC6','MPI-ESM1-2-LR','EC-Earth3'],
    'pr' : ['CanESM5','cesm_lens','MIROC6','MPI-ESM1-2-LR'],
    'tasmax': ['CanESM5','cesm_lens','MIROC6','MPI-ESM1-2-LR','EC-Earth3'],
}
for region in ['USwest', 'europe', 'australia', 'tropics', 'USeast', 'iceland']:
    lat = regions[region]['lat']
    lon = regions[region]['lon']
    print(f'Processing {region} future for {variable}')

    MMLE = MultiModelLargeEnsemble(models=models_for_vars[variable],
                                   variable=variable, granularity='day',
                                   lat=lat, lon=lon,
                                   bucket='climateai_data_repository',
                                   path='tmp/internal_variability',
                                   load=True)

    path = f'gcs://climateai_data_repository/tmp/internal_variability/era_files/{region}/reanalysis_daily.zarr'
    reanalysis_daily = xr.open_zarr(path, consolidated=True).load()

    large_ens_hist = qdm_large_ensemble(
        MMLE.hist[variable],
        MMLE.hist[variable],
        reanalysis_daily[cmip2era[variable]]
    ).to_dataset(name=variable)

    # large_ens_future = qdm_large_ensemble(
    #     MMLE.future[variable],
    #     MMLE.hist[variable],
    #     reanalysis_daily[cmip2era[variable]]
    # ).to_dataset(name=variable)

    cp = ClimateProjection(
        lat, lon, cmip2era[variable], 'ssp585', projection_name=region,
        gcs_bucket='climateai_data_repository', gcs_path='tmp/internal_variability/era_files'
    )
    #cp._save_ds(large_ens_future, f'future_qdm_{variable}', chunks=None)
    cp._save_ds(large_ens_hist, f'hist_qdm_{variable}', chunks=None)
    print(f'Done post-processing {region}!')
