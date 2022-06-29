from app.main.src.climate_projection import ClimateProjection
from app.main.src.datasets import ERA

regions = {
    'USeast': {'lat': 41.3, 'lon': -72.5},
    'USwest':  {'lat': 37.7, 'lon': -122.4, },
    'iceland':  {'lat': 65, 'lon': -19},
    'europe':  {'lat': 51, 'lon': 10.5},
    'australia': {'lat': -25.2, 'lon': 133.7},
    'tropics': {'lat': 3.9, 'lon': -53.1}
}


for region in ['iceland', 'europe', 'australia', 'tropics']:

    lat = regions[region]['lat']
    lon = regions[region]['lon']


    variables = ['t2m', 'tp', 't2m_max']

    cp = ClimateProjection(
        lat,lon,variables,'ssp585',projection_name=region,
        gcs_bucket='climateai_data_repository', gcs_path='tmp/internal_variability/era_files'
    )

    cp.reanalysis_daily = ERA(cp.lat, cp.lon, cp.variables['reanalysis']['daily'], granularity='daily'
                ).load(ram=True, interp_dx=None)
    cp.reanalysis_monthly = ERA(cp.lat, cp.lon, cp.variables['reanalysis']['monthly'], granularity='monthly'
                ).load(ram=True, interp_dx=None)


    cp._save_ds(cp.reanalysis_daily, 'reanalysis_daily', chunks=None)
    cp._save_ds(cp.reanalysis_monthly, 'reanalysis_monthly', chunks=None)

    print(f'Reanalysis for {region} has been uploaded to GCS')