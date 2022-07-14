from multi_model_large_ensemble import MultiModelLargeEnsemble


# mmle = MultiModelLargeEnsemble(['MIROC6', 'cesm_lens', 'CanESM5', 'MPI-ESM1-2-LR', 'EC-Earth3'], 'tasmax', 'day', 
#                                lat=slice(None, None), lon=slice(None, None), 
#                                bucket='climateai_data_repository', path='tmp/global_cmip_2.5deg')
# mmle.compute_x(x_type='quantile_return', name='tasmax_default_quantile_return_10yr')

# mmle = MultiModelLargeEnsemble(['MIROC6', 'cesm_lens', 'CanESM5', 'MPI-ESM1-2-LR', 'EC-Earth3'], 'pr', 'day', 
#                                lat=slice(None, None), lon=slice(None, None), 
#                                bucket='climateai_data_repository', path='tmp/global_cmip_2.5deg')
# mmle.compute_x(x_type='quantile_return', name='pr_default_quantile_return_10yr')

# mmle = MultiModelLargeEnsemble(['MIROC6', 'cesm_lens', 'CanESM5', 'MPI-ESM1-2-LR', 'EC-Earth3'], 'tas', 'day', 
#                                lat=slice(None, None), lon=slice(None, None), 
#                                bucket='climateai_data_repository', path='tmp/global_cmip_2.5deg')
# mmle.compute_x(x_type='quantile_return', name='tas_default_quantile_return_10yr')


scenarios = ['ssp126', 'ssp245', 'ssp370', 'ssp585']
variables = ['tasmax', 'tas', 'pr']
for v in variables:
    for s in scenarios:
        mmle = MultiModelLargeEnsemble('cmip6', v, 'day', 
                                       lat=slice(None, None), lon=slice(None, None), scenario=s,
                                       bucket='climateai_data_repository', path='tmp/global_cmip_2.5deg')
        mmle.compute_x(x_type='quantile_return', name=f'cmip6_{s}_{v}_default_quantile_return_10yr')

