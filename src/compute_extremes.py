from multi_model_large_ensemble import MultiModelLargeEnsemble

mmle = MultiModelLargeEnsemble(['MIROC6', 'cesm_lens', 'CanESM5', 'MPI-ESM1-2-LR'], 'tas', 'day', 
                               lat=slice(None, None), lon=slice(None, None), 
                               bucket='climateai_data_repository', path='tmp/global_cmip_2.5deg')
mmle.compute_x(x_type='quantile_return', name='default_quantile_return_10yr')