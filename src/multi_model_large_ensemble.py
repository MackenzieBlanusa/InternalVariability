"""
Large Ensemble Class
"""
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import intake 
import pprint
import cftime 
import psutil
import dask
import math


from tqdm import tqdm
def loop_over_chunks(hist, future, f, n_chunks=4, restart_every=10):
    n_lat, n_lon = hist.data.shape[2:]
    chunksize_lat, chunksize_lon = hist.data.chunksize[2:]
    n_chunks_lat = math.ceil(n_lat / chunksize_lat)
    n_chunks_lon = math.ceil(n_lon / chunksize_lon)
    out_lat_hist, out_lat_future = [], []
    counter = 0
    pbar = tqdm(total=n_chunks_lat*((n_chunks_lon + n_chunks) // n_chunks))
    for i in range(n_chunks_lat):
        lat_start = i * chunksize_lat
        lat_end = (i + 1) * chunksize_lat
        out_lon_hist, out_lon_future = [], []
        for j in range((n_chunks_lon + n_chunks) // n_chunks):
            lon_start = j * (chunksize_lon * n_chunks)
            lon_end = (j + 1) * (chunksize_lon * n_chunks)
    #         print((lat_start, lat_end), (lon_start, lon_end))
            chunk_hist = hist.isel(lat=slice(lat_start, lat_end), lon=slice(lon_start, lon_end)).load()
            chunk_future = future.isel(lat=slice(lat_start, lat_end), lon=slice(lon_start, lon_end)).load()
            result_hist, result_future = f(chunk_hist, chunk_future)
            out_lon_hist.append(result_hist); out_lon_future.append(result_future)
            counter += 1
            pbar.update(1)
            if counter % restart_every == 0:
                client.restart()
        out_lat_hist.append(xr.concat(out_lon_hist, 'lon')); out_lat_future.append(xr.concat(out_lon_future, 'lon'))
    out_hist = xr.concat(out_lat_hist, 'lat'); out_future = xr.concat(out_lat_future, 'lat')
    return out_hist, out_future


class MultiModelLargeEnsemble():
    def __init__(self, models, variable, granularity, lat, lon, bucket, path):
        """Multi Model Large Ensemble class used to get CMIP6 and CESM data, and merge.
        
        Parameters
        ---------
        models: list, 
            Name of models (source_id in intake)
        variable: string,
            Name of variable 
        granularity: string,
            "monthly" or "daily"
        lat: float
            Latitude, single location, from -90 to 90 
        lon: float,
            Longitude, single location, from -180 to 180 
        bucket: string
            Bucket on google cloud to save data in 
        path: string
            Path to save data 
        load: True or False, default is False
            Load saved data 
            
        """
        self.models = models  # e.g. ['cesm_lens', 'mpi-esm']
        self.variable = variable
        self.granularity = granularity
        self.lat = lat
        self.lon = lon
        self.bucket = bucket
        self.path = path 
#         self.load = load 
        
        self.hist_dsets, self.future_dsets = self.load_datasets()   # dicts at this point with model as key
        self.x = None
        self.results = xr.Dataset()
        
        
    def load_datasets(self):
        hist_dsets, future_dsets = {}, {}
        for model in self.models:
            hist_path = f'gcs://{self.bucket}/{self.path}/{model}/historical/{self.granularity}/{self.variable}.zarr'
            future_path = f'gcs://{self.bucket}/{self.path}/{model}/ssp585/{self.granularity}/{self.variable}.zarr'
            hist = xr.open_zarr(hist_path, consolidated=True)[self.variable]
            future = xr.open_zarr(future_path, consolidated=True)[self.variable]
            hist = hist.assign_coords({'member_id': np.arange(len(hist.member_id)) + 1})
            future = future.assign_coords({'member_id': np.arange(len(future.member_id)) + 1})
            if type(self.lat) is slice:
                hist = hist.sel(lat=self.lat, lon=self.lon)
                future = future.sel(lat=self.lat, lon=self.lon)
            else:
                hist = hist.sel(lat=[self.lat], lon=[self.lon], method='nearest')
                future = future.sel(lat=[self.lat], lon=[self.lon], method='nearest')
            hist_dsets[model] = hist
            future_dsets[model] = future
        return hist_dsets, future_dsets
        
    def compute_x(self, x_type='quantile_return', load=False, name='', **kwargs):
#         cluster = dask.distributed.LocalCluster(
#                     n_workers=8,
#                     threads_per_worker=1,
#         #             silence_logs=logging.ERROR
#         )
#         self.client = dask.distributed.Client(cluster)
        
        
#         x_hist = []
#         x_future = []
        x = []
        for model in self.models:
            if x_type == 'quantile_return':
                hist, future = self.hist_dsets[model], self.future_dsets[model]
                out = self.compute_quantile_return(hist, future, **kwargs)
#                 x_hist.append(out[0]); x_future.append(out[1])
                x.append(out)
        
        model_dim = xr.DataArray(self.models, coords={'model': self.models}, name='model')
#         x_hist = xr.concat(x_hist, dim=model_dim)
#         x_future = xr.concat(x_future, dim=model_dim)
#         self.x = xr.concat([x_hist, x_future], 'time')
#         self.x_hist = x_hist; self.x_future = x_future
        self.x = xr.concat(x, dim=model_dim)
    
    def compute_quantile_return(self, hist, future, return_period=10, consec_days=1, coarsen=1, 
                                hist_slice=slice(None, None), 
                                rolling_average=10
                               ):
        hist = hist.sel(time=hist_slice)
        future = future
        
        #rolling average 
        hist = hist.rolling(time=consec_days, center=True).mean()
        future = future.rolling(time=consec_days, center=True).mean()

        #coarsen 
        hist = hist.coarsen(time=coarsen, boundary='trim').max()
        future = future.coarsen(time=coarsen, boundary='trim').max()
        
        # find number of expected events in period covered by x
        expected_events = len(np.unique(hist.time.dt.year)) / return_period
        q = 1 - expected_events / len(hist.time)
        
        def quantile_func(da_hist, da_future):
            q_values = da_hist.quantile(q, ('time', 'member_id'))
            occ_hist = da_hist > q_values
            occ_future = da_future > q_values
            occ_future = occ_future.assign_coords({'q_values': q_values})
            occ_hist = occ_hist.assign_coords({'q_values': q_values})
            return occ_hist, occ_future
            
        occ_hist, occ_future = loop_over_chunks(hist, future, quantile_func)
        
        # Resample and average
#         occ_hist = occ_hist.resample(time='AS').sum().rolling(time=rolling_average, center=True).sum()
#         occ_future = occ_future.resample(time='AS').sum().rolling(time=rolling_average, center=True).sum()
#         return occ_hist, occ_future
        occ = xr.concat([occ_hist, occ_future], 'time')
        occ = occ.resample(time='AS').sum().rolling(time=rolling_average, center=True).sum()
        print('WARNING: Check again for EC-Earth with different # members')
        return occ
        
        
    
    def compute_fit(self):
        data = self.x.isel(member_id=0)
        coeffs = data.polyfit('time', 4)
        fit = xr.polyval(data.time, coeffs)
        return fit['polyfit_coefficients']
    
    def compute_LE(self):
        self.results['M_LE'] = self.x.mean('member_id').var('model')
        self.results['I_LE'] = self.x.var('member_id')
        self.results['Ibar_LE'] = self.results['I_LE'].mean('model')
        self.results['T_LE'] = self.x.var(('model', 'member_id'))
        
    def compute_FIT(self):
        self.results['FIT'] = self.compute_fit()
        self.results['M_FIT'] = self.results['FIT'].var('model')
        self.results['I_FIT'] = (self.x - self.results['FIT']).var('time')
        self.results['Ibar_FIT'] = self.results['I_LE'].mean('model')
        self.results['T_FIT'] = self.x.var(('model', 'member_id'))