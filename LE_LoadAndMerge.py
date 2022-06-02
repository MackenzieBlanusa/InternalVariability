"""
Large Ensemble Class
"""
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import intake 
import pprint
import sys  
sys.path.insert(0, '/home/jupyter/InternalVariability/AdaptationAnalysis')
from app.main.src.utils import *



class LargeEnsemble():
    def __init__(self, model_name, variable, granularity, lat, lon, bucket, path, 
                 load=False):
        """Large Ensemble class used to get CMIP6 and CESM data, and merge.
        
        Parameters
        ---------
        model_name: string, 
            Name of model (source_id in intake)
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
        self.model_name = model_name
        self.variable = variable
        self.granularity = granularity
        self.lat = lat
        self.lon = lon % 360
        self.bucket = bucket
        self.path = path 
        self.load = load 
        
        # self.lon = self.convert_lon()
        
        lat_str = str(lat)
        lon_str = str(lon)
        hist = 'hist'
        future = 'future'
        ds_name_hist = self.model_name+'_'+self.granularity+'_'+hist+'_'+self.variable+'_'+lat_str+'_'+lon_str
        ds_name_future = self.model_name+'_'+self.granularity+'_'+future+'_'+self.variable+'_'+lat_str+'_'+lon_str
        
        # define using self
        self.ds_name_hist = ds_name_hist
        self.ds_name_future = ds_name_future
        self.hist_path = f'gcs://{self.bucket}/{self.path}/{self.ds_name_hist}.zarr'
        self.future_path = f'gcs://{self.bucket}/{self.path}/{self.ds_name_future}.zarr'
        
        # load the saved data or retrieve the data 
        if load:
            self.hist = xr.open_zarr(self.hist_path, consolidated=True)
            self.future = xr.open_zarr(self.future_path, consolidated=True)
        else:
            if model_name == 'cesm_lens':
                self.hist, self.future = self.load_cesm_lens() # xr.DataArray [time, member]
            else:
                self.hist, self.future = self.load_cmip6()

            self.hist_path, self.future_path = self.save()

    # def load_cesm_lens(self):
    #     self.variable
    #     ...
    #     # conform to CMIP conventions
    #     # also check lon
    #     return hist, future 
    
    def drop_bounds_height(self,ds):
        
        """Drop coordinates like 'time_bounds' from datasets,
        which can lead to issues when merging."""
        drop_vars = [vname for vname in ds.coords
                if (('_bounds') in vname ) or ('_bnds') in vname or ('height') in vname]
        return ds.drop(drop_vars)
    
    def load_cmip6(self):
        self.variable 
        
        # catalog URL 
        url = 'https://storage.googleapis.com/cmip6/pangeo-cmip6.json'
        raw_cat = intake.open_esm_datastore(url)
        # Define specific search
        cat = raw_cat.search(
            experiment_id=['historical','ssp585'],
            variable_id=self.variable,
            table_id = self.granularity,
            source_id = self.model_name
            )
        # load dictionary of searched datasets, comma added to ignore warnings 
        dset = cat.to_dataset_dict(zarr_kwargs={'consolidated':True}, storage_options={"anon": True});
        # define the datasets by sorted keys
        keys = sorted(dset.keys())
        hist = dset[keys[0]]
        future = dset[keys[1]]
        # select specific lat/lon
        hist = hist.sel(lat=self.lat,lon=self.lon,method='nearest')
        future = future.sel(lat=self.lat,lon=self.lon,method='nearest')
        #chunk
        hist = hist.chunk({'member_id': 1, 'time': -1})
        future = future.chunk({'member_id': 1, 'time': -1})
        #load 
        hist = hist.load()
        future = future.load()
        # convert lon to -180-180 if needed
        if hist.lon > 180 or hist.lon < -180:
            hist = hist.assign_coords(lon=((hist.lon + 180) % 360 - 180))
        else: 
            pass 

        if future.lon > 180 or future.lon < -180:
            future = future.assign_coords(lon=((future.lon + 180) % 360 - 180))
        else: 
            pass
        #drop bounds
        hist = self.drop_bounds_height(ds=hist)
        future = self.drop_bounds_height(ds=future)
        
        # need to add in conver_calendar 
        
        return hist, future

    def save(self):

        hist_path = self.hist.to_zarr(
            self.hist_path,
            consolidated=True,
            mode='w'
        )
        
        future_path = self.future.to_zarr(
            self.future_path,
            consolidated=True,
            mode='w'
        )
        return hist_path, future_path
    
#     def convert_lon(self,lon):
        
#         lon = lon % 360
        
#         return lon
    


# cesm = LargeEnsemble('cesm_lens', ...)
# cesm.hist_path


# class MultiModelLargeEnsemble():
#     def __init__(self, models, ... same as above):
#         self.models = models  # e.g. ['cesm_lens', 'mpi-esm']
#         ...
#         self.le_dict = self.load_large_ensembles()
#         self.hist, self.future = self.merge_datasets()

#     def load_large_ensembles(self):
#         le_dict = {}
#         for model in self.models():
#             le = LargeEnsemble(...)
#             le_dict[model] = le
#         return le_dict

#     def merge_dataset(self):
#         ...
#         return hist, future # [time, model, member]

#     # def compute_internal_variability(self):

# mmle = MultiModelLargeEnsemble([list of models], ...)



        


