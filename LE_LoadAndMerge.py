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

    def load_cesm_lens(self):
        self.variable
        #catalog URL 
        url = "https://raw.githubusercontent.com/NCAR/cesm-lens-aws/main/intake-catalogs/aws-cesm1-le.json"
        raw_cat = intake.open_esm_datastore(url)
        #Define specific search
        cat = raw_cat.search(
            experiment=['RCP85','20C'],
            variable=['TREFHT'],
            frequency=['monthly']
            )
        # load data into xarray datasets
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
        # convert calendar 
        hist = self.convert_calendar(ds=hist,granularity='monthly')
        future = self.convert_calendar(ds=future,granularity='monthly')
        #rename variables
        future = future.rename_vars({'TREFHT':'tas'})
        hist = hist.rename_vars({'TREFHT':'tas'})
        # concat and split so time is in cmip convention
        CESM = xr.concat(
            [hist, future], dim='time'
        )
        hist = CESM.sel(time=slice('1920','2014'))
        future = CESM.sel(time=slice('2015','2100'))
        # conform to CMIP conventions
        # also check lon
        return hist, future 
    
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
        #slice to 2100, some models that go out to 2300 raise error when converting calendar 
        future = future.sel(time=slice('2015','2100'))
        hist = hist.sel(time=slice('1920','2014'))
        # need to add in conver_calendar 
        hist = self.convert_calendar(ds=hist,granularity='monthly')
        future = self.convert_calendar(ds=future,granularity='monthly')
        
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
    
    def convert_calendar(self,ds, granularity: str):
        """Convert to common calendar.

        For "monthly", simply converts to numpy datetime objects and sets dates to first
        of the month.

        For "daily", converts all calendars to standard calendar. Leap years are ignored.
        Error from this is assumed to be small.

        Parameters
        ----------
        ds: xr.Dataset
            CMIP model dataset in either daily or monthly resolution
        granularity: str
            Either "daily" or "monthly"

        Returns
        -------
        ds: xr.Dataset
            Same CMIP dataset with converted calendar

        """

        if granularity == 'daily':
            if type(ds.time.values[0]) in [
                cftime.DatetimeNoLeap,
                cftime.DatetimeProlepticGregorian,
                cftime.DatetimeGregorian,
                cftime.DatetimeJulian
            ]:
                ds = ds.assign_coords({'time': ds.time.values.astype('datetime64[D]')})
            elif isinstance(ds.time.values[0], cftime.Datetime360Day):
                # Whoever came up with this should smolder in hell
                # The code below maps the 360 days to a 365 day calendar
                new_times = []
                for year in np.unique(ds.time.dt.year):
                    times360 = ds.sel(time=str(year)).time
                    delta = np.timedelta64(int(365 / 360 * 24 * 60), 'm')
                    new_time = np.arange(f'{year}-01-01', f'{year}-12-31',
                                         delta, dtype='datetime64').astype('datetime64[D]')
                    new_time = new_time[:len(times360)]
                    new_times.append(new_time)
                new_times = np.concatenate(new_times)
                ds = ds.assign_coords({'time': new_times})

        elif granularity == 'monthly':
            if not type(ds.time.values[0]) is np.datetime64:
                new_time = [
                    np.datetime64(c.isoformat()[:10]).astype('datetime64[M]') for c in ds.time.values
                ]
            else:
                new_time = ds.time.values.astype('datetime64[M]')
            ds = ds.assign_coords({'time': new_time})

        else:
            raise NotImplementedError(f'Granularity {granularity} not implemented.')

        return ds
    
#     def convert_lon(self,lon):
        
#         lon = lon % 360
        
#         return lon
    


# cesm = LargeEnsemble('cesm_lens', ...)
# cesm.hist_path


class MultiModelLargeEnsemble():
    def __init__(self, models, variable, granularity, lat, lon, bucket, path,
                 load=False):
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
        self.lon = lon % 360
        self.bucket = bucket
        self.path = path 
        self.load = load 
        self.le_dict = self.load_large_ensembles()
        self.hist, self.future = self.merge_datasets()
        self.internal_variability = self.compute_internal_variability()

    def load_large_ensembles(self):
        le_dict = {}
        for model in self.models:
            le = LargeEnsemble(model_name=model, variable=self.variable, granularity=self.granularity,
                               lat=self.lat,lon=self.lon,bucket=self.bucket,path=self.path,load=self.load)
            le_dict[model] = le
        return le_dict

    def merge_datasets(self):
        # this is ugly but it works 
        hist_models = []
        future_models = []
        for model in self.models:
            hist = self.le_dict[model].hist
            hist = hist.drop(['lat','lon'])
            hist = hist.assign_coords(model=model)
            hist = hist.expand_dims('model')
            member_number = np.arange(0,len(hist.member_id),1)
            hist = hist.assign_coords({'member_number':member_number})
            hist['tas'] = hist.tas.swap_dims({'member_id':'member_number'})
            hist['member_id'] = hist.member_id.swap_dims({'member_id':'member_number'})
            hist_models.append(hist)

            future = self.le_dict[model].future
            future = future.drop(['lat','lon'])
            future = future.assign_coords(model=model)
            future = future.expand_dims('model')
            member_number = np.arange(0,len(future.member_id),1)
            future = future.assign_coords({'member_number':member_number})
            future['tas'] = future.tas.swap_dims({'member_id':'member_number'})
            future['member_id'] = future.member_id.swap_dims({'member_id':'member_number'})
            future_models.append(future)
        future = xr.concat(future_models,dim='model')
        hist = xr.concat(hist_models,dim='model')
        return hist, future
    
    def polyfit(self,data):
        """Perform 4th order polynomial fit for ensembles in CESM dataset

        Parameters
        ---------
        data: CESM dataset 

        Returns
        -------
        fit: dataset of CESM fitted data 
        """
        # create X and Y variables for the model fit 
        X = np.arange(len(data.time))    # x variable is length of time
        Y = data.values  # y is the temp data

        # the polynomial fit (4th order)
        Z = np.polyfit(X,Y,4)
        fit = data.copy()

        # calculate the fit using coefs from Z
        for i, m in enumerate(fit.model):
            p = np.poly1d(Z[:,i])
            fit[:, i] = p(X)

        return fit
    
    def compute_internal_variability(self):
        """Int Var calculation
        """
    
        # get reference period 
        data_ref = self.hist.sel(time=slice('1995','2014')).resample(time='AS').mean(dim='time')
        data_ref = data_ref.tas - 273.15   # convert to celcius 
        data_ref = data_ref.mean(dim=('time','model','member_number')).rename('tas_ref')
        data_ref.load()

        # prepare temp data
        data = self.future.tas - 273.15 #convert to celcius 
        data = data.transpose()
        # resample yearly
        data = data.resample(time='AS').mean(dim='time')
        #decadal rolling average 
        data = data.rolling(time=10, center=True).mean()   #dropna not working 
        #implicit bias correction
        data = (data-data_ref).rename('tas')
        #load data
        data = data.load()

        # Internal var via LE method 
        ensemble_mean = data.mean('member_number')
        model_le = ensemble_mean.var('model').rename('model_le')
        internal_le = data.var('member_number')
        internal_le = internal_le.mean(dim='model').rename('internal_le')
        total_le = (internal_le + model_le).rename('total_le')
        internal_le_frac = ((internal_le/total_le)*100).rename('internal_le_frac')
        model_le_frac = ((model_le/total_le)*100).rename('model_le_frac')
        total_direct_le = data.var(dim=('model','member_number')).rename('total_direct_le')

        # Internal var via FIT method
        data_fit = data.isel(member_number=0).rename('data_fit')  # select first ensemble member for the fit 
        data_fit = data_fit.dropna(dim='time')   # drop nans, not sure why this only works here when you have two dimensions???
        fit = self.polyfit(data_fit).rename('fit')
        residual  = data_fit - fit
        internal_fit = residual.var('time').rename('internal_fit')
        internal_fit = internal_fit.mean()
        model_fit = fit.var('model').rename('model_fit')
        total_fit = (internal_fit + model_fit).rename('total_fit')
        internal_fit_frac = ((internal_fit/total_fit)*100).rename('internal_fit_frac')
        model_fit_frac = ((model_fit/total_fit)*100).rename('model_fit_frac')
        total_direct_fit = data_fit.var('model').rename('total_direct_fit')

        dataset = xr.merge([data_ref,data,model_le,model_le_frac,internal_le,internal_le_frac,total_le,total_direct_le,
                             data_fit,fit,model_fit,model_fit_frac,internal_fit,internal_fit_frac,total_fit,total_direct_fit],
                            compat='override')
        return dataset 

#     # def compute_internal_variability(self):

# mmle = MultiModelLargeEnsemble([list of models], ...)



        


