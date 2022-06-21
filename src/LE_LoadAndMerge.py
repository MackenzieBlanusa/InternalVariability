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
        self.lon = lon % 360       # make sure input lon is in 0-360, convserion to -180 - 180 happens later 
        self.bucket = bucket
        self.path = path 
        self.load = load 
        
        lat_str = str(lat)
        lon_str = str(self.lon)
        
        # define using self
        self.ds_name_hist = f'{self.model_name}_{self.granularity}_hist_{self.variable}_{lat_str}_{lon_str}'
        self.ds_name_future = f'{self.model_name}_{self.granularity}_future_{self.variable}_{lat_str}_{lon_str}'
        self.hist_path = f'gcs://{self.bucket}/{self.path}/{self.ds_name_hist}.zarr'
        self.future_path = f'gcs://{self.bucket}/{self.path}/{self.ds_name_future}.zarr'
        
        # print(self.hist_path)
        # print(self.future_path)
        
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
        """TO_DO : ADD DOC STRING"""
        #catalog URL 
        url = "https://raw.githubusercontent.com/NCAR/cesm-lens-aws/main/intake-catalogs/aws-cesm1-le.json"
        raw_cat = intake.open_esm_datastore(url)
        #Define specific search
        if self.granularity == "Amon":
            frequency = 'monthly'
        elif self.granularity == "day":
            frequency = 'daily'
        if self.variable == 'tas':
            cat = raw_cat.search(
                experiment=['RCP85','20C'],
                variable=['TREFHT'],
                frequency=frequency
                )
        elif self.variable == 'pr' and self.granularity=='Amon':
            cat = raw_cat.search(
                experiment=['RCP85','20C'],
                variable=['PRECL','PRECC'],
                frequency=frequency
                )
        elif self.variable == 'pr' and self.granularity == 'day':
            cat = raw_cat.search(
                experiment=['RCP85','20C'],
                variable=['PRECT'],
                frequency=frequency
                )
            
        # load data into xarray datasets
        dset = cat.to_dataset_dict(zarr_kwargs={'consolidated':True}, storage_options={"anon": True});
        # define the datasets by sorted keys
        keys = sorted(dset.keys())
        hist = self.process_dataset(dataset=dset[keys[0]])
        future = self.process_dataset(dataset=dset[keys[1]])
        future = self.convert_calendar(future,granularity=frequency)
        hist = self.convert_calendar(hist,granularity=frequency)
        # for precip: calculate pr 
        if self.variable == 'pr' and self.granularity == 'Amon':
            hist = self.cesm_total_precip(ds=hist)
            future = self.cesm_total_precip(ds=future)
        elif self.variable == 'pr' and self.granularity == 'day':
            future = future.rename_vars({'PRECT':'pr'})
            hist = hist.rename_vars({'PRECT':'pr'})
            future['pr'] = future.pr * 997
            hist['pr'] = hist.pr * 997  #unit conversion
        #rename temp variable
        elif self.variable == 'tas':
            future = future.rename_vars({'TREFHT':'tas'})
            hist = hist.rename_vars({'TREFHT':'tas'})
        # concat and split so time is in cmip convention
        CESM = xr.concat(
            [hist, future], dim='time'
        )
        hist = CESM.sel(time=slice('1920','2014'))
        future = CESM.sel(time=slice('2015','2100'))
        
        return hist, future 
    
    def cesm_total_precip(self,ds):
        """Doc String
        """
        ds['pr'] = (ds['PRECC'] + ds['PRECL'])*997 # unit conversion 
        ds = ds.drop_vars(['PRECC','PRECL'])
        # CESM precip units are in m/s and CMIP are in kg m^-2 s^-1, I could get CESM units to CMIP units by multiplying by density of water
        # but density of water is ~1 so this wouldnt do anything. Getting weird results for CESM precip right now. 
    
        return ds
    
    def drop_bounds_height(self,ds):
        
        """Drop coordinates like 'time_bounds' from datasets,
        which can lead to issues when merging."""
        drop_vars = [vname for vname in ds.coords
                if (('_bounds') in vname ) or ('_bnds') in vname or ('height') in vname]
        return ds.drop(drop_vars)
    
    def load_cmip6(self):
        """TO_DO : ADD DOC STRING"""
        
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
        hist = self.process_dataset(dataset=dset[keys[0]])
        future = self.process_dataset(dataset=dset[keys[1]])
        #slice to 2100, some models that go out to 2300 raise error when converting calendar 
        future = future.sel(time=slice('2015','2100'))
        hist = hist.sel(time=slice('1920','2014'))
        if self.granularity == 'Amon':
            future = self.convert_calendar(future,granularity='monthly')
            hist = self.convert_calendar(hist,granularity='monthly')
        elif self.granularity == 'day':
            future = self.convert_calendar(future,granularity='daily')
            hist = self.convert_calendar(hist,granularity='daily')
            
        
        return hist, future
    
    def process_dataset(self,dataset):
        """ADD DOC STRING
        """
        # select specific lat/lon
        dataset = dataset.sel(lat=self.lat,lon=self.lon,method='nearest')
        #chunk
        dataset = dataset.chunk({'member_id': 1, 'time': -1})
        #load 
        dss = []
        for member_id in dataset.member_id:
            ds = dataset.sel(member_id = member_id).load()
            dss.append(ds)
            print(psutil.virtual_memory()[3] / 1024 / 1024 / 1024)
        dataset = xr.concat(dss,'member_id')
        # convert lon to -180-180 if needed
        if dataset.lon > 180 or dataset.lon < -180:
            dataset = dataset.assign_coords(lon=((dataset.lon + 180) % 360 - 180))
        #drop bounds
        dataset = self.drop_bounds_height(ds=dataset)
        
        return dataset

    def save(self):
        """TO_DO : ADD DOC STRING"""

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

class MultiModelLargeEnsemble():
    def __init__(self, models, variable, granularity, lat, lon, bucket, path, return_period, hist_slice, coarsen,
                 conseq_days, rolling_average,load=False):
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
        self.return_period = return_period
        self.hist_slice = hist_slice
        self.coarsen = coarsen
        self.conseq_days = conseq_days 
        self.rolling_average = rolling_average
        self.load = load 
        self.le_dict = self.load_large_ensembles()
        self.hist, self.future = self.merge_datasets()
#         if self.granularity == 'Amon':
#             self.internal_variability = self.compute_internal_variability()
#         elif self.granularity == 'day':
#             self.occurance_hist, self.occurance_future = self.quantile_occurance()
#             self.extreme_internal = self.extreme_internal_variability()

    def load_large_ensembles(self):
        """TO_DO : ADD DOC STRING"""
        le_dict = {}
        for model in self.models:
            le = LargeEnsemble(model_name=model, variable=self.variable, granularity=self.granularity,
                               lat=self.lat,lon=self.lon,bucket=self.bucket,path=self.path,load=self.load)
            le_dict[model] = le
        return le_dict

    
    def merge_datasets(self):  
        """TO_DO : ADD DOC STRING"""
        hist_models = []
        future_models = []
        for model in self.models:
            hist = self.le_dict[model].hist
            hist = self.prepare_merge(model=model,data=hist)
            hist_models.append(hist)

            future = self.le_dict[model].future
            future = self.prepare_merge(model=model,data=future)
            future_models.append(future)
        future = xr.concat(future_models,dim='model')
        hist = xr.concat(hist_models,dim='model')
        hist = hist.load()
        future = future.load()
        return hist, future
    
    def prepare_merge(self,model,data):
        """Prepare CMIP and CESM for merging
        """
        dataset = data.drop(['lat','lon'])
        dataset = dataset.assign_coords(model=model)
        dataset = dataset.expand_dims('model')
        member = np.arange(0,len(data.member_id),1)
        dataset = dataset.assign_coords({'member':member})
        dataset[self.variable] = dataset[self.variable].swap_dims({'member_id':'member'})
        dataset['member_id'] = dataset.member_id.swap_dims({'member_id':'member'})
        dataset['time'] = dataset.time.values.astype('datetime64[D]')
        if self.variable == 'tas':
            dataset[self.variable] = dataset[self.variable] - 273.15

        return dataset
    

    
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
    
    def compute_modelLE(self,data):
        ensemble_mean = data.mean('member')
        model = ensemble_mean.var('model')
    
        return model
    
    def compute_internalLE(self,data):
        internal = data.var('member')
        internal = internal.mean(dim='model')
        return internal
    
    def compute_total_uncertainty(self,internal,model):
        total = internal + model
        return total
    
    def compute_total_direct(self,data):
        total_direct = data.var(dim=('model','member'))
        return total_direct
    
    def compute_percent_contribution(self,internal,model,total):
        internal_percent = ((internal/total)*100)
        model_percent = ((model/total)*100)
        return internal_percent,model_percent
    
    def get_fit(self,data):
        first_member = data.isel(member=0).dropna(dim='time')
        fit = self.polyfit(first_member)
        return fit
    
    def compute_internalFIT(self,data,fit):
        residual = data - fit
        internal = residual.var('time').mean()
        return internal
    
    def compute_modelFIT(self,fit):
        model = fit.var('model')
        return model
    
    def compute_totaldirect_fit(self,data):
        total_direct = data.var('model')
        return total_direct
    
    
    def compute_internal_variability(self):
        """Int Var calculation
        TO_DO : ADD DOC STRING
        """
        dataset = xr.Dataset()
        
        # get reference period 
        data_ref = self.hist[self.variable]
        data_ref = data_ref.load()
        data_ref = data_ref.sel(time=slice('1995','2014'))
        dataset[self.variable+'_ref'] = data_ref.resample(time='AS').mean(dim='time').mean(dim=('time','member'))

        # prepare temp data
        data = self.future[self.variable]
        data = data.load()
        data = data.transpose()    # need to transpose for polyfit, time needs to be first dimension
        # resample yearly
        data = data.resample(time='AS').mean(dim='time')
        #decadal rolling average 
        data = data.rolling(time=10, center=True).mean()   #dropna not working 
        #implicit bias correction
        if self.variable == 'tas':
            data = (data-dataset[self.variable+'_ref'])
        elif self.variable == 'pr':
            data = (((data-dataset[self.variable+'_ref'])/dataset[self.variable+'_ref'])*100)      #percent change (not sure if this is right, getting weird results)
        dataset[self.variable] = data 
        
        # Internal var via LE method 
        dataset['model_le'] = self.compute_modelLE(data=dataset[self.variable])
        dataset['internal_le'] = self.compute_internalLE(data=dataset[self.variable])
        dataset['total_le'] = self.compute_total_uncertainty(internal=dataset['internal_le'],model=dataset['model_le'])
        dataset['total_direct_le'] = self.compute_total_direct(data=dataset[self.variable])
        dataset['internal_le_frac'],dataset['model_le_frac']=self.compute_percent_contribution(internal=dataset['internal_le'],
                                                                                               model=dataset['model_le'],
                                                                                               total = dataset['total_le'])
        # Internal var via FIT method
        dataset['fit'] = self.get_fit(data=dataset[self.variable])
        dataset['internal_fit'] = self.compute_internalFIT(data=dataset[self.variable].isel(member=0),
                                                           fit = dataset['fit'])
        dataset['model_fit'] = self.compute_modelFIT(fit=dataset['fit'])
        dataset['total_fit'] = self.compute_total_uncertainty(internal=dataset['internal_fit'],model=dataset['model_fit'])
        dataset['internal_fit_frac'],dataset['model_fit_frac']= self.compute_percent_contribution(internal=dataset['internal_fit'],
                                           model=dataset['model_fit'],
                                           total = dataset['total_fit'])
        dataset['total_direct_fit'] = self.compute_totaldirect_fit(data=dataset[self.variable].isel(member=0))
 
        return dataset


    def quantile_occurance(self, return_period, conseq_days=1, coarsen=1, 
                           hist_slice=slice(None, None)):

        hist = self.hist[self.variable].sel(time=hist_slice)
        future = self.future[self.variable]
        
        #rolling average 
        hist = hist.rolling(time=conseq_days, center=True).mean()
        future = future.rolling(time=conseq_days, center=True).mean()

        #coarsen 
        hist = hist.coarsen(time=coarsen,boundary='trim').max()
        future = future.coarsen(time=coarsen,boundary='trim').max()
        
        # find number of expected events in period covered by x
        expected_events = len(np.unique(hist.time.dt.year)) / return_period
        q = 1 - expected_events / len(hist.time)

        # get quantile 
        quantile = hist.quantile(q, ('time','member'))
        occurance_hist = hist > quantile
        occurance_hist = occurance_hist.where(np.isfinite(hist), np.NaN)
        occurance_future = future > quantile
        occurance_future = occurance_future.where(np.isfinite(future), np.NaN)

        return occurance_hist, occurance_future



#     def quantile_occurance(self):
    
#         hist = self.hist[self.variable].sel(time=slice('1995','2014')) # do we want to use the full 1920-2014 period? 
#         future = self.future[self.variable]
#         quantile = hist.quantile(self.q, ('time','member'))
#         occurance_hist = hist > quantile
#         occurance_hist = occurance_hist.where(np.isfinite(hist), np.NaN)
#         occurance_future = future > quantile
#         occurance_future = occurance_future.where(np.isfinite(future), np.NaN)

#         return occurance_hist, occurance_future




    def extreme_internal_variability(self, return_period, conseq_days=1, coarsen=1, 
                           hist_slice=slice(None, None), rolling_average=10):
    
        dataset = xr.Dataset()
        
        occurance_hist, occurance_future = self.quantile_occurance(
            return_period=return_period,
            conseq_days=conseq_days,
            coarsen=coarsen,
            hist_slice=hist_slice
        )
        
        occurance = occurance_future.resample(time='AS').mean().rolling(
            time=rolling_average, center=True).mean()
        dataset[self.variable+'_occurance'] = occurance 

        # Internal var via LE method 
        dataset['model_le'] = self.compute_modelLE(
            data=dataset[self.variable+'_occurance'])
        dataset['internal_le'] = self.compute_internalLE(
            data=dataset[self.variable+'_occurance'])
        dataset['total_le'] = self.compute_total_uncertainty(
            internal=dataset['internal_le'], model=dataset['model_le'])
        dataset['total_direct_le'] = self.compute_total_direct(
            data=dataset[self.variable+'_occurance'])
        dataset['internal_le_frac'], dataset['model_le_frac']= self.compute_percent_contribution(
            internal=dataset['internal_le'], model=dataset['model_le'], total=dataset['total_le'])
        
        # Internal var via FIT method
        dataset['fit'] = self.get_fit(data=dataset[self.variable+'_occurance'].T)
        dataset['internal_fit'] = self.compute_internalFIT(
            data=dataset[self.variable+'_occurance'].isel(member=0), fit=dataset['fit'])
        dataset['model_fit'] = self.compute_modelFIT(fit=dataset['fit'])
        dataset['total_fit'] = self.compute_total_uncertainty(
            internal=dataset['internal_fit'], model=dataset['model_fit'])
        dataset['internal_fit_frac'], dataset['model_fit_frac']= self.compute_percent_contribution(
            internal=dataset['internal_fit'], model=dataset['model_fit'], total=dataset['total_fit'])
        dataset['total_direct_fit'] = self.compute_totaldirect_fit(
            data=dataset[self.variable+'_occurance'].isel(member=0))

        return dataset

        

        


