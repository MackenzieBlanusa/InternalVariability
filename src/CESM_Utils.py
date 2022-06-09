import xarray as xr
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import intake 
import pprint

# TO_DO: Make functions more general so they can be applied to any variable 

def fix_lon_temp(data):
    """Converts longitude from 0-360 to -180-180, sorts longitude values so they are monotonically increasing,
    converts temp in Kelvin to celcius 
    
    Parameters
    ----------
    data: a CESM temperature dataset 
    
    Returns
    -------
    data: the corrected dataset 
    """
    
    # convert lon from 0-360 to -180 to 180
    data = data.assign_coords(lon=((data.lon + 180) % 360 - 180))
    # make longitude monotonically inreasing
    # this is necessary in order to use the .sel method=nearest 
    data['lon'] = data.lon.sortby(data.lon,ascending=True)
    # convert Kelvin to celcius 
    data['TREFHT'] = data.TREFHT - 273.15     # make this more general 
    
    return data 

def get_local_data(data,lat,lon):
    """Get local data from CESM dataset, resample yearly, and take decadal rolling average
    
    Parameters
    ----------
    data: a CESM dataset
    lat: latitude point 
    lon: longitude point
    
    Returns
    --------
    data: local dataset of specified lat/lon
    
    """
    # resample yearly
    data = data.resample(time='AS').mean(dim='time')
    # select location based on lat/lon
    data = data.sel(lon=lon,lat=lat,method='nearest')
    # take a decadal rolling average and drop times
    data = data.rolling(time=10, center=True).mean().dropna('time')
    
    return data 

def get_global_data(data):
    """Using CESM dataset, this function resamples the dataset yearly, takes mean of lat/lon to get global coverage, and
    takes decadal rolling average 
    
    Parameters
    ---------
    data: CESM dataset
    
    Returns
    -------
    data: CESM global dataset 

    """
    # resample yearly
    data = data.resample(time='AS').mean(dim='time')
    # take mean lat/lon to get global data 
    data = data.mean(dim=('lon','lat'))
    # take a decadal rolling average and drop times
    data = data.rolling(time=10, center=True).mean().dropna('time')
    
    return data 

def get_local_reference(data,lat,lon,year1,year2):
    """Get reference CESM data for calculating anomalies
    
    Parameters
    ----------
    data: CESM dataset
    lat: latitude point
    lon: longitude point
    year1: start of reference period 
    year2: end of reference period 
    
    Returns
    -------
    data: CESM dataset of reference value locally for given period 
    """
    # here we have 1920-2005 so I will use 1995-2005 as base period 
    data = data.sel(time=slice(year1,year2))
    # resample yearly
    data = data.resample(time='AS').mean(dim='time')
    # select location based on lat/lon
    data = data.sel(lon=lon,lat=lat,method='nearest')
    # take mean of ensembles and time to get one temp for reference period 
    data = data.mean(dim=('member_id','time'))
    
    return data 

def get_global_reference(data,year1,year2):
    """Get reference data for global estimate 
    
    Parameters
    ---------
    data: CESM dataset
    year1: start of reference period 
    year2: end of reference period  
    
    Returns
    -------
    data: CESM dataset of reference value for global coverage for given period 
    """
    # here we have 1920-2005 so I will use 1995-2005 as base period 
    data = data.sel(time=slice(year1,year2))
    # resample yearly
    data = data.resample(time='AS').mean(dim='time')
    # average along lat/lon, time, and member_id to get base temp
    data = data.mean(dim=('lon','lat','member_id','time'))
    
    return data 

def polyfit(data):
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
    for i, m in enumerate(fit.member_id):
        p = np.poly1d(Z[:,i])
        fit[:, i] = p(X)
    
    return fit

def internal_variability(data,data_ref,variable):
    """Calculate internal variablity of CESM data via FIT method and LE method, and merge variables into xarray dataset
    
    Parameters
    ---------
    data: CESM dataset
    data_ref: CESM reference dataset
    variable: string, CESM variable of interest
    
    Returns
    ------
    dataset: New dataset with CESM data, CESM reference data, internal variability estimates, and CESM fit 
    """
    #load the data
    data = data.load()
    data_ref = data_ref.load()
    
    #define variable to use
    data = data[variable]
    data_ref = data_ref[variable]
    data_ref = data_ref.rename('TREFHT_ref')      # figure out how to generalize this 
    # take anomaly
    data = data-data_ref
    data = data.rename('TREFHT')
    #do the 4th order polyfit
    fit = polyfit(data)
    fit = fit.rename('TREFHT_fit')
    # calculate residual
    residual = data - fit
    #calculate internal var via FIT
    internal_fit = residual.var('time').mean('member_id')
    internal_fit = internal_fit.rename('internal_fit')
    
    # calculate internal variability via LE method 
    internal_le = data.var('member_id')
    internal_le = internal_le.rename('internal_le')
    
    #create xarray dataset and merge results together 
    dataset = xr.merge([data,data_ref,fit,internal_fit,internal_le])
    
    return dataset

