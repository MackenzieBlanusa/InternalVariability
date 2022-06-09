import xarray as xr
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import intake 
import pprint
import xesmf
import sys  
sys.path.insert(0, '/home/jupyter/repositories/InternalVariability/AdaptationAnalysis')
from app.main.src.climate_projection import ClimateProjection
from app.main.src.year_of_departure import *
from app.main.src.datasets import *
from app.main.src.post_processing import BiasCorrection, NoPostProcessing

# TO-DO: Make things more general so functions can be applied to any variable 
# Current workflow relies on manually changing lat/lon outside of functions, running the local datasets through the functions
# and then saving the datasets. It would be cool to have a workflow that you could pass a list of various lat/lons and get an 
# output that is a dataset with region as a dimension - it would be more organized and less manual stuff to do. 

def prepare_data(dataset1,dataset2):
    """Concat datasets, resample yearly, drop models without the variable, take decadal rolling average 
    
    Parameters
    ----------
    dataset1: xarray dataset (e.g. cp.pp_hist_monthly)
    dataset2: xarray dataset (e.g. cp.pp_future_monthly)
    
    Returns
    -------
    data: xarray dataset 
    """
    # prepare local data 
    data = xr.concat([dataset1, dataset2], dim='time').resample(time='AS').mean().t2m.dropna('model')
    data = data.rolling(time=10, center=True).mean().dropna('time')
    
    return data

def get_reference_data(dataset,year1,year2):
    """Slice dataset to reference period, resample yearly, and average over models and time
    
    Parameters
    ----------
    dataset: xarray dataset 
    year1: string, start of reference period in years 
    year2: string, end of reference period in years
    
    Returns
    -------
    data_ref: data array 
    """
    # get reference data 
    data_ref = dataset.sel(time=slice(year1,year2))
    data_ref = data_ref.resample(time='AS').mean(dim='time').t2m.dropna('model')
    data_ref = data_ref.mean(dim=('model','time'))
    
    return data_ref

def polyfit(data):
    """Perform 4th order polynomial fit for models 
    
    Parameters
    ---------
    data: xarray dataset 
    
    Returns
    -------
    fit: xarray dataset of fitted data 
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

def internal_variability(data,data_ref):
    """Calculate fourth order polynomial fit of data, residual, internal varaibility, model uncertainity,
    total uncertaintiy (internal + model), direct total uncertainty, fractional contributions to total uncertainty,
    and merge all variables into xarray dataset 
    
    Parameters
    ---------
    data: xarray dataset
    data_ref: xarray data array 
    
    Returns:
    dataset: xarray dataset 
    """
    # need to rename all variables in order to merge later
    data_ref = data_ref.rename('t2m_ref')
    
    # implicit bias correction 
    data = data-data_ref
    data = data.rename('t2m')
    
    #do the 4th order polyfit
    fit = polyfit(data)
    fit = fit.rename('t2m_fit')
    
    # calculate residual
    residual = data - fit
    
    #calculate internal variability 
    internal = residual.var('time').mean('model')
    internal = internal.rename('internal_fit')
    
    #calculate model uncertainty 
    model_uncertainty = fit.var('model')
    model_uncertainty = model_uncertainty.rename('model_fit')
    
    #calculate total uncertainty directly and via variances 
    total_direct = data.var('model')
    total_direct = total_direct.rename('total_direct')
    total = internal + model_uncertainty
    total = total.rename('total')
    
    #calculate internal and model fractional contribiution to total uncertainty (%) 
    internal_frac = (internal/total)*100
    internal_frac = internal_frac.rename('internal_frac')
    model_frac = (model_uncertainty/total)*100
    model_frac = model_frac.rename('model_frac')
    
    #create xarray dataset and merge results together 
    dataset = xr.merge([data,data_ref,fit,internal,model_uncertainty,total_direct,total,internal_frac,model_frac])
    
    return dataset 
