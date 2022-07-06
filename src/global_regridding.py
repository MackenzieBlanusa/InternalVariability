import xarray as xr
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import intake 
import pprint
import cftime 
import psutil
import xesmf as xe
from tqdm import tqdm
from fire import Fire
from LE_LoadAndMerge import convert_calendar

import dask

dask.config.set({"distributed.workers.memory.spill": 0.90})
dask.config.set({"distributed.workers.memory.target": 0.80})
dask.config.set({"distributed.workers.memory.terminate": 0.98})


def regrid_ds(ds_in, dx):
    ds_ref = xr.Dataset(
        {
            "lat": (["lat"], np.arange(-90 + dx, 90, dx)),
            "lon": (["lon"], np.arange(0, 360, dx)),
        }
    )
    regridder = xe.Regridder(ds_in, ds_ref, "bilinear", periodic=True)
    ds_out = regridder(ds_in, keep_attrs=True)
    return ds_out

def load_lens_cat(experiment_id, variable_id, table_id, source_id):
    url = "https://raw.githubusercontent.com/NCAR/cesm-lens-aws/main/intake-catalogs/aws-cesm1-le.json"
    raw_cat = intake.open_esm_datastore(url)

    if table_id == "Amon":
        frequency = 'monthly'
    elif table_id == "day":
        frequency = 'daily'
    if variable_id == 'tas':
        cat = raw_cat.search(
            experiment=['RCP85','20C'],
            variable=['TREFHT'],
            frequency=frequency
        )
    elif variable_id == 'tasmax':
        cat = raw_cat.search(
            experiment=['RCP85','20C'],
            variable=['TREFHTMX'],
            frequency=frequency
            )
    elif variable_id == 'tasmin':
        cat = raw_cat.search(
            experiment=['RCP85','20C'],
            variable=['TREFHTMN'],
            frequency=frequency
            )
            
    elif variable_id == 'pr' and table_id=='Amon':
        cat = raw_cat.search(
            experiment=['RCP85','20C'],
            variable=['PRECL','PRECC'],
            frequency=frequency
            )
        raise NotImplementedError
    elif variable_id == 'pr' and table_id == 'day':
        cat = raw_cat.search(
            experiment=['RCP85','20C'],
            variable=['PRECT'],
            frequency=frequency
            )
    return cat

def rename_cesm(ds, variable_id, table_id):
    if variable_id == 'pr' and table_id == 'day':
        ds = ds.rename_vars({'PRECT':'pr'})
        ds['pr'] = ds.pr * 997
    #rename temp variable
    elif variable_id == 'tas':
        ds = ds.rename_vars({'TREFHT':'tas'})
    elif variable_id == 'tasmin':
        ds = ds.rename_vars({'TREFHTMN':'tasmin'})
    elif variable_id == 'tasmax':
        ds = ds.rename_vars({'TREFHTMX':'tasmax'})
    return ds

def concat_cesm(dsets, experiment_id):
#     import pdb; pdb.set_trace()
    keys = sorted(dsets.keys())
    hist = dsets[keys[0]]
    future = dsets[keys[1]]
    ds = xr.concat([hist, future], 'time')
    return ds


def load_cmip_cat(experiment_id, variable_id, table_id, source_id, member_id=None):
    url = 'https://storage.googleapis.com/cmip6/pangeo-cmip6.json'
    raw_cat = intake.open_esm_datastore(url)
    cat = raw_cat.search(
        experiment_id=experiment_id,
        variable_id= variable_id,
        table_id = table_id,
        source_id = source_id,
#         grid_label=['gn','gr','gr1'],
#         member_id=member_id
    )
    if member_id:
        cat = cat.search(
            grid_label=['gn','gr','gr1'],
            member_id=member_id
        )
    return cat

def fix_ecearth_lat(ds):
    lat = xr.open_dataarray('EC-Earth3_lat.nc')
    return ds.assign_coords({'lat': lat})

def regrid_global(dx, bucket, path, source_id, experiment_id, variable_id, table_id='day', 
                  out_chunks={'time': 100_000, 'lon': 5, 'lat': 5}, n_workers=4,
                 ):
    
    cluster = dask.distributed.LocalCluster(
                n_workers=n_workers,
                threads_per_worker=1,
    #             silence_logs=logging.ERROR
    )
    client = dask.distributed.Client(cluster)
    
    # Load catalog
    if source_id == 'cmip6':
        models = ['CMCC-CM2-SR5','CMCC-ESM2',
          'EC-Earth3','EC-Earth3-Veg-LR','GFDL-ESM4','IITM-ESM','INM-CM4-8','INM-CM5-0',
          'IPSL-CM6A-LR','KACE-1-0-G','MIROC6','MPI-ESM1-2-HR','MPI-ESM1-2-LR','NorESM2-MM']
        cat = load_cmip_cat(
            experiment_id=experiment_id,
            variable_id= variable_id,
            table_id = table_id,
            source_id = models,
            member_id='r1i1p1f1'
        )
    elif source_id == 'cesm_lens':
        cat = load_lens_cat(
            experiment_id=experiment_id,
            variable_id= variable_id,
            table_id = table_id,
            source_id = source_id
        )
    else:
        cat = load_cmip_cat(
            experiment_id=experiment_id,
            variable_id= variable_id,
            table_id = table_id,
            source_id = source_id
        )
    
    # Open all datasets
    dsets = cat.to_dataset_dict(
        zarr_kwargs={'consolidated':True}, storage_options={"anon": True}, 
#         aggregate=True,
        aggregate=False,
        preprocess=fix_ecearth_lat if source_id == 'EC-Earth3' else None
    )
    if source_id == 'cesm_lens':
        dsets = concat_cesm(dsets, experiment_id)
        dsets = rename_cesm(dsets, variable_id, table_id)
    else:
        dsets = list(dsets.values())#[0]
    
    save_path = f'gcs://{bucket}/{path}/{source_id}/{experiment_id}/{table_id}/{variable_id}.zarr'
    print('Saving:', save_path)

    
    # Regrid and save every dataset
    first = True
#     for i in tqdm(range(len(dsets.member_id))):
    for i in tqdm(range(len(dsets))):
#         import pdb; pdb.set_trace()
#         ds = dsets.isel(member_id=[i])
        ds = dsets[i]
        ds = ds.assign_coords({'member_id': ds.variant_label}).expand_dims('member_id')
        if source_id == 'cmip6':
            print(ds.source_id)
            assert ds.source_id in models, 'Wrong model name'
            ds = ds.assign_coords({'model': ds.source_id}).expand_dims('model')
        if source_id == 'EC-Earth3':
            ds = fix_ecearth_lat(ds)
#         import pdb; pdb.set_trace()
        
        # time slice
        if experiment_id == 'historical':
            if source_id in ['cmip6', 'EC-Earth3']:
                ds = ds.sel(time=slice('1970', '2014'))
            else:
                ds = ds.sel(time=slice('1920', '2014'))
        else:
            ds = ds.sel(time=slice('2015', '2099'))
        
        ds = convert_calendar(ds, 'daily' if table_id=='day' else 'monthly')
        ds = drop_bounds_height(ds)
        
        if experiment_id == 'historical':
            if source_id in ['cmip6', 'EC-Earth3']:   # Fix for slightly varying time dimensions
                t = np.arange('1970-01-01', '2015-01-01', np.timedelta64(1, 'D'), dtype='datetime64')
                t = xr.DataArray(t, coords={'time': t}, name='tmp')
                ds = xr.merge([ds, t])[[variable_id]]
        else:
            if source_id == 'cmip6':
                t = np.arange('2015-01-01', '2100-01-01', np.timedelta64(1, 'D'), dtype='datetime64')
                t = xr.DataArray(t, coords={'time': t}, name='tmp')
                ds = xr.merge([ds, t])[[variable_id]]
                
#         if ds.lon > 180 or ds.lon < -180:
#             ds = ds.assign_coords(lon=((ds.lon + 180) % 360 - 180))
#         ds = ds.assign_coords({'member_id': ds.variant_label}).expand_dims('member_id')
        ds_out = regrid_ds(ds, dx)
        ds_out = ds_out.chunk(out_chunks)
        del ds_out.attrs['intake_esm_varname']   # Have to do that because can't save None...
        print(ds_out)
        if first:
            ds_out.to_zarr(save_path, consolidated=True, mode='w')
            first = False
        else:
            ds_out.to_zarr(save_path, consolidated=True, mode='a', 
                           append_dim='model' if source_id == 'cmip6' else 'member_id')
        client.restart()
        
def drop_bounds_height(ds):
        
        """Drop coordinates like 'time_bounds' from datasets,
        which can lead to issues when merging."""
        drop_vars = [vname for vname in ds.coords
                if (('_bounds') in vname ) or ('_bnds') in vname or ('height') in vname]
        return ds.drop(drop_vars)
    
if __name__ == '__main__':
    Fire(regrid_global)