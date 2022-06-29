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

def regrid_global(dx, bucket, path, source_id, experiment_id, variable_id, table_id='day', 
                  out_chunks={'time': 100_000, 'lon': 5, 'lat': 5}, n_workers=4):
    cluster = dask.distributed.LocalCluster(
                n_workers=n_workers,
                threads_per_worker=1,
    #             silence_logs=logging.ERROR
    )
    client = dask.distributed.Client(cluster)
    
    # Load catalog
    url = 'https://storage.googleapis.com/cmip6/pangeo-cmip6.json'
    raw_cat = intake.open_esm_datastore(url)
    cat = raw_cat.search(
        experiment_id=experiment_id,
        variable_id= variable_id,
        table_id = table_id,
        source_id = source_id
    )
    
    # Open all datasets
    dsets = cat.to_dataset_dict(zarr_kwargs={'consolidated':True}, storage_options={"anon": True}, aggregate=False)
    
    save_path = f'gcs://{bucket}/{path}/{source_id}/{experiment_id}/{table_id}/{variable_id}.zarr'
    print('Saving:', save_path)

    
    # Regrid and save every dataset
    first = True
    for k, ds in tqdm(dsets.items()):
        if experiment_id == 'historical':
            ds = ds.sel(time=slice('1920', None))
        else:
            ds = ds.sel(time=slice(None, '2100'))
        ds = ds.assign_coords({'member_id': ds.variant_label}).expand_dims('member_id')
        ds_out = regrid_ds(ds, dx)
        ds_out = ds_out.chunk(out_chunks)
        del ds_out.attrs['intake_esm_varname']   # Have to do that because can't save None...
        ds_out = drop_bounds_height(ds_out)
        if first:
            ds_out.to_zarr(save_path, consolidated=True, mode='w')
        else:
            ds_out.to_zarr(save_path, consolidated=True, mode='a', append_dim='member_id')
        
def drop_bounds_height(ds):
        
        """Drop coordinates like 'time_bounds' from datasets,
        which can lead to issues when merging."""
        drop_vars = [vname for vname in ds.coords
                if (('_bounds') in vname ) or ('_bnds') in vname or ('height') in vname]
        return ds.drop(drop_vars)
    
if __name__ == '__main__':
    Fire(regrid_global)