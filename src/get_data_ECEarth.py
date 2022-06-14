# import ...
from fire import Fire
from tqdm import tqdm
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import intake 
import pprint
from LE_LoadAndMerge import LargeEnsemble
import warnings
warnings.filterwarnings("ignore")

def main(granularity, bucket, path,load=False):
    #'cesm_lens','ACCESS-ESM1-5','CanESM5','EC-Earth3'
    
    # also need to get EC-Earth3 data for precip for (41.3,-72.5), I already have temp for that location 
    
    model_name=['EC-Earth3']
    variables = ['tas','pr']
    lats = [37.7,65,51,-25.2,3.9]
    lons = [-122.4,-19,10.5,133.7,-53.1]
    
    for variable in variables:
        for lat,lon in zip(lats,lons):
            for model in model_name:
                LargeEnsemble(model_name=model,variable=variable, granularity=granularity, lat=lat, lon=lon,
                      bucket=bucket, path=path,load=False)
    for i in tqdm(range(5)):
        pass

if __name__ == '__main__':
#     main()
    Fire(main)
