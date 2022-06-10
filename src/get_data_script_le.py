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

def main(model_name, variable, granularity, lat, lon, bucket, path,load=False):
    
    LargeEnsemble(model_name=model_name,variable=variable, granularity=granularity, lat=lat, lon=lon, bucket=bucket, path=path,load=False)
    for i in tqdm(range(5)):
        pass

if __name__ == '__main__':
#     main()
    Fire(main)
