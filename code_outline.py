class LargeEnsemble():
    def __init__(self, model_name, variable, granularity, lat, lon, bucket, path, 
                 load=False):
        self.model_name = model_name
        self.variable = variable
        ...

        self.hist_path = f'gcs://{self.bucket}/{self.path}/{...}.zarr'
        # same for future
        if load:
            self.hist = xr.open_zarr(self.hist_path, consolidated=True)
        else:
            if model_name == 'cesm_lens':
                self.hist, self.future = self.load_cesm_lens() # xr.DataArray [time, member]
            else:
                self.hist, self.future = self.load_cmip6()

            self.hist_path, self.future_path = self.save()

    def load_cesm_lens(self):
        self.variable
        ...
        # conform to CMIP conventions
        # also check lon
        return hist, future 
    
    def load_cmip6(self):
        self.variable 
        # some other modifications to make compatible
        return hist, future

    def save(self):

        self.hist.to_zarr(
            self.hist_path,
            consolidated=True,
        )
        return hist_path, future_path


cesm = LargeEnsemble('cesm_lens', ...)
cesm.hist_path


class MultiModelLargeEnsemble():
    def __init__(self, models, ... same as above):
        self.models = models  # e.g. ['cesm_lens', 'mpi-esm']
        ...
        self.le_dict = self.load_large_ensembles()
        self.hist, self.future = self.merge_datasets()

    def load_large_ensembles(self):
        le_dict = {}
        for model in self.models():
            le = LargeEnsemble(...)
            le_dict[model] = le
        return le_dict

    def merge_dataset(self):
        ...
        return hist, future # [time, model, member]

    # def compute_internal_variability(self):

mmle = MultiModelLargeEnsemble([list of models], ...)


    
        


