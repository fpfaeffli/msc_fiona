"""
author: Eike E. Köhn
date: June 21, 2024
description: this file serves to create an object that carries all threshold properties that need to be set and that are applicable for observations and models
"""

#%%
class ThresholdParameters:

    def __init__(self, **kwargs):
        # Set default values
        self.percentile = kwargs.get('percentile', 0)
        self.resolution = kwargs.get('resolution', 0)
        self.baseline_start_year = kwargs.get('baseline_start_year', 0)
        self.baseline_end_year = kwargs.get('baseline_end_year', 0)
        self.baseline_type = kwargs.get('baseline_type', 0)
        self.daysinyear = kwargs.get('daysinyear', 0)
        self.aggregation_window_size = kwargs.get('aggregation_window_size', 0)
        self.smoothing_window_size = kwargs.get('smoothing_window_size', 0)
        self.rootdir = kwargs.get('rootdir', 0)

        self.param_names = {
            'percentile': 'Percentile',
            'resolution': 'Temporal resolution of climatology/threshold',
            'baseline_start_year': 'Baseline period start year',
            'baseline_end_year': 'Baseline period end year',
            'baseline_type': 'Baseline type',
            'daysinyear': 'Number of days in climatology/threshold',
            'aggregation_window_size': 'Size of aggregation window in days',
            'smoothing_window_size': 'Size of smoothing window in days',
            'rootdir': 'rootdir'}

    def __repr__(self):
        params = ', '.join(f'{key}={value}' for key, value in self.__dict__.items())
        return f'ThresholdParameters({params})'

    @classmethod
    def standard_instance(cls):
        return cls(
            percentile=90.,
            baseline_start_year=2011,
            baseline_end_year=2021,
            baseline_type='fixed',
            daysinyear=365,
            aggregation_window_size=11,
            smoothing_window_size=31,
            rootdir='/nfs/kryo/work/koehne/roms/analysis/pactcs30/future_sim/extreme_analysis/thresholds_and_climatology/',
        )

    def get_param_by_name(self, name):
        for key, value in self.param_names.items():
            if value == name:
                return getattr(self, key)
        raise ValueError(f'Parameter with name "{name}" not found.')


#%%
#%% Example usage to create the default instance
default_params = ThresholdParameters.standard_instance()
print(default_params)

#%%
# Accessing a parameter by name
print(default_params.get_param_by_name('Percentile'))
# %%
