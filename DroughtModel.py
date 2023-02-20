import numpy as np

def create_drought_timeseries(normal_mean_rainfall, percipitation_prop_decrease, drought_timesteps, drought_offset, drought_number, timesteps):

    drought_timeseries = np.full((timesteps), normal_mean_rainfall, dtype = np.float32)

    drought_starts = drought_offset + int((timesteps - 2 * drought_offset) / drought_number) * np.arange(drought_number, dtype = np.int32)
    drought_ends = drought_starts + drought_timesteps

    for d in range(drought_number):
        drought_timeseries[drought_starts[d]:drought_ends[d]] *= percipitation_prop_decrease

    return drought_timeseries

def create_drought_availability_decrease_timeseries(drought_timesteps, drought_water_availability_decrease, drought_offset, drought_number, timesteps):
    
    drought_availability_decrease_timeseries = np.zeros((timesteps), dtype = np.float32)

    drought_starts = drought_offset + int((timesteps - 2 * drought_offset) / drought_number) * np.arange(drought_number, dtype = np.int32)
    drought_ends = drought_starts + drought_timesteps

    for d in range(drought_number):
        drought_availability_decrease_timeseries[drought_starts[d]:drought_ends[d]] = drought_water_availability_decrease

    return drought_availability_decrease_timeseries