import numpy as np

def create_pollution_availability_decrease_timeseries(pollution_resource_availability_decrease_rate, pollution_offset, timesteps):
    
    pollution_resource_availability_decrease_timeseries = np.zeros((len(pollution_resource_availability_decrease_rate), timesteps), dtype = np.float32)

    pollution_resource_availability_decrease_timeseries[:, pollution_offset:] = np.array(pollution_resource_availability_decrease_rate).reshape(-1, 1) * np.arange(timesteps - pollution_offset).reshape(1, -1)

    return pollution_resource_availability_decrease_timeseries

def create_habitat_threshold_increase_timeseries(habitat_threshold_increase_rate, habitat_offset, timesteps):

    habitat_threshold_increase_timeseries = np.zeros((timesteps), dtype = np.float32)

    habitat_threshold_increase_timeseries[habitat_offset:] = np.array(habitat_threshold_increase_rate) * np.arange(timesteps - habitat_offset)

    return habitat_threshold_increase_timeseries