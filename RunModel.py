import numpy as np
import numba as nb
import scipy
from Model import mycorrhizal_intervention_model

def simulate_mycorrhizal_intervention_model(initial_resources, initial_populations, 
                                            resource_regeneration_rates,
                                            sizes, max_resource_uptakes, resource_thresholds, reproduction_multipliers,
                                            dt, timesteps, identifier):
    '''
    # Inital Conditions
    initial_resources
    initial_populations

    # Time Series
    resource_regeneration_rate
    mean_percipitation

    # Plant Characteristics
    sizes
    max_resource_uptakes
    resource_thresholds
    reproduction_multipliers

    # Simulation Parameters
    dt
    timesteps
    identifier
    '''

    # Infer sizes
    run_number = initial_resources.shape[-1]
    resource_number = initial_resources.shape[0]
    species_number = initial_populations.shape[0]

    # Create logs
    populations = np.empty((timesteps, species_number, run_number), dtype = np.float32)
    resources = np.empty((timesteps, resource_number, run_number), dtype = np.float32)
    population_changes = np.empty((timesteps, species_number, run_number), dtype = np.float32)
    available_resources_per_plant = np.empty((timesteps, resource_number, species_number, run_number), dtype = np.float32)
    uptook_resources_per_plant = np.empty((timesteps, resource_number, species_number, run_number), dtype = np.float32)
    consumed_resources_per_plant = np.empty((timesteps, resource_number, species_number, run_number), dtype = np.float32)
    cooperation_matrix = np.empty((timesteps, resource_number, species_number, species_number, run_number), dtype = np.float32)

    # Set initial conditions
    populations[0] = initial_populations
    resources[0] = initial_resources

    # Run simulation loop
    for t in range(timesteps):
        (resources[t + 1], populations[t + 1], population_changes[t + 1],
        available_resources_per_plant[t + 1], uptook_resources_per_plant[t + 1], consumed_resources_per_plant[t + 1],
        cooperation_matrix[t + 1]) = mycorrhizal_intervention_model(resources[t], populations[t], resource_regeneration_rates[t],
                                                                    sizes, max_resource_uptakes, resource_thresholds, reproduction_multipliers,
                                                                    dt, 
                                                                    cooperation_matrix[t + 1])
    
    np.save("populations_%s" % identifier, populations)
    np.save("resources_%s" % identifier, resources)
    np.save("population_changes_%s" % identifier, population_changes)
    np.save("available_resources_per_plant_%s" % identifier, available_resources_per_plant)
    np.save("uptook_resources_per_plant_%s" % identifier, uptook_resources_per_plant)
    np.save("consumed_resources_per_plant_%s" % identifier, consumed_resources_per_plant)
    np.save("cooperation_matrix_%s" % identifier, cooperation_matrix)

def run_simulation():

    initial_resources = 0
    initial_populations = 0

    resource_regeneration_rates = 0

    sizes = 1
    max_resource_uptakes = 0 
    resource_thresholds = 0
    reproduction_multipliers = 0

    dt = 1
    timesteps = 1 
    identifier = "fungus"

    simulate_mycorrhizal_intervention_model(initial_resources, initial_populations, 
                                            resource_regeneration_rates,
                                            sizes, max_resource_uptakes, resource_thresholds, reproduction_multipliers,
                                            dt, timesteps, identifier)