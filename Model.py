import numpy as np
from numba import njit
import scipy

# su = size unit
# ru = resource unit
# tu = time unit

# ------------------
# Sub-Models 

# Resource Uptake Model
#@njit
def resource_uptake_model(max_resource_uptakes, available_resources_per_plant):
    '''
    max_resource_uptakes [ru/plant] = (run, resource, species)
    available_resources_per_plant [ru/plant] = (run, resource, species) 
    [ru/plant] (run, resource, species) 
    '''

    h = 29.94 # [ru/plant]
    k = 1 # [plant/ru]

    return max_resource_uptakes / (1 + np.exp((h - available_resources_per_plant) * k))

# Resource Need Model
# Number of thresholds away from threshold
#@njit
def resource_need_model(threshold_deviations):
    '''
    threshold_deviations [] = (run, resource, species) 
    [] (run, resource, species) 
    '''

    return 1 / (1 + np.exp(-threshold_deviations)) # Need scaling constants?

# Michaelis-Menten Model
#@njit
def mechaelis_menton_model(resources):
    '''
    resources [ru/plant] = (run, resource, species) 
    [] (run, resource, species) 
    '''

    n_half = 1 # [ru/plant]

    return resources / (n_half + resources)


# ------------------
# Helper Functions

# Relative Size Per Plant
#@njit
def calculate_relative_size_per_plant(populations, sizes):
    '''
    populations [plant] = (run, 1, species) 
    sizes [su/plant] = (run, 1, species)
    [1/plant] (run, 1, species) 
    '''

    return sizes / np.sum(populations * sizes, axis = 2, keepdims = True)

# Relative Population
#@njit
def calculate_relative_population(populations):
    '''
    populations [plant] = (run, 1, species) 
    [] (run, 1, species) 
    '''

    return populations / np.sum(populations, axis = 2, keepdims = True)


# ------------------
# Calculate Rates

# Available Resources Per Plant
#@njit
def calculate_available_resources_per_plant(resources, populations, sizes):
    '''
    resources [ru] = (run, resource, 1) 
    populations [plant] = (run, 1, species) 
    sizes [su/plant] = (run, 1, species)
    [ru/plant] (run, resource, species) 
    '''

    return resources * calculate_relative_size_per_plant(populations, sizes)

# Uptook Resources Per Plant
#@njit
def calculate_uptook_resources_per_plant(available_resources_per_plant, max_resource_uptakes):
    '''
    available_resources_per_plant [ru/plant] = (run, resource, species) 
    max_resource_uptakes [ru/plant] = (run, resource, species) 
    populations [plant] = (run, 1, species)
    [ru/plant] (run, resource, species) 
    '''

    return resource_uptake_model(max_resource_uptakes, available_resources_per_plant)

# Stochastic Cooperation Matrix
# Entries are the resource given per resource have 
#@njit
def calculate_cooperation_matrix(populations, uptook_resources_per_plant, resource_thresholds, cooperation_matrix): 
    '''
    populations [plant] = (run, 1, species)
    uptook_resources_per_plant [ru/plant] = (run, resource, species)
    resource_thresholds [ru/plant] = (run, resource, species)
    cooperation_matrix [] = (run, resource, species, species)
    [] (run, resource, species, species)
    '''

    # Calculate the needs of each species
    threshold_deviations = (resource_thresholds - uptook_resources_per_plant) / resource_thresholds
    resource_needs = resource_need_model(threshold_deviations)
    relative_needs = resource_needs / np.sum(resource_needs, axis = 2, keepdims = True)
    relative_proportional_needs = relative_needs * calculate_relative_population(populations)

    proportion_given = np.sum(relative_proportional_needs, axis = 2, keepdims = True) - relative_proportional_needs

    # Construct matrix
    # Initialize rows
    for s in range(populations.shape[2]):
        cooperation_matrix[:,:,s,:] = np.expand_dims(relative_proportional_needs[:, :, s], axis = 2)
        cooperation_matrix[:,:,s,s] = 1 - proportion_given[:, :, s]

    return cooperation_matrix

# Consumed Resources
#@njit
def calculate_consumed_resources_per_plant(cooperation_matrix, uptook_resources, populations):
    '''
    cooperation_matrix [] = (run, resource, species, species)
    uptook_resources [ru] = (run, resource, species)
    populations [plant] = (run, 1, species)
    [ru/plant] (run, resource, species)
    '''

    consumed_resources = np.matmul(cooperation_matrix, np.expand_dims(uptook_resources, axis = 3))

    return np.squeeze(consumed_resources, axis = 3) / populations

# Normal Law of Plant Growth
#@njit
def calculate_population_changes(consumed_resources_per_plant, reproduction_multipliers, resource_thresholds, populations):
    '''
    consumed_resources_per_plant [ru/plant/tu] = (run, resource, species)
    reproduction_multipliers [1/tu] = (run, 1, species)
    resource_thresholds [ru/plant/tu] = (run, resource, species)
    populations [plant] = (run, 1, species)
    [plant/tu] (run, 1, species)
    '''

    # Calculate the resources in relation to thresholds terms
    mechaelis_menton_terms = mechaelis_menton_model(consumed_resources_per_plant) - mechaelis_menton_model(resource_thresholds)

    return reproduction_multipliers * populations * np.min(mechaelis_menton_terms, axis = 1, keepdims = True)


# ------------------
# Integrate Using Euler's Method

# Update Resources
#@njit
def update_resources(resources, resource_regeneration_rates, uptook_resources, populations, sizes, dt):
    '''
    resources [ru] = (run, resource, 1)
    resource_regeneration_rates [ru/tu] = (run, resource, 1)
    uptook_resources [ru] = (run, resource, species)
    populations [plant] = (run, 1, species)
    sizes [su/plant] = (run, 1, species)
    dt [tu] = (run, 1, 1)
    (run, resource, 1)
    '''

    #total_size = np.sum(populations * sizes, axis = 2, keepdims = True)

    resources += resource_regeneration_rates * dt - np.sum(uptook_resources, axis = 2, keepdims = True)

    return np.maximum(resources, 0)

# Update Populations
@njit
def update_populations(populations, population_changes, dt):
    '''
    populations = (run, 1, species)
    population_changes = (run, 1, species)
    dt = (run, 1, 1)
    (run, 1, species)
    '''

    populations += population_changes * dt

    return np.maximum(populations, 0)


# ------------------
# Model

# Run main model one step
def mycorrhizal_intervention_model(resources, populations, resource_regeneration_rates, resource_availability_decrease,
                                   sizes, max_resource_uptakes, resource_thresholds, reproduction_multipliers,
                                   dt, 
                                   cooperation_matrix):
    '''
    Perform one timestep of the model and returns system state
    '''

    # Calculate Rates
    available_resources_per_plant = calculate_available_resources_per_plant(resources, populations, sizes)
    uptook_resources_per_plant = calculate_uptook_resources_per_plant(available_resources_per_plant - resource_availability_decrease, max_resource_uptakes)
    uptook_resources = uptook_resources_per_plant * populations
    cooperation_matrix = calculate_cooperation_matrix(populations, uptook_resources_per_plant, resource_thresholds, cooperation_matrix)
    consumed_resources_per_plant = calculate_consumed_resources_per_plant(cooperation_matrix, uptook_resources, populations)
    population_changes = calculate_population_changes(consumed_resources_per_plant, reproduction_multipliers, resource_thresholds, populations)

    # Integrate
    resources = update_resources(resources, resource_regeneration_rates, uptook_resources, populations, sizes, dt)
    populations = update_populations(populations, population_changes, dt)

    return (resources, populations,
            available_resources_per_plant, uptook_resources_per_plant, consumed_resources_per_plant,
            cooperation_matrix)