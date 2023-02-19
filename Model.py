import numpy as np
import numba as nb
import scipy

# su = size unit
# ru = resource unit
# tu = time unit

# ------------------
# Models 

# Resource Uptake Model
def resource_uptake_model(max_resource_uptakes, available_resources_per_plant):
    '''
    max_resource_uptakes [ru/plant] = (resource, species, run)
    available_resources_per_plant [ru/plant] = (resource, species, run) 
    [ru/plant] (resource, species, run) 
    '''

    u_1 = 1 # [ru/plant]
    u_2 = 1 # [plant/ru]

    return max_resource_uptakes / (1 + np.exp((u_1 - available_resources_per_plant) * u_2))

# Resource Need Model
# Number of thresholds away from threshold
def resource_need_model(threshold_deviations):
    '''
    threshold_deviations [] = (resource, species, run) 
    [] (resource, species, run) 
    '''

    return 1 / (1 + np.exp(-threshold_deviations)) # Need scaling constants?

# Michaelis-Menten Model
def mechaelis_menton_model(resources):
    '''
    resources [ru/plant] = (resource, species, run) 
    [] (resource, species, run) 
    '''

    n_half = 1 # [ru/plant]

    return resources / (n_half + resources)


# ------------------
# Helper Functions

# Relative Size Per Plant
def calculate_relative_size_per_plant(populations, sizes):
    '''
    populations [plant] = (species, run) 
    sizes [su/plant] = (species, run)
    [1/plant] (species, run) 
    '''

    return sizes / np.sum(populations * sizes, axis = 0)

# Relative Population
def calculate_relative_population(populations):
    '''
    populations [plant] = (species, run) 
    [] (species, run) 
    '''

    return populations / np.sum(populations, axis = 0)


# ------------------
# Calculate Rates

# Available Resources Per Plant
def calculate_available_resources_per_plant(resources, populations, sizes):
    '''
    resources [ru] = (resource, 1, run) 
    populations [plant] = (species, run) 
    sizes [su/plant] = (species, run)
    [ru/plant] (resource, species, run) 
    '''

    return resources * calculate_relative_size_per_plant(populations, sizes)

# Uptook Resources Per Plant
def calculate_uptook_resources_per_plant(available_resources_per_plant, max_resource_uptakes, populations):
    '''
    available_resources_per_plant [ru/plant] = (resource, species, run) 
    max_resource_uptakes [ru/plant] = (resource, species, run) 
    populations [plant] = (species, run)
    [ru/plant] (resource, species, run) 
    '''

    return resource_uptake_model(max_resource_uptakes, available_resources_per_plant)

# Stochastic Cooperation Matrix
# Entries are the resource given per resource have 
def calculate_cooperation_matrix(populations, uptook_resources_per_plant, resource_thresholds, cooperation_matrix): 
    '''
    populations [plant] = (species, run)
    uptook_resources_per_plant [ru/plant] = (resource, species, run)
    resource_thresholds [ru/plant] = (resource, species, run)
    cooperation_matrix [] = (resource, species, species, run)
    [] (resource, species, species, run)
    '''

    # Calculate the needs of each species
    threshold_deviations = (resource_thresholds - uptook_resources_per_plant) / resource_thresholds
    resource_needs = resource_need_model(threshold_deviations)
    relative_needs = resource_needs / np.sum(resource_needs)
    relative_proportional_needs = relative_needs * calculate_relative_population(populations)

    proportion_given = np.sum(relative_proportional_needs, axis = 1, keepdims = True) - relative_proportional_needs

    # Construct matrix
    # Initialize rows
    for s in range(populations.shape[0]):
        cooperation_matrix[s,:] = relative_proportional_needs[s]
        cooperation_matrix[s,s] = 1 - proportion_given[s]

    return cooperation_matrix

# Consumed Resources
def calculate_consumed_resources_per_plant(cooperation_matrix, uptook_resources, populations):
    '''
    cooperation_matrix [] = (resource, species, species, run)
    uptook_resources [ru] = (resource, species, run)
    populations [plant] = (species, run)
    [ru/plant] (resource, species, run)
    '''

    return cooperation_matrix.dot(uptook_resources) / populations # verify works correctly with shapes

# Normal Law of Plant Growth
def calculate_population_changes(consumed_resources_per_plant, reproduction_multipliers, resource_thresholds, populations):
    '''
    consumed_resources_per_plant [ru/plant] = (resource, species, run)
    reproduction_multipliers [1/tu] = (species, run)
    resource_thresholds [ru/plant] = (resource, species, run)
    populations [plant] = (species, run)
    [plant/tu] (species, run)
    '''

    # Calculate the resources in relation to thresholds terms
    mechaelis_menton_terms = mechaelis_menton_model(consumed_resources_per_plant) - mechaelis_menton_model(resource_thresholds)

    return reproduction_multipliers * populations * np.min(mechaelis_menton_terms, axis = 0)


# ------------------
# Integrate Using Euler's Method

# Update Resources
def update_resources(resources, resource_regeneration_rates, uptook_resources, dt):
    '''
    resources [ru] = (resource, 1, run)
    resource_regeneration_rates [ru/tu] = (resource, 1, run)
    uptook_resources [ru] = (resource, species, run)
    dt [tu] = (run)
    (resource, 1, run)
    '''

    resources += resource_regeneration_rates * dt - np.sum(uptook_resources, axis = 1, keepdims = True)

    return np.max(resources, 0, axis = 0, keepdims = True)

# Update Populations
def update_populations(populations, population_changes, dt):
    '''
    populations = (species, run)
    population_changes = (species, run)
    dt = (run)
    (species, run)
    '''

    populations += population_changes * dt

    return np.max(populations, 0, axis = 0, keepdims = True)


# Run main model one step
def mycorrhizal_intervention_model(resources, populations, resource_regeneration_rates,
                                   sizes, max_resource_uptakes, resource_thresholds, reproduction_multipliers,
                                   dt, 
                                   cooperation_matrix):
    '''
    Perform one timestep of the model and returns system state
    '''

    # Calculate Rates
    available_resources_per_plant = calculate_available_resources_per_plant(resources, populations, sizes)
    uptook_resources_per_plant = calculate_uptook_resources_per_plant(available_resources_per_plant, max_resource_uptakes, populations)
    uptook_resources = uptook_resources_per_plant * populations
    cooperation_matrix = calculate_cooperation_matrix(populations, uptook_resources_per_plant, resource_thresholds, cooperation_matrix)
    consumed_resources_per_plant = calculate_consumed_resources_per_plant(cooperation_matrix, uptook_resources, populations)
    population_changes = calculate_population_changes(consumed_resources_per_plant, reproduction_multipliers, resource_thresholds, populations)

    # Integrate
    resources = update_resources(resources, resource_regeneration_rates, uptook_resources, dt)
    populations = update_populations(populations, population_changes, dt)

    return (resources, populations, population_changes,
            available_resources_per_plant, uptook_resources_per_plant, consumed_resources_per_plant,
            cooperation_matrix)