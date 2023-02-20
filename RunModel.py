import numpy as np
from numba import njit
import scipy
import matplotlib.pyplot as plt
from Model import mycorrhizal_intervention_model
from DroughtModel import create_drought_timeseries
from DroughtModel import create_drought_availability_decrease_timeseries
from KillerModel import create_pollution_availability_decrease_timeseries
from KillerModel import create_habitat_threshold_increase_timeseries
from Analysis import pick_plants_characteristic
from Analysis import calculate_composition_initial_populations

def simulate_mycorrhizal_intervention_model(initial_resources, initial_populations, 
                                            resource_regeneration_rates, resource_availability_decrease, threshold_increase,
                                            sizes, max_resource_uptakes, resource_thresholds, reproduction_multipliers,
                                            species_names,
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
    run_number = initial_resources.shape[0]
    resource_number = initial_resources.shape[1]
    species_number = initial_populations.shape[1]

    # Create logs
    # [time, runs, resources, species]
    populations = np.zeros((timesteps, run_number, 1, species_number), dtype = np.float32)
    resources = np.zeros((timesteps, run_number, resource_number, 1), dtype = np.float32)
    available_resources_per_plant = np.zeros((timesteps, run_number, resource_number, species_number), dtype = np.float32)
    uptook_resources_per_plant = np.zeros((timesteps, run_number, resource_number, species_number), dtype = np.float32)
    consumed_resources_per_plant = np.zeros((timesteps, run_number, resource_number, species_number), dtype = np.float32)
    cooperation_matrix = np.zeros((timesteps, run_number, resource_number, species_number, species_number), dtype = np.float32)

    # Fix shapes
    resource_regeneration_rates = np.expand_dims(resource_regeneration_rates, axis = 3)
    resource_availability_decrease = np.expand_dims(resource_availability_decrease, axis = 3)
    threshold_increase = np.expand_dims(threshold_increase, axis = (2, 3))
    sizes = np.expand_dims(sizes, axis = 1)
    dt = dt.reshape(-1, 1, 1)

    # Set initial conditions
    populations[0] = np.expand_dims(initial_populations, axis = 1)
    resources[0] = np.expand_dims(initial_resources, axis = 2)

    # Create time array
    time = np.arange(timesteps).reshape(-1, 1) * dt.reshape(1, -1)

    # Run simulation loop
    for t in range(timesteps - 1):
        resources[t + 1] = resources[t]
        populations[t + 1] = populations[t]

        (resources[t + 1], populations[t + 1],
        available_resources_per_plant[t + 1], uptook_resources_per_plant[t + 1], consumed_resources_per_plant[t + 1],
        cooperation_matrix[t + 1]) = mycorrhizal_intervention_model(resources[t + 1], populations[t + 1], resource_regeneration_rates[t], resource_availability_decrease[t],
                                                                    sizes, max_resource_uptakes, resource_thresholds + threshold_increase[t], reproduction_multipliers,
                                                                    dt, 
                                                                    cooperation_matrix[t + 1])
    
    species_labels = species_names[0] if populations.shape[-1] == 1 else species_names[0:populations.shape[-1]]
    
    #plt.plot(time[:, 0], consumed_resources_per_plant[:, 0, 0, :] / resource_thresholds[0,0,:], label = "uptake threshold ratio")

    '''
    plt.plot(time[:, 0], available_resources_per_plant[:, 0, 0, :], label = "available")
    plt.plot(time[:, 0], uptook_resources_per_plant[:, 0, 0, :], label = "uptook")
    plt.plot(time[:, 0], consumed_resources_per_plant[:, 0, 0, :], label = "consumed")
    '''
    '''
    total_population = np.sum(populations[:, 0, 0, :], axis = 1, keepdims = True)
    #plt.plot(time[:, 0], resources[:, 0, :, 0], label = ["Water", "Nutrients"])
    plt.plot(time[:, 0], populations[:, 0, 0, :], label = species_labels)
    bottom, top = plt.ylim()
    plt.ylim((0, top))

    plt.legend()
    #plt.show()
    plt.savefig('./Graphs/Run-%s.png' % identifier)
    plt.close()
    '''
    np.save("populations_%s" % identifier, populations)
    np.save("resources_%s" % identifier, resources)
    np.save("available_resources_per_plant_%s" % identifier, available_resources_per_plant)
    np.save("uptook_resources_per_plant_%s" % identifier, uptook_resources_per_plant)
    np.save("consumed_resources_per_plant_%s" % identifier, consumed_resources_per_plant)
    np.save("cooperation_matrix_%s" % identifier, cooperation_matrix)

def run_simulation():   

    # tu = year

    # Physical Constants
    initial_population = 20
    initial_nutrients = 600
    initial_water = 600
    mean_nutrient_influx = 29.94 # nu/year
    mean_rain = 29.94 # in rain/year
    drought_percipitation_prop_decrease = [0.8, 0.6, 0.4]
    drought_water_availability_decrease_value = [0.5, 1, 1.5]
    drought_length = [0.5, 0.5, 0.5]
    drought_number = [5, 4, 3]
    drought_offset = 3
    pollution_offset = 3
    habitat_offset = 3

    # Parameters
    number_runs = 1
    number_resources = 2
    time_step = 1 / 52
    years = 18
    timesteps = int(years / time_step)
    dt = np.full((number_runs), time_step)

    # Create Killers ----------------------------------------------------------------
    # Pollution - reduce available resources per plant over time
    pollution_resource_availability_decrease_rate = np.asarray([[0, 0], [0.05, 0.05]])

    pollution_resource_availability_decrease = np.zeros((2, number_resources, timesteps), dtype = np.float32)
    pollution_resource_availability_decrease[1] = create_pollution_availability_decrease_timeseries(pollution_resource_availability_decrease_rate[1] * time_step, int(pollution_offset / time_step), timesteps)

    # Habitat Destruction - death term and resource loss over time ??? All at once???
    habitat_threshold_increase_rate = np.array([0, 0.005]) # proportion
    
    habitat_threshold_increase = np.zeros((2, timesteps), dtype = np.float32)
    habitat_threshold_increase[1] = create_habitat_threshold_increase_timeseries(habitat_threshold_increase_rate[1] * time_step, int(habitat_offset / time_step), timesteps)

    killers = [[0, 0], [0, 1], [1, 0], [1, 1]]

    # Create Compositions ----------------------------------------------------------------
    species_Nthr_edges = np.linspace(0.05, 0.2, 4)
    species_Wthr_edges = np.linspace(0.05, 0.2, 4)
    species_C_Nmax_edges = species_Nthr_edges * np.linspace(2, 5, 4)
    species_C_Wmax_edges = species_Nthr_edges * np.linspace(2, 5, 4)
    species_R_edges = np.linspace(0, 2, 4)

    species_Nthr_intervals = [2, 0, 1, 1]
    species_Wthr_intervals = [2, 0, 1, 1]
    species_C_Nmax_intervals = [2, 0, 2, 1]
    species_C_Wmax_intervals = [2, 0, 1, 2]
    species_R_intervals = [2, 0, 1, 1]

    species_Nthr = pick_plants_characteristic(species_Nthr_edges, species_Nthr_intervals)
    species_Wthr = pick_plants_characteristic(species_Wthr_edges, species_Wthr_intervals)
    species_C_Nmax = pick_plants_characteristic(species_C_Nmax_edges, species_C_Nmax_intervals)
    species_C_Wmax = pick_plants_characteristic(species_C_Wmax_edges, species_C_Wmax_intervals)
    species_R = pick_plants_characteristic(species_R_edges, species_R_intervals)

    species_population_proportions = [0.3, 0.2, 0.25, 0.25]

    compositions = [[0], [1], [2], [3], [2, 3], [0, 2, 3], [1, 2, 3], [0, 1, 2, 3]]
    
    initial_total_populations = calculate_composition_initial_populations(species_population_proportions, compositions, initial_population)

    species_names = np.array(["Species A", "Species B", "Species C", "Species D"])

    # Create Droughts
    droughts = np.zeros((4, timesteps), dtype = np.float32)
    droughts[1] = create_drought_timeseries(mean_rain, drought_percipitation_prop_decrease[0], int(drought_length[0] / time_step), int(drought_offset / time_step), drought_number[0], timesteps)
    droughts[2] = create_drought_timeseries(mean_rain, drought_percipitation_prop_decrease[1], int(drought_length[1] / time_step), int(drought_offset / time_step), drought_number[1], timesteps)
    droughts[3] = create_drought_timeseries(mean_rain, drought_percipitation_prop_decrease[2], int(drought_length[2] / time_step), int(drought_offset / time_step), drought_number[2], timesteps)
    
    drought_water_availability_decrease = np.zeros((4, timesteps), dtype = np.float32)
    drought_water_availability_decrease[1] = create_drought_availability_decrease_timeseries(int(drought_length[0] / time_step), drought_water_availability_decrease_value[0], int(drought_offset / time_step), drought_number[0], timesteps)
    drought_water_availability_decrease[2] = create_drought_availability_decrease_timeseries(int(drought_length[1] / time_step), drought_water_availability_decrease_value[1], int(drought_offset / time_step), drought_number[1], timesteps)
    drought_water_availability_decrease[3] = create_drought_availability_decrease_timeseries(int(drought_length[2] / time_step), drought_water_availability_decrease_value[2], int(drought_offset / time_step), drought_number[2], timesteps)

    # Create Arrays
    initial_resources = np.empty((number_runs, number_resources), dtype = np.float32)
    resource_regeneration_rates = np.empty((timesteps, number_runs, number_resources), dtype = np.float32)
    resource_availability_decrease = np.empty((timesteps, number_runs, number_resources), dtype = np.float32)
    threshold_increase = np.empty((timesteps, number_runs), dtype = np.float32)

    runs_range = np.arange(number_runs)

    for composition in range(8): #8
        number_species = len(compositions[composition])

        # Initial Conditions
        initial_resources[:, 1].fill(initial_nutrients)
        initial_resources[:, 0].fill(initial_water)
        initial_populations = np.empty((number_runs, number_species), dtype = np.float32)
        initial_populations[runs_range, :] = np.array(initial_total_populations[composition])

        resource_regeneration_rates[:,:,1].fill(mean_nutrient_influx)

        # Plant Characteristics
        sizes = np.full((number_runs, number_species), 1)
        max_resource_uptakes = np.empty((number_runs, number_resources, number_species), dtype = np.float32)
        max_resource_uptakes[runs_range, 0] = np.array(species_C_Wmax[compositions[composition]])
        max_resource_uptakes[runs_range, 1] = np.array(species_C_Nmax[compositions[composition]])
        resource_thresholds = np.empty((number_runs, number_resources, number_species), dtype = np.float32)
        resource_thresholds[runs_range, 0] = np.array(species_Wthr[compositions[composition]])
        resource_thresholds[runs_range, 1] = np.array(species_Nthr[compositions[composition]])
        reproduction_multipliers = np.empty((number_runs, number_species), dtype = np.float32)
        reproduction_multipliers[runs_range, :] = np.array(species_R[compositions[composition]])

        for drought in range(4):
            resource_regeneration_rates[:, :, 0] = droughts[drought].reshape(-1, 1)

            for killer in range(4):
                resource_availability_decrease[:, :, 0] = (drought_water_availability_decrease[drought] + pollution_resource_availability_decrease[killers[killer][0], 0]).reshape(-1, 1)
                resource_availability_decrease[:, :, 1] = pollution_resource_availability_decrease[killers[killer][0], 1].reshape(-1, 1)
                threshold_increase[:, runs_range] = habitat_threshold_increase[killers[killer][1]].reshape(-1, 1)

                identifier = ("c%.f-d%.f-k%.f" % (composition, drought, killer))

                simulate_mycorrhizal_intervention_model(initial_resources, initial_populations, 
                                                        resource_regeneration_rates, resource_availability_decrease, threshold_increase,
                                                        sizes, max_resource_uptakes, resource_thresholds, reproduction_multipliers,
                                                        species_names[compositions[composition]],
                                                        dt, timesteps, identifier)

run_simulation()