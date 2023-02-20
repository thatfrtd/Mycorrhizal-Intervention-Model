import numpy as np

def calculate_composition_initial_populations(species_population_proportions, compositions, initial_population):

    initial_populations = []

    for c in range(len(compositions)):
        composition_population_proportions = np.array(species_population_proportions)[compositions[c]]

        composition_population_proportions /= np.sum(composition_population_proportions)

        initial_populations.append(composition_population_proportions * initial_population)

    return initial_populations

def pick_plants_characteristic(characteristic_edges, characteristic_intervals):

    characteristic = np.empty((len(characteristic_intervals)), np.float32)

    for s in range(len(characteristic_intervals)):
        lower = characteristic_edges[characteristic_intervals[s]]
        upper = characteristic_edges[characteristic_intervals[s] + 1]

        characteristic[s] = np.random.uniform(lower, upper)

    return characteristic