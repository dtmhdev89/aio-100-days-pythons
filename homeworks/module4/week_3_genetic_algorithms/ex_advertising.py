import random
random.seed(0)

import numpy as np

import matplotlib.pyplot as plt

from dataset_man import dataset_manager

def load_dataset_from_file(fileName='m4.advertising'):
    advertising_path = dataset_manager.get_dataset_path_by_id(fileName)
    data = np.genfromtxt(advertising_path, dtype=None, delimiter=',', skip_header=1)
    features_X = data[:, :3]
    sales_y = data[:, 3]

    features_X = np.c_[np.ones((len(sales_y), 1)), features_X]

    return features_X, sales_y

def generate_gen(bound=10):
    return random.random() * bound - (bound / 2)

def create_individual(n=4, bound=10):
    individual = []

    for i in range(n):
        individual.append(generate_gen(bound))

    return individual

def compute_loss(features_X, sales_y, individual):
    theta = np.array(individual)
    y_hat = features_X.dot(theta)
    loss = np.multiply((y_hat - sales_y), (y_hat - sales_y)).mean()

    return loss

def compute_fitness(features_X, sales_y, individual):
    loss = compute_loss(features_X, sales_y, individual)
    fitness_value = 1 / (loss + 1)

    return fitness_value

def crossover(individual1, individual2, crossover_rate=0.9):
    individual1_new = individual1.copy()
    individual2_new = individual2.copy()

    for i in range(len(individual1)):
        if random.random() < crossover_rate:
            individual1_new[i], individual2_new[i] = individual2[i], individual1[i]

    return individual1_new, individual2_new

def mutate(individual, mutation_rate=0.05, bound=10):
    individual_m = individual.copy()

    for i in range(len(individual)):
        if random.random() < mutation_rate:
            individual_m[i] = generate_gen(bound)

    return individual_m

def initialize_population(m):
    population = [create_individual() for _ in range(m)]

    return population

def selection(sorted_old_population, m=100):
    index1 = random.randint(0, m - 1)
    while True:
        index2 = random.randint(0, m - 1)
        if (index2 != index1):
            break
    
    individual_s = sorted_old_population[index1]
    if index2 > index1:
        individual_s = sorted_old_population[index2]
    
    return individual_s

def create_new_population(features_X, sales_y, old_population, elitism=2, gen=1):
    m = len(old_population)
    sorted_population = sorted(old_population, key=lambda individual: compute_fitness(features_X, sales_y, individual))

    if gen % 1 == 0:
        print("Best loss:", compute_loss(features_X, sales_y, sorted_population[m - 1]), "with chromosome: ", sorted_population[m - 1])

    new_population = []
    while len(new_population) < m - elitism:
        individual1_s = selection(sorted_population, m)
        individual2_s = selection(sorted_population, m)

        individual1_cr, individual2_cr = crossover(individual1_s, individual2_s)

        individual1_m = mutate(individual1_cr)
        individual2_m = mutate(individual2_cr)

        new_population.append(individual1_m)
        new_population.append(individual2_m)

    for ind in sorted_population[m - elitism:]:
        new_population.append(ind)

    return new_population, compute_loss(features_X, sales_y, sorted_population[m - 1])

def run_GA():
    n_generations = 100
    m = 600
    features_X, sales_y = load_dataset_from_file()
    population = initialize_population(m)
    losses_list = []

    for i in range(n_generations):
        old_population = population.copy()
        population, loss = create_new_population(features_X, sales_y, old_population, gen=i)
        losses_list.append(loss)

    return losses_list, population

def visualize_loss(losses_list):
    x_axis = list(range(len(losses_list)))
    plt.plot(x_axis, losses_list)
    plt.show()

def visualize_predict_gt(features_X, sales_y, old_population):
    # visualization of ground truth and predict value
    sorted_population = sorted(old_population, key=lambda individual: compute_fitness(features_X, sales_y, individual))
    print(sorted_population[-1])
    theta = np.array(sorted_population[-1])
    estimated_prices = []
    for feature in features_X :
        estimated_prices.append(feature.T.dot(theta))

    plt.subplots( figsize=(10, 6))
    plt.xlabel('Samples')
    plt.ylabel('Price')
    plt.plot(sales_y, c='green', label= 'Real Prices')
    plt.plot(estimated_prices, c='blue', label='Estimated Prices')
    plt.legend()
    plt.show()

def main():
    features_X, sales_y = load_dataset_from_file()
    print(features_X[:5, :])

    print('---Question 3:\t', sales_y.shape)

    # individual = create_individual()
    # print(individual)

    individual = [4.09, 4.82, 3.10, 4.02]
    fitness_score = compute_fitness(features_X, sales_y, individual)
    print('---Question 4:\t', fitness_score)

    # question 5
    individual1 = [4.09, 4.82, 3.10, 4.02]
    individual2 = [3.44, 2.57, -0.79, -2.41]
    individual1 , individual2 = crossover(individual1, individual2, 2.0)
    print('---Question5:')
    print("individual1: ", individual1)
    print("individual2: ", individual2)

    before_individual = [4.09, 4.82, 3.10, 4.02]
    after_individual = mutate(individual, mutation_rate=2.0)
    print('---Question 6:')
    print(before_individual)
    print(after_individual)
    print(before_individual == after_individual)

    individual1 = [4.09, 4.82, 3.10, 4.02]
    individual2 = [3.44, 2.57, -0.79, -2.41]
    old_population = [individual1, individual2]
    print('---Question 7:')
    new_population , _ = create_new_population(features_X, sales_y, old_population, elitism=2, gen=1)

    print('---Ex 10:')
    losses_list, latest_population = run_GA()
    visualize_loss(losses_list)

    visualize_predict_gt(features_X, sales_y, latest_population)

if __name__ == "__main__":
    main()
