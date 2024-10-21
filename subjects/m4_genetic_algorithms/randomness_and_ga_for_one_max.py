# one-max problem
import random

import matplotlib.pyplot as plt

# gene
def generate_01():
    return random.randint(0, 1)

# evaluate
def compute_fitness(vector):
    return sum(gen for gen in vector)

def create_vector(n):
    return [generate_01() for _ in range(n)]

def exchange(vector1, vector2, n, rate=0.9):
    vector1_new = vector1.copy()
    vector2_new = vector2.copy()
    
    for i in range(n):
        if random.random() < rate:
            vector1_new[i] = vector2[i]
            vector2_new[i] = vector1[i]            
    
    return vector1_new, vector2_new

def explore(vector, n, rate=0.05):
    vector_m = vector.copy()
    
    for i in range(n):
        if random.random() < rate:
            vector_m[i] = generate_01()
        
    return vector_m

# population is sorted according to fitness
def selection(sorted_vectors, m):    
    index1 = random.randint(0, m-1)  
    index2 = random.randint(0, m-1)
    
    while index2 == index1:
        index2 = random.randint(0, m-1)
    
    
    vector = sorted_vectors[index1]
    if index2 > index1:
        vector = sorted_vectors[index2]
    
    return vector

def main():
    n_chromo = 20 # size of individual (chromosome)
    m_population = 50 # size of population
    n_generations = 20 # number of generations <--> epochs

    # for presentation data
    fitnesses = []

    # create population
    vectors = [create_vector(n_chromo) for _ in range(m_population)]

    # loops through generations
    for i in range(n_generations):
        # step 1: sort population. This helps to compare fitness between individuals more quickly with their index
        sorted_vectors = sorted(vectors, key=compute_fitness)

        if i % 1 == 0: # look like meaningless
            fitnesses.append(compute_fitness(sorted_vectors[m_population-1]))
            print("BEST:\t", compute_fitness(sorted_vectors[m_population-1]))
        
        new_vectors = []
        while(len(new_vectors) < m_population):
            # step2: selection
            vector_s1 = selection(sorted_vectors, m_population)
            vector_s2 = selection(sorted_vectors, m_population)

            # step3: crossover --> exploit
            vector_c1, vector_c2 = exchange(vector_s1, vector_s2, n_chromo)

            # step4: mutation --> explore
            vector_m1 = explore(vector_c1, n_chromo)
            vector_m2 = explore(vector_c2, n_chromo)

            new_vectors.append(vector_m1)
            new_vectors.append(vector_m2)
        
        # inheritance
        vectors = new_vectors

    sorted_vectors = sorted(vectors, key=compute_fitness)
    vector_best = sorted_vectors[m_population-1]
    print(vector_best)

    plt.plot(fitnesses)
    plt.xlabel('Generations')
    plt.ylabel('Fitness')
    plt.show()

if __name__ == "__main__":
    main()
