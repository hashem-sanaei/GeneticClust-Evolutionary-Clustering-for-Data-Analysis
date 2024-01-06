import numpy as np
import random
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from multiprocessing import Pool

# Load the Iris dataset
data = np.genfromtxt('iris.data', delimiter=',', dtype='float')
data = np.nan_to_num(data)

# Parameters
n = len(data)
k = 3
num_chromosomes = 50
num_generations = 100

def calculate_centroids(chromosome):
    clusters = {i: [] for i in range(1, k+1)}
    for idx, cluster_id in enumerate(chromosome):
        clusters[cluster_id].append(data[idx])
    centroids = {i: np.mean(clusters[i], axis=0) for i in clusters if len(clusters[i]) > 0}
    return centroids

def advanced_crossover(chromosome1, chromosome2):
    centroids1 = calculate_centroids(chromosome1)
    centroids2 = calculate_centroids(chromosome2)

    child = np.zeros(n, dtype=int)
    for i in range(n):
        dist_to_c1 = euclidean(data[i], centroids1.get(chromosome1[i], np.zeros(data.shape[1])))
        dist_to_c2 = euclidean(data[i], centroids2.get(chromosome2[i], np.zeros(data.shape[1])))
        child[i] = chromosome1[i] if dist_to_c1 < dist_to_c2 else chromosome2[i]
    return child

def targeted_mutation(chromosome, less_optimal_clusters):
    for i in range(n):
        if chromosome[i] in less_optimal_clusters:
            chromosome[i] = random.randint(1, k)
    return chromosome

def initialize_population():
    return [np.random.randint(1, k+1, n) for _ in range(num_chromosomes)]

# Objective function
def objective_function(chromosome, data):
    clusters = {}
    for idx, cluster_id in enumerate(chromosome):
        if cluster_id not in clusters:
            clusters[cluster_id] = []
        clusters[cluster_id].append(data[idx])

    intra_cluster_distance = 0
    for points in clusters.values():
        if len(points) > 1:
            for i in range(len(points)):
                for j in range(i + 1, len(points)):
                    intra_cluster_distance += euclidean(points[i], points[j])

    return -intra_cluster_distance
# Calculate the average silhouette score for the population
def average_population_silhouette(population):
    silhouette_scores = [silhouette_score(data, chrom, metric='euclidean') for chrom in population]
    return np.mean(silhouette_scores)

def evaluate_population(population):
    with Pool() as pool:
        fitness_scores = pool.starmap(objective_function, [(chrom, data) for chrom in population])
    return fitness_scores
if __name__ == '__main__':
    # Genetic algorithm
    population = initialize_population()
    overall_silhouette = average_population_silhouette(population)
    threshold = overall_silhouette

    for generation in range(num_generations):
        # Calculate less optimal clusters based on silhouette scores
        cluster_silhouette_scores = {}  # Implement calculation of silhouette scores per cluster
        # Calculate average silhouette score for the current population
        
        less_optimal_clusters = {cluster for cluster, score in cluster_silhouette_scores.items() if score < threshold}

        for i in range(0, num_chromosomes, 2):
            if i+1 < num_chromosomes:
                child = advanced_crossover(population[i], population[i+1])
                population.append(targeted_mutation(child, less_optimal_clusters))

        # Evaluate fitness and select the best
        fitness_scores = evaluate_population(population)
        population, fitness_scores = zip(*sorted(zip(population, fitness_scores), key=lambda x: x[1], reverse=True))
        population = list(population[:num_chromosomes])

    # Best solution and fitness
    best_solution = population[0]
    best_fitness = fitness_scores[0]
    # Print the best solution and fitness
    print("Best Solution:", best_solution)

    # Visualization
    plt.scatter(data[:,0], data[:,1], c=best_solution)
    plt.title('Cluster Visualization')
    plt.show()

    plt.plot(fitness_scores)
    plt.title('Fitness Scores Over Generations')
    plt.show()


    # Silhouette score
    score = silhouette_score(data, best_solution, metric='euclidean')
    print("Silhouette Score:", score)
