# ai-project
 Optimizing Himmelblau Function with Genetic Algorithm
Pandit Deendayal Energy University 
Department of ICT 
Sem-VI 
AI Systems Lab 
Enrolment No: 21BIT135 ICT Department.
Project

Aim: To optimize Himmelblau Function using Genetic Algorithm .

Theory: 
The Himmelblau function, named after mathematician David M. Himmelblau, is a popular 
test problem in optimization and computational mathematics. It's commonly used to 
benchmark optimization algorithms due to its multiple local minima and complex landscape.
Mathematical Formulation:
f (x, y) = (x2 + y - 11)2 + (x + y2
- 7)2
The Himmelblau function possesses four identical global minima, each with a function value 
of zero. These minima are symmetrically positioned within the function's landscape. 
Specifically, the global minima occur at the points:
f (3,2) = f (−2.805118,3.131312) = f (−3.779310, −3.283186) = f (3.584428, −1.848126) = 0
These points represent the lowest possible values of the function across its entire domain. 
Despite the presence of numerous local minima and other critical points, the global minima 
stand out as the primary targets for optimization algorithms seeking to minimize the 
function. The identification and convergence to these global minima pose significant 
challenges due to the complex and multimodal nature of the function's landscape. 
Nonetheless, the presence of these globally optimal points highlights the utility of the 
Himmelblau function as a benchmark for evaluating optimization algorithms and exploring 
nonlinear, multimodal optimization problems.

Optimization Challenges:
1. Multiple Optima:
 - The presence of numerous local minima and maxima poses a significant challenge for 
optimization algorithms.
 - Algorithms may struggle to differentiate between global and local optima, potentially 
converging on suboptimal solutions.
2. Narrow Valleys:
 - The narrow valleys within the function's landscape hinder optimization algorithms' ability 
to explore the search space effectively.
 - Algorithms may encounter difficulties in convergence or become trapped in local 
minima.
3. Gradient Information:
 - The Himmelblau function does not readily provide gradient information, rendering it 
unsuitable for gradient-based optimization methods.
 - Optimization algorithms must rely on heuristic approaches such as random search or 
evolutionary algorithms to navigate the search space effectively.
Sure, here are the algorithm steps for the provided genetic algorithm code:
Algorithm:
1. Initialization:Define parameters such as population size, number of variables, bits per 
variable, variable ranges, number of generations, tournament size, crossover rate, and 
mutation rate.Generate an initial population of individuals using generate_population 
function.
2. Evaluation:Evaluate the fitness of each individual in the population using the 
evaluate_population function. This is done by decoding the binary representation of 
each individual to obtain real values for each variable, then evaluating the fitness 
using the Himmelblau function.
3. Main Loop:Iterate through a specified number of generations.
4. Selection:Perform tournament selection to choose parents for the next generation. This 
is done by selecting individuals randomly for tournaments, comparing their fitness, 
and selecting the best ones as parents.
5. Crossover and Mutation:Perform crossover and mutation operations to generate new 
offspring. Crossover is applied with a certain probability (crossover_rate). Mutation is 
applied to each bit with a certain probability (mutation_rate).
Pandit Deendayal Energy University 
6. Replacement:Replace the old population with the new population of offspring.
7. Termination: Repeat steps 2 to 6 for a specified number of generations.Terminate the 
algorithm and return the best solution found along with its fitness.
8. Output: Print the best solution found and its fitness.

Code: 
import numpy as np
# Define the Himmelblau function
def himmelblau(x, y):
return (x**2 + y - 11)**2 + (x + y**2 - 7)**2
# Define the function to generate a population of individuals
def generate_population(pop_size, num_variables, bits_per_variable, variable_ranges):
population = []
for _ in range(pop_size):
individual = ''
for i in range(num_variables):
lower_bound, upper_bound = variable_ranges[i]
# Generate binary representation for each variable
bits = format(np.random.randint(0, 2**bits_per_variable), 
'0{}b'.format(bits_per_variable))
# Scale the binary value to the variable range
scaled_value = lower_bound + int(bits, 2) * (upper_bound - lower_bound) /
(2**bits_per_variable - 1)
individual += bits
population.append(individual)
return population
# Define the function to evaluate the fitness of each individual in the population
def evaluate_population(population, num_variables, bits_per_variable, variable_ranges):
fitness_scores = []
for individual in population:
decoded_values = []
for i in range(num_variables):
lower_bound, upper_bound = variable_ranges[i]
# Extract bits corresponding to each variable and decode it to the real value
bits = individual[i*bits_per_variable:(i+1)*bits_per_variable]
decoded_value = lower_bound + int(bits, 2) * (upper_bound - lower_bound) /
(2**bits_per_variable - 1)
decoded_values.append(decoded_value)
# Evaluate fitness using the Himmelblau function
fitness = himmelblau(*decoded_values)
fitness_scores.append(fitness)
Pandit Deendayal Energy University 
Department of ICT 
Sem-VI 
AI Systems Lab 
Enrolment No: 21BIT135 ICT Department.
return fitness_scores
# Define the function for tournament selection
def tournament_selection(population, fitness_scores, tournament_size):
selected_parents = []
for _ in range(len(population)):
# Randomly select individuals for the tournament
tournament_indices = np.random.choice(len(population), tournament_size, 
replace=False)
tournament_fitness = [fitness_scores[i] for i in tournament_indices]
# Select the winner of the tournament (individual with the lowest fitness)
winner_index = tournament_indices[np.argmin(tournament_fitness)]
selected_parents.append(population[winner_index])
return selected_parents
# Define the function for crossover
def crossover(parent1, parent2, crossover_rate):
if np.random.rand() < crossover_rate:
# Perform crossover at a randomly chosen point
crossover_point = np.random.randint(1, len(parent1))
child1 = parent1[:crossover_point] + parent2[crossover_point:]
child2 = parent2[:crossover_point] + parent1[crossover_point:]
return child1, child2
else:
return parent1, parent2
# Define the function for mutation
def mutate(individual, mutation_rate):
mutated_individual = ''
for bit in individual:
# Apply mutation with a certain probability to each bit
if np.random.rand() < mutation_rate:
mutated_bit = '0' if bit == '1' else '1'
else:
mutated_bit = bit
mutated_individual += mutated_bit
return mutated_individual
# Define the genetic algorithm
def genetic_algorithm(pop_size, num_variables, bits_per_variable, variable_ranges, 
num_generations, tournament_size, crossover_rate, mutation_rate):
# Generate an initial population
population = generate_population(pop_size, num_variables, bits_per_variable, 
variable_ranges)
best_solution = None
best_fitness = float('inf')
# Iterate through a specified number of generations
for generation in range(num_generations):
Pandit Deendayal Energy University 
Department of ICT 
Sem-VI 
AI Systems Lab 
Enrolment No: 21BIT135 ICT Department.
# Evaluate the fitness of the population
fitness_scores = evaluate_population(population, num_variables, bits_per_variable, 
variable_ranges)
# Update the best solution found so far
for i, ind_fitness in enumerate(fitness_scores):
if ind_fitness < best_fitness:
best_fitness = ind_fitness
best_solution = population[i]
# Select parents using tournament selection
selected_parents = tournament_selection(population, fitness_scores, 
tournament_size)
new_population = []
# Perform crossover and mutation to generate new offspring
for i in range(0, pop_size, 2):
parent1 = selected_parents[i]
parent2 = selected_parents[i+1]
child1, child2 = crossover(parent1, parent2, crossover_rate)
child1 = mutate(child1, mutation_rate)
child2 = mutate(child2, mutation_rate)
new_population.extend([child1, child2])
population = new_population
# Print best fitness in each generation
print(f"Generation {generation + 1}, Best Fitness: {best_fitness}")
return best_solution, best_fitness
if __name__ == "__main__":
# Set up parameters
pop_size = 20
num_variables = 2
bits_per_variable = 8
variable_ranges = [(-5, 5), (-5, 5)]
num_generations = 20
tournament_size = 3
crossover_rate = 0.9
mutation_rate = 0.1
# Call the genetic_algorithm function
best_solution, best_fitness = genetic_algorithm(pop_size, num_variables, 
bits_per_variable, variable_ranges, num_generations, tournament_size, crossover_rate, 
mutation_rate)
# Print the best solution and its fitness
print("Best solution:", best_solution)
decoded_values = [int(best_solution[i*bits_per_variable:(i+1)*bits_per_variable], 2) /
((2**bits_per_variable - 1) / (variable_ranges[i][1] - variable_ranges[i][0])) +
variable_ranges[i][0] for i in range(num_variables)]
print("Decoded values:", decoded_values)
print("Best fitness:", best_fitness)
Pandit Deendayal Energy University 
Department of ICT 
Sem-VI 
AI Systems Lab 
Enrolment No: 21BIT135, 21BIT138 ICT Department.
