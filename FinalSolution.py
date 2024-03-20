# Import necessary libraries
import numpy as np
import time
import random
import matplotlib.pyplot as plt
from heapq import heappop, heappush

# Class representing the grid environment
class GridEnvironment:
    def __init__(self, size, obstacle_density):
        """
        Initialize the grid environment with obstacles, start, and goal points.

        Parameters:
        - size: Size of the grid (size x size)
        - obstacle_density: Percentage of grid cells occupied by obstacles
        """
        self.size = size
        self.obstacle_density = obstacle_density
        self.grid = self.generate_grid()
        self.start = None
        self.goal = None
        self.place_start_and_goal()

    def generate_grid(self):
        """
        Generate the grid with obstacles based on the specified density.
        Ensure start and goal points are not blocked by obstacles.
        """
        grid = np.zeros((self.size, self.size), dtype=int)
        num_obstacles = int(self.size * self.size * self.obstacle_density / 100)
        obstacle_positions = random.sample(range(self.size * self.size), num_obstacles)

        for pos in obstacle_positions:
            x, y = divmod(pos, self.size)
            grid[x, y] = 1  # 1 represents an obstacle
            if (x, y) == (9, 0) or (x, y) == (0, 6):
                grid[x, y] = 0  # Ensure start and goal points are obstacle-free

        return grid

    def is_valid_cell(self, cell):
        """
        Check if a cell is within the grid boundaries and not occupied by an obstacle.

        Parameters:
        - cell: Tuple representing the (x, y) coordinates of the cell.

        Returns:
        - True if the cell is valid, False otherwise.
        """
        x, y = cell
        return 0 <= x < self.size and 0 <= y < self.size and self.grid[x, y] != 1

    def place_start_and_goal(self):
        """
        Place the start and goal points on valid and obstacle-free cells.
        """
        # Ensure start and goal are not blocked by obstacles
        self.start = self.get_valid_random_cell()
        while not self.is_valid_cell(self.start):
            self.start = self.get_valid_random_cell()

        self.goal = self.get_valid_random_cell()
        while self.start == self.goal or not self.is_valid_cell(self.goal):
            self.goal = self.get_valid_random_cell()

    def get_valid_random_cell(self):
        """
        Get a random valid and obstacle-free cell.

        Returns:
        - A tuple representing the (x, y) coordinates of a valid cell.
        """
        valid_cells = [(x, y) for x in range(self.size) for y in range(self.size) if self.is_valid_cell((x, y))]
        if not valid_cells:
            # Handle the case where no valid cells are available (adjust obstacle density, etc.)
            self.grid = self.generate_grid()
            return self.get_valid_random_cell()

        return random.choice(valid_cells)

# Class representing the Genetic Algorithm for pathfinding
class GeneticAlgorithm:
    def __init__(self, grid_env, start, goal, population_size, generations, crossover_rate=0.8, mutation_rate=0.2):
        """
        Initialize the Genetic Algorithm parameters and attributes.

        Parameters:
        - grid_env: Instance of GridEnvironment representing the grid environment.
        - start: Tuple representing the (x, y) coordinates of the start point.
        - goal: Tuple representing the (x, y) coordinates of the goal point.
        - population_size: Size of the population in each generation.
        - generations: Number of generations to run the algorithm.
        - crossover_rate: Probability of crossover between parents.
        - mutation_rate: Probability of mutation in the offspring.
        """
        self.grid_env = grid_env
        self.start = start
        self.goal = goal
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate

    def run_genetic_algorithm(self):
        """
        Run the Genetic Algorithm to find the best path from start to goal.

        Returns:
        - Tuple containing the best path found and its fitness value.
        """
        population = self.initialize_population()

        for generation in range(self.generations):
            population = self.evolve_population(population)

        # Return the best path and its fitness found in the final population
        best_path = min(population, key=lambda path: self.fitness(path))
        return best_path, self.fitness(best_path)

    def initialize_population(self):
        """
        Initialize the population with random paths.

        Returns:
        - List of paths representing the initial population.
        """
        return [self.generate_random_path() for _ in range(self.population_size)]

    def generate_random_path(self):
        """
        Generate a random path from start to goal.

        Returns:
        - List representing the random path.
        """
        path = [self.start]
        current = self.start

        while current != self.goal:
            next_moves = get_neighbors(current, self.grid_env)
            if not next_moves:
                break  # No valid moves
            next_move = random.choice(next_moves)
            path.append(next_move)
            current = next_move

        return path

    def evolve_population(self, population):
        """
        Evolve the population through selection, crossover, and mutation.

        Parameters:
        - population: List of paths representing the current population.

        Returns:
        - List of paths representing the new population.
        """
        new_population = []

        # Implement elitism (copy over the best individuals)
        #The code first implements elitism, which means that a certain percentage of the best individuals from the current population are directly copied to the new population without any changes.
        elites = int(0.1 * self.population_size)#The number of elites is determined by taking 10% (0.1) of the population size and converting it to an integer value using int().
        new_population.extend(sorted(population, key=lambda path: self.fitness(path))[:elites])
#The sorted() function is used to sort the population based on the fitness of each path, with the lowest fitness (best path) coming first.
        #After adding the elite individuals, the code enters a loop to fill up the remaining positions in the new population until it reaches the desired population size.
        while len(new_population) < self.population_size:
            parent1, parent2 = random.sample(population, 2)#In each iteration of the loop, two parents (parent1 and parent2) are randomly selected from the current population using the random.sample() function.
#With a probability determined by the crossover rate (self.crossover_rate), crossover is performed between the two parents using the crossover() method, resulting in a child path.
            #If the random number generated is greater than or equal to the crossover rate, no crossover occurs, and parent1 is simply copied to the child path.
            if random.random() < self.crossover_rate:
                child = self.crossover(parent1, parent2)
            else:
                child = parent1  # No crossover, just copy the parent
#With a probability determined by the mutation rate (self.mutation_rate), mutation is applied to the child path using the mutate() method.
            if random.random() < self.mutation_rate:
                child = self.mutate(child)
#The child path (either from crossover or from copying the parent) is then added to the new population using the append() method.
# ان الحقنا ال الي طلع معنا ب new population
            new_population.append(child)

        return new_population

    def crossover(self, parent1, parent2):
        """
        Perform crossover (recombination) between two parents.

        Parameters:
        - parent1: List representing the first parent.
        - parent2: List representing the second parent.

        Returns:
        - List representing the child path after crossover.
        """
        # Implement two-point crossover
        # For example, select two random points and swap the segments between them
        point1, point2 = sorted(random.sample(range(1, min(len(parent1), len(parent2)) - 1), 2))

        child = parent1[:point1] + parent2[point1:point2] + parent1[point2:]
        return child

    def mutate(self, path):
        """
        Perform mutation on a path.

        Parameters:
        - path: List representing the path to be mutated.

        Returns:
        - List representing the mutated path.
        """
        # Implement path mutation
        # For example, randomly select a point in the path and change it to a valid neighbor
        mutated_path = path.copy()

        index = random.randint(1, len(mutated_path) - 2)
        mutated_path[index] = random.choice(get_neighbors(mutated_path[index], self.grid_env))

        return mutated_path

    def fitness(self, path):
        """
        Calculate the fitness of a path based on path length and distance to the goal.

        Parameters:
        - path: List representing the path.

        Returns:
        - Fitness value of the path.
        """
        # Updated fitness function to consider both path length and distance to goal
        return len(path) + heuristic(path[-1], self.grid_env.goal)

# Heuristic function for A* search
def heuristic(a, b):
    """
    Calculate the Manhattan distance heuristic between two points.

    Parameters:
    - a: Tuple representing the (x, y) coordinates of the first point.
    - b: Tuple representing the (x, y) coordinates of the second point.

    Returns:
    - Manhattan distance between the two points.
    مسافة مانهاتن، والمعروفة أيضًا بمسافة التاكسي أو مسافة L1، تقيس المسافة بين نقطتين عن طريق جمع الفروق المطلقة لإحداثياتهما على طول كل محور (أفقي وعمودي). وهو يمثل الحد الأدنى لعدد الحركات المطلوبة للانتقال من نقطة إلى أخرى عندما يُسمح فقط بالحركات الرأسية والأفقية (لا توجد حركات قطرية).
    """
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

# A* search algorithm
def astar_search(grid_env, start, goal):
    """
    Perform A* search to find the path from start to goal.

    Parameters:
    - grid_env: Instance of GridEnvironment representing the grid environment.
    - start: Tuple representing the (x, y) coordinates of the start point.
    - goal: Tuple representing the (x, y) coordinates of the goal point.

    Returns:
    - List representing the path from start to goal.
    """
    open_set = [(0, start)]
    came_from = {}

    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}

    while open_set:
        current = heappop(open_set)[1]

        if current == goal:
            path = reconstruct_path(came_from, goal)
            return path

        for neighbor in get_neighbors(current, grid_env):
            tentative_g_score = g_score[current] + 1

            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                heappush(open_set, (f_score[neighbor], neighbor))

    return None  # No path found

def get_neighbors(cell, grid_env):
    """
    Get valid neighbors of a cell in the grid.

    Parameters:
    - cell: Tuple representing the (x, y) coordinates of the cell.
    - grid_env: Instance of GridEnvironment representing the grid environment.

    Returns:
    - List of valid neighboring cells.
    through the combinations of i and j values in the ranges [-1, 0, 1].
    """
    x, y = cell
    neighbors = [(x + i, y + j) for i in [-1, 0, 1] for j in [-1, 0, 1]]
    return [neighbor for neighbor in neighbors if grid_env.is_valid_cell(neighbor)]

def reconstruct_path(came_from, current):
    """
    Reconstruct the path from start to current based on the came_from dictionary.

    Parameters:
    - came_from: Dictionary containing the mapping of cells to their predecessors.
    - current: Tuple representing the (x, y) coordinates of the current cell.

    Returns:
    - List representing the reconstructed path.
    """
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    return path[::-1]

# Function to run an algorithm once and measure time
def run_algorithm_once(algorithm, grid_env, start, goal):
    """
    Run a pathfinding algorithm once and measure the time taken.

    Parameters:
    - algorithm: String indicating the algorithm ('A*' or 'Genetic').
    - grid_env: Instance of GridEnvironment representing the grid environment.
    - start: Tuple representing the (x, y) coordinates of the start point.
    - goal: Tuple representing the (x, y) coordinates of the goal point.

    Returns:
    - Tuple containing the path, cost, and time taken by the algorithm.
    """
    start_time = time.time()

    if algorithm == 'A*':
        path = astar_search(grid_env, start, goal)
        cost = len(path) if path is not None else float('inf')
    else:  # algorithm == 'Genetic'
        genetic_algorithm = GeneticAlgorithm(grid_env, start, goal, population_size=50, generations=10)
        path, cost = genetic_algorithm.run_genetic_algorithm()

    end_time = time.time()

    return path, cost, end_time - start_time

# Constants
size = 10
obstacle_density = 10
generations = 10
start = (9, 0)
goal = (0, 6)

# Create grid environment
grid_env = GridEnvironment(size, obstacle_density)

# Run A* algorithm once and measure time
astar_path, astar_cost, astar_time = run_algorithm_once('A*', grid_env, start, goal)
if astar_path:
    print("A* Algorithm:")
    print(f"Time: {astar_time:.5f} seconds")
    print(f"Path: {astar_path}")
    print(f"Path Cost (transmissions count): {astar_cost}")

# Run Genetic Algorithm once and measure time
genetic_path, genetic_cost, genetic_time = run_algorithm_once('Genetic', grid_env, start, goal)
if genetic_path is not None:
    print("\nGenetic Algorithm:")
    print(f"Time: {genetic_time:.5f} seconds")
    print(f"Path: {genetic_path}")
    print(f"Path Cost (transmissions count): {genetic_cost}")

# Visualization
plt.figure(figsize=(12, 6))

# A* Visualization
plt.subplot(1, 2, 1)
plt.imshow(grid_env.grid, cmap='binary', origin='upper')
plt.scatter(start[1], start[0], color='green', s=50, marker='o', label='Start')
plt.scatter(goal[1], goal[0], color='red', s=50, marker='x', label='Goal')
if astar_path is not None:
    path_x, path_y = zip(*astar_path)
    plt.plot(path_y, path_x, color='blue', label='A* Path')
plt.title("A* Pathfinding")
plt.legend()

# Genetic Algorithm Visualization
plt.subplot(1, 2, 2)
plt.imshow(grid_env.grid, cmap='binary', origin='upper')
plt.scatter(start[1], start[0], color='green', s=50, marker='o', label='Start')
plt.scatter(goal[1], goal[0], color='red', s=50, marker='x', label='Goal')
if genetic_path is not None:
    path_x, path_y = zip(*genetic_path)
    plt.plot(path_y, path_x, color='orange', label='Genetic Path')
plt.title("Genetic Algorithm Pathfinding")
plt.legend()

plt.show()
