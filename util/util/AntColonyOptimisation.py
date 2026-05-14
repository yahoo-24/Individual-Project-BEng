import numpy as np
from .Abstraction import max_pool, redefine_values, avg_pool

# Algorithm hyperparameters
TAU_0 = 2.0  # Initial pheromone level
ALPHA = 2.0  # Pheromone influence exponent
BETA = 2  # Heuristic influence exponent
Q = 5  # Pheromone deposit constant
RHO = 0.08  # Pheromone evaporation rate
ANT_POPULATION = 100  # Number of ants per iteration
ITERATIONS = 100  # Maximum number of iterations

class AntColony():
    """
    Implements Ant Colony Optimization algorithm for solving the Traveling Salesman Problem.
    
    Uses pheromone trails and heuristic distances to guide ants toward optimal paths.
    """
    
    def __init__(self, nodes, distances):
        """
        Initializes the Ant Colony Optimization solver.
        
        Args:
            nodes: An nx2 numpy array of node coordinates.
            distances: An nxn numpy array of distances between all pairs of nodes.
        """
        self.nodes = nodes
        self.distances = distances
        self.num_of_nodes = len(nodes)
        self.__initialise_unvisited_nodes()

        self.__initialise_heuristic()
        self.__initialise_pheromones()
        self.__initialise_paths()
        self.__all_unnormalised_posteriors()

    def __initialise_paths(self):
        """
        Initializes the paths array to store ant trajectories.
        
        Creates a zero array to record which nodes each ant visits.
        """
        self.ant_paths = np.zeros((ANT_POPULATION, self.num_of_nodes - 1, 2), dtype=np.uint16)

    def __initialise_unvisited_nodes(self):
        """
        Initializes the list of unvisited nodes.
        
        Tracks which nodes have not yet been visited in the current path.
        """
        self.unvisited_nodes = np.linspace(0, self.num_of_nodes - 1, self.num_of_nodes, dtype=np.uint16)

    def __initialise_pheromones(self):
        """
        Initializes pheromone levels on all edges.
        
        Sets all pheromone values to TAU_0.
        """
        self.pheromones = np.ones(shape=self.distances.shape) * TAU_0

    def __initialise_heuristic(self):
        """
        Computes heuristic values based on distances between nodes.
        
        Heuristic is the inverse of distance, favoring shorter paths.
        """
        # normalise = self.distances / np.max(self.distances)
        # self.heuristic = 1 / (normalise + 1e-12)
        self.heuristic = 1 / (self.distances + 1e-12)

    def __all_unnormalised_posteriors(self):
        """
        Computes unnormalized posterior probabilities for all node transitions.
        
        Combines pheromone and heuristic information using ALPHA and BETA exponents.
        """
        pheromones = np.power(self.pheromones, ALPHA)
        heuristic = np.power(self.heuristic, BETA)
        self.unnormalised_posteriors = np.multiply(pheromones, heuristic)

    def __calculate_unvisited_probabilities(self, current_node_index):
        """
        Calculates transition probabilities to unvisited nodes from current node.
        
        Args:
            current_node_index: Index of the current node.
            
        Returns:
            ndarray: Normalized probabilities for each unvisited node.
        """
        numerator = self.unnormalised_posteriors[current_node_index, self.unvisited_nodes]
        denominator = self.unnormalised_posteriors[current_node_index, self.unvisited_nodes]
        denominator = np.sum(denominator)

        return numerator / denominator
    
    def __choose_next_node(self, probabilities):
        """
        Selects the next node probabilistically based on transition probabilities.
        
        Removes the selected node from the unvisited nodes list.
        
        Args:
            probabilities: Normalized probabilities for each unvisited node.
            
        Returns:
            int: Index of the selected next node.
        """
        next_node = np.random.choice(self.unvisited_nodes, p=probabilities)
        self.unvisited_nodes = self.unvisited_nodes[self.unvisited_nodes != next_node]
        return next_node
    
    def __update_path(self, current, next, ant_num, node_num):
        """
        Records a node transition in an ant's path.
        
        Args:
            current: Index of the current node.
            next: Index of the next node.
            ant_num: Index of the ant.
            node_num: Position in the ant's path.
        """
        self.ant_paths[ant_num, node_num, :] = [int(current), int(next)]
    
    def __update_pheromones(self):
        """
        Updates pheromone levels based on ant paths and their quality.
        
        Evaporates pheromones globally and deposits new pheromones proportional to path quality.
        Shorter paths deposit more pheromones.
        
        Returns:
            tuple: (shortest_path, shortest_distance) from this iteration.
        """
        # Get the distances between each 2 nodes in the path
        path_lengths = self.distances[self.ant_paths[:, :, 0], self.ant_paths[:, :, 1]]
        path_lengths = np.sum(path_lengths, axis=1)  # Sum those distances to get the length of the path for each ant
        delta_tau = (1 / path_lengths) * Q  # Shorter paths get better updates (more pheromones deposited)
        self.pheromones = self.pheromones * (1 - RHO)  # Evaporate the pheromones
        for i in range(len(self.ant_paths)):
            self.pheromones[self.ant_paths[i, :, 0], self.ant_paths[i, :, 1]] += delta_tau[i]  # Add the deposited pheromones
            # The opposite elements also updated to keep the matrix symmetrical
            self.pheromones[self.ant_paths[i, :, 1], self.ant_paths[i, :, 0]] += delta_tau[i]

        shortest_path = self.ant_paths[np.argmin(path_lengths)]
        shortest_distance = np.min(path_lengths)
        return shortest_path, shortest_distance
    
    def ant_colony(self, starting_node, terminate_no_progress_threshold=15):
        """
        Runs the Ant Colony Optimization algorithm.
        
        Iteratively constructs paths and updates pheromones until convergence or max iterations reached.
        Terminates early if no improvement is found for terminate_no_progress_threshold iterations.
        
        Args:
            starting_node: Index of the starting node for all ants.
            terminate_no_progress_threshold: Number of iterations without improvement before termination (default: 15).
            
        Returns:
            tuple: (shortest_path, shortest_distance) found during optimization.
        """
        shorterst_path = None
        shorterst_distance = np.inf
        stop_counter = 0
        for _ in range(ITERATIONS):
            for ant in range(ANT_POPULATION):
                current_node = starting_node
                self.unvisited_nodes = self.unvisited_nodes[self.unvisited_nodes != current_node]
                for node in range(self.num_of_nodes - 1):
                    probabilities = self.__calculate_unvisited_probabilities(current_node)
                    next_node = self.__choose_next_node(probabilities)
                    self.__update_path(current_node, next_node, ant, node)
                    current_node = next_node
                self.__initialise_unvisited_nodes()

            path, path_length = self.__update_pheromones()
            if path_length < shorterst_distance:
                shorterst_path = path
                shorterst_distance = path_length
                stop_counter = 0
            else:
                stop_counter += 1
                if stop_counter == terminate_no_progress_threshold:
                    return shorterst_path, shorterst_distance
                
            self.__all_unnormalised_posteriors()
            self.__initialise_paths()

        return shorterst_path, shorterst_distance
    
    def greedy_nearest_neigbour(self, starting_node):
        """
        Solves the TSP using a greedy nearest neighbor heuristic.
        
        Always selects the nearest unvisited node based on heuristic values.
        Provides a baseline comparison for ant colony optimization.
        
        Args:
            starting_node: Index of the starting node.
            
        Returns:
            tuple: (path, distance) representing the greedy solution.
        """
        starting_node = int(starting_node)
        path = np.zeros(self.num_of_nodes).astype(np.int32)
        path[0] = starting_node
        self.__initialise_unvisited_nodes()
        self.unvisited_nodes = self.unvisited_nodes[self.unvisited_nodes != starting_node]

        distance = 0
        counter = 0
        while counter < self.num_of_nodes - 1:
            counter += 1
            heuristic = self.heuristic[starting_node, self.unvisited_nodes]
            index = np.argmax(heuristic)
            next_node = self.unvisited_nodes[index]
            distance += self.distances[starting_node, next_node]
            path[counter] = int(next_node)
            self.unvisited_nodes = self.unvisited_nodes[self.unvisited_nodes != next_node]
            starting_node = next_node

        return path, distance

def pre_process_data(mask, coordinates, size=30):
    """
    Preprocesses image data and coordinates for ant colony optimization.
    
    Applies average pooling to the mask and redefines coordinate values to match pooled resolution.
    Extracts nodes corresponding to positive mask regions and computes pairwise distances.
    
    Args:
        mask: A 2D binary numpy array where positive values indicate valid regions.
        coordinates: A 2D or 3D numpy array of coordinate values matching mask dimensions.
        size: Target number of clusters (default: 30).
        
    Returns:
        tuple: (nodes, distances, filtered_mask) where nodes are valid coordinates, 
               distances is the nxn pairwise distance matrix, and filtered_mask is the pooled mask.
    """
    mask[mask > 0] = 1
    total_blood_pixels = np.sum(mask)
    kernel_size = np.ceil(np.sqrt(total_blood_pixels / size))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel_size = int(kernel_size)

    # print(f"Kernel Size:\n {kernel_size}\n\n")
    # print(f"Mask:\n {mask}\n\n")
    # print(f"Coordinates:\n {coordinates}\n\n")
    avg_pooled_image = avg_pool(mask, kernel_size, threshold=0.2)
    # print(f"Mask Filtered:\n {avg_pooled_image}\n\n")
    abstracted_coord = redefine_values(coordinates, kernel_size, dimensions=2)
    # print(f"Coordinates Filtered:\n {abstracted_coord}\n\n")

    flattened_avg_pooled_image = avg_pooled_image.flatten()
    flattened_abstracted_coord = abstracted_coord.reshape(-1, 2)
    nodes = flattened_abstracted_coord[flattened_avg_pooled_image == 1]

    # Compute pairwise Euclidean distances between all nodes
    d = []
    for node in nodes:
        dist = np.linalg.norm(nodes - node, axis=1)
        d.append(dist)
    d = np.array(d)
    
    return nodes, d, avg_pooled_image


if __name__ == "__main__":
    import cv2
    import matplotlib.pyplot as plt

    dataset_location = '/Location to Testing Data/'
    for i in range(1, 9):
        file_name = f'test_data_{i}.png'
        image = cv2.imread(dataset_location + file_name, 0)
        image[image > 0] = 2
        image[image == 0] = 1
        image[image == 2] = 0
        size = image.shape
        row_indexes = np.linspace(0, size[0] - 1, size[0])
        col_indexes = np.linspace(0, size[1] - 1, size[1])
        coordinates = np.ones((size[0], size[1], 2))
        coordinates[:, :, 0] *= row_indexes[:, np.newaxis]
        coordinates[:, :, 1] *= col_indexes[np.newaxis, :]

        total_blood_pixels = np.sum(image)
        kernel_size = np.ceil(np.sqrt(total_blood_pixels / 30))
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel_size = int(kernel_size)
        print(total_blood_pixels, kernel_size)

        #kernel_size = 35
        avg_pooled_image = avg_pool(image, kernel_size)
        abstracted_coord = redefine_values(coordinates, kernel_size, 2)

        # print(avg_pooled_image)
        # print(abstracted_coord)
        flattened_avg_pooled_image = avg_pooled_image.flatten()
        flattened_abstracted_coord = abstracted_coord.reshape(-1, 2)
        nodes = flattened_abstracted_coord[flattened_avg_pooled_image == 1]

        d = []
        for node in nodes:
            dist = np.linalg.norm(nodes - node, axis=1)
            d.append(dist)
        d = np.array(d)
        
        # Ant Colony Optimisation
        model = AntColony(nodes, d)
        shortest_path, shortest_dist = model.ant_colony(0)
        print("Path:\n", shortest_path)
        print("Distance =", shortest_dist)

        node_list = shortest_path[:, 1]
        node_list = np.concatenate(([0], node_list))
        path = nodes[node_list]

        plt.imshow(image)
        plt.plot(path[:, 1], path[:, 0])
        plt.show()

        # Greedy Nearest Neighbour
        model = AntColony(nodes, d)
        shortest_path, shortest_dist = model.greedy_nearest_neigbour(0)
        print("Path:\n", shortest_path)
        print("Distance =", shortest_dist)

        node_list = np.concatenate(([0], shortest_path))
        path = nodes[node_list]

        plt.imshow(image)
        plt.plot(path[:, 1], path[:, 0])
        plt.show()


    # node1 = np.array([0, 0, 0])
    # node2 = np.array([1, 2, 0])
    # node3 = np.array([2, 5, 0])
    # node4 = np.array([4, 3, 0])
    # node5 = np.array([4, 1, 0])
    # nodes = np.array([node1, node2, node3, node4, node5])
    # d = []
    # for node in nodes:
    #     dist = np.linalg.norm(nodes - node, axis=1)
    #     d.append(dist)
    # d = np.array(d)
