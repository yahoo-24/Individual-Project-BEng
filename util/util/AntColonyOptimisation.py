import numpy as np
from .Abstraction import max_pool, redefine_values, avg_pool

TAU_0 = 2.0 # 2
ALPHA = 2.0 # 2
BETA = 2 # 2
Q = 5 # 5
RHO = 0.08 # 0.08
ANT_POPULATION = 100 # 100
ITERATIONS = 100 # 100

class AntColony():
    def __init__(self, nodes, distances):
        self.nodes = nodes
        self.distances = distances
        self.num_of_nodes = len(nodes)
        self.__initialise_unvisited_nodes()

        self.__initialise_heuristic()
        self.__initialise_pheromones()
        self.__initialise_paths()
        self.__all_unnormalised_posteriors()

    def __initialise_paths(self):
        self.ant_paths = np.zeros((ANT_POPULATION, self.num_of_nodes - 1, 2), dtype=np.uint16)

    def __initialise_unvisited_nodes(self):
        self.unvisited_nodes = np.linspace(0, self.num_of_nodes - 1, self.num_of_nodes, dtype=np.uint16)

    def __initialise_pheromones(self):
        self.pheromones = np.ones(shape=self.distances.shape) * TAU_0

    def __initialise_heuristic(self):
        # normalise = self.distances / np.max(self.distances)
        # self.heuristic = 1 / (normalise + 1e-12)
        self.heuristic = 1 / (self.distances + 1e-12)

    def __all_unnormalised_posteriors(self):
        pheromones = np.power(self.pheromones, ALPHA)
        heuristic = np.power(self.heuristic, BETA)
        self.unnormalised_posteriors = np.multiply(pheromones, heuristic)

    def __calculate_unvisited_probabilities(self, current_node_index):
        numerator = self.unnormalised_posteriors[current_node_index, self.unvisited_nodes]
        denominator = self.unnormalised_posteriors[current_node_index, self.unvisited_nodes]
        denominator = np.sum(denominator)

        return numerator / denominator
    
    def __choose_next_node(self, probabilities):
        next_node = np.random.choice(self.unvisited_nodes, p=probabilities)
        self.unvisited_nodes = self.unvisited_nodes[self.unvisited_nodes != next_node]
        return next_node
    
    def __update_path(self, current, next, ant_num, node_num):
        self.ant_paths[ant_num, node_num, :] = [int(current), int(next)]
    
    def __update_pheromones(self):
        # Get the distances between each 2 nodes in the path
        path_lengths = self.distances[self.ant_paths[:, :, 0], self.ant_paths[:, :, 1]]
        path_lengths = np.sum(path_lengths, axis=1) # Sum those distances to get the length of the path for each ant
        delta_tau = (1 / path_lengths) * Q # Shorter paths get better updates (more pheromones deposited)
        self.pheromones = self.pheromones * (1 - RHO) # Evaporate the pheromones
        for i in range(len(self.ant_paths)):
            self.pheromones[self.ant_paths[i, :, 0], self.ant_paths[i, :, 1]] += delta_tau[i] # Add the deposited pheromones
            # The opposite elements also updated to keep the matrix symmetrical
            self.pheromones[self.ant_paths[i, :, 1], self.ant_paths[i, :, 0]] += delta_tau[i]

        shortest_path = self.ant_paths[np.argmin(path_lengths)]
        shortest_distance = np.min(path_lengths)
        return shortest_path, shortest_distance
    
    def ant_colony(self, starting_node, terminate_no_progress_threshold=15):
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

    d = []
    for node in nodes:
        dist = np.linalg.norm(nodes - node, axis=1)
        d.append(dist)
    d = np.array(d)
    
    return nodes, d, avg_pooled_image


if __name__ == "__main__":
    import cv2
    import matplotlib.pyplot as plt

    dataset_location = '/home/yahia/Downloads/Dataset/Testing Data/'
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

        # Greed Nearest Neighbour
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