import numpy as np

class ArtificialPotentialField():
    def __init__(self, targets, obstacles, att_gain, rep_gain, max_distance, mini_batches=False):
        """
        Initializes an Artificial Potential Field (APF) for path planning and force calculations.
        
        The APF method uses attractive forces from target points and repulsive forces from 
        obstacles to compute resultant forces that guide movement toward targets while 
        avoiding obstacles.
        
        Args:
            targets: An nx3 numpy array of n target positions with x, y, z coordinates.
            obstacles: An mx3 numpy array of m obstacle positions with x, y, z coordinates.
            att_gain: A positive multiplier that increases the strength of attractive forces.
            rep_gain: A positive multiplier that increases the strength of repulsive forces.
            max_distance: The maximum distance from an obstacle where repulsion force acts.
            mini_batches: If True, uses random sampling of targets for faster computation (default: False).
        """
        self.targets = targets
        self._update_len_targets()
        self.obstacles = obstacles
        self._update_len_obstacles()
        # Ensure obstacles count is at least 1 to avoid division by zero
        if obstacles is not None:
            self.len_obstacles = len(obstacles)
        else:
            self.len_obstacles = 1
        # Use absolute values to ensure gains are positive
        self.att_gain = abs(att_gain)
        self.rep_gain = abs(rep_gain)
        self.maximum_distance = abs(max_distance)
        self.mini_batches = mini_batches

    def _update_len_targets(self):
        """
        Updates the count of target points.
        
        Ensures the target count is at least 1 to avoid division by zero errors.
        """
        if self.targets is not None:
            self.len_targets = len(self.targets)
        else:
            self.len_targets = 1

    def _update_len_obstacles(self):
        """
        Updates the count of obstacle points.
        
        Ensures the obstacle count is at least 1 to avoid division by zero errors.
        """
        if self.obstacles is not None:
            self.len_obstacles = len(self.obstacles)
        else:
            self.len_obstacles = 1

    def attraction(self, position):
        """
        Calculates the attractive force pulling the current position toward target points.
        
        The force is proportional to the displacement vectors from the current position 
        to each target, scaled by the attraction gain. When mini_batches is enabled, 
        only a random subset of targets is used for computational efficiency.
        
        Args:
            position: A 1x3 numpy array representing the current x, y, z position.
            
        Returns:
            A 1x3 numpy array representing the resultant attractive force vector.
            Returns 0 if no targets are defined.
        """
        if self.targets is None:
            return 0
        
        # Use random sampling of targets for faster computation if mini_batches is enabled
        if self.mini_batches:
            size = int(self.len_targets / 100)
            choices = np.random.randint(0, self.len_targets, size=size)
            targets = self.targets[choices]
        else:
            targets = self.targets
        
        # Calculate displacement vectors from position to each target (nx3 array)
        differences = targets - position
        # Apply attraction gain and sum all forces
        att_force = self.att_gain * differences
        att_force = np.sum(att_force, axis=0)

        return att_force

    def repulsion(self, position):
        """
        Calculates the repulsive force pushing the current position away from obstacle points.
        
        The repulsive force only acts on obstacles within the maximum distance threshold. 
        The force magnitude increases as the position gets closer to obstacles, following 
        an inverse distance relationship.
        
        Args:
            position: A 1x3 numpy array representing the current x, y, z position.
            
        Returns:
            A 1x3 numpy array representing the resultant repulsive force vector.
            Returns 0 if no obstacles are defined.
        """
        if self.obstacles is None:
            return 0
        
        # Calculate displacement vectors from each obstacle to position (mx3 array)
        differences = position - self.obstacles
        # Compute Euclidean distances from position to each obstacle
        distances = np.linalg.norm(differences, axis=1)

        # Filter to only include obstacles within the maximum distance threshold
        remain = distances < self.maximum_distance
        differences = differences[remain]
        distances = distances[remain]
        
        # Normalize displacement vectors to unit vectors (add small epsilon to avoid division by zero)
        unit_vector_diff = (differences.T / (distances + 1e-12)).T
        # Calculate repulsion magnitude: inverse of distance minus inverse of max distance
        d = (1 / (distances + 1e-12)) - (1 / (self.maximum_distance + 1e-12))
        # Calculate repulsion force for each obstacle and sum
        rep_force = self.rep_gain * (unit_vector_diff.T * d).T
        rep_force = np.sum(rep_force, axis=0)

        return rep_force

    def resultant_force(self, position, targets=None, obstacles=None):
        """
        Calculates the combined attractive and repulsive forces at a given position.
        
        Returns the normalized sum of attraction and repulsion forces, accounting for 
        the number of targets and obstacles to provide consistent force magnitudes 
        regardless of the number of targets/obstacles.
        
        Args:
            position: A 1x3 numpy array representing the current x, y, z position.
            targets: Optional. An nx3 numpy array of target positions to override stored targets.
            obstacles: Optional. An mx3 numpy array of obstacle positions to override stored obstacles.
            
        Returns:
            A 1x3 numpy array representing the normalized resultant force vector.
        """
        # Update targets and obstacles if provided
        if targets is not None:
            self.targets = targets
        if obstacles is not None:
            self.obstacles = obstacles
        
        # Calculate individual force components
        attraction_force = self.attraction(position)
        repulsion_force = self.repulsion(position)

        # Update counts in case targets/obstacles were changed
        self._update_len_obstacles()
        self._update_len_targets()

        # Normalize forces by the number of targets and obstacles (add epsilon to avoid division by zero)
        return attraction_force / (self.len_targets + 1e-12) + repulsion_force / (self.len_obstacles + 1e-12)

    
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Create sample target points along a line from (0.1, 0.1, 0) to (1.0, 1.0, 0)
    targets_x = np.linspace(1, 10, 10) / 10
    targets_y = targets_x
    targets_z = np.zeros(10)
    targets = np.array([targets_x, targets_y, targets_z]).T
    
    # Create obstacles slightly below the target points (z = -0.05)
    obstacles = targets.copy()
    obstacles[:, 2] -= 0.05

    # Initialize the APF model
    model = ArtificialPotentialField(
        targets=targets,
        obstacles=obstacles,
        att_gain=0.01,
        rep_gain=0.01,
        max_distance=0.15
    )

    # Start simulation from the origin
    position = np.array([0.0, 0.0, 0.0])

    # Iteratively move the position by applying the resultant force until convergence
    # Convergence is determined when the position change falls below a threshold
    previous_position = np.array([-1, -1, -1])
    counter = 1
    while np.linalg.norm(np.abs(position - previous_position)) > 0.001:
        previous_position = position.copy()
        force = model.resultant_force(position)
        position += force
        print(f"At time {counter}:\n Force is in the direction: {force} \nNew position is now at {position}\n")
        counter += 1
