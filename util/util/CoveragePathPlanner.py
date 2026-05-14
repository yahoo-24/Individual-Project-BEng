import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt

class PathPlanner():
    """
    Path planning utility for generating coverage trajectories over
    segmented blood masks.

    Supported planning algorithms:
    --------------------------------
    - SPIRAL : Generates an inward/outward spiral trajectory.
    - SWEEP  : Generates a zigzag raster sweep trajectory.

    Parameters
    ----------
    mask : numpy.ndarray
        Binary mask representing valid target regions.

    algorithm : str
        Name of the planning algorithm to use.

    depths : numpy.ndarray
        Coordinate/depth map associated with the mask.

    res : int, optional
        Angular resolution used for spiral planning.
    """
    
    def __init__(self, mask, algorithm, depths, res=50):
        self.mask = mask
        self.algorithm = algorithm
        self.depths = depths
        self.res = res


    def generate_path(self):
        """
        Generate a trajectory using the selected planning algorithm.

        Returns
        -------
        numpy.ndarray
            Planned trajectory points.
        """
        if self.algorithm.upper() == "SPIRAL":
            return self.spiral_path(self.res)
        else:
            return self.sweep_planner()


    def find_targets(self):
        """
        Extract active target pixel locations from the mask.

        Returns
        -------
        numpy.ndarray
            Array of target pixel indices with shape (n, 2).
        """
        targets = np.array(self.mask)
        points = np.where(targets > 0)
        points = np.array([points[0], points[1]]).T
        return points
    
    def check_empty(self, mask):
        """
        Check whether a mask contains any active pixels.

        Parameters
        ----------
        mask : numpy.ndarray
            Binary mask to evaluate.

        Returns
        -------
        bool
            True if the mask is empty, otherwise False.
        """
        if np.sum(mask) == 0:
            return True
        return False

    def frame_direction(self):
        """
        Estimate the principal direction of the target region.

        Principal Component Analysis (PCA) is used to determine the
        dominant orientation of the segmented region. The mask is then
        rotated to align the axis that is the second component (opposite
        of the principal axis) horizontally for more
        efficient sweep planning.

        Returns
        -------
        tuple
            rotated_matrix_same_shape : numpy.ndarray
                Rotated mask with original dimensions preserved.

            best_angle : float
                Rotation angle in degrees.

            eigvec : numpy.ndarray
                Eigenvectors from PCA analysis.
        """
        points = self.find_targets()
        if len(points) == 0:
            return None, None, None
        elif len(points) <= 10:
            # 10 points are not significant to perform all the calculations below
            return self.mask, 0, None
        mean = points.mean(axis=0)
        centered_points = points - mean
        covariance = np.cov(centered_points.T)
        if np.inf in covariance or np.NaN in covariance:
            return None, None, None
        result = np.linalg.eig(covariance)

        #eigval, eigvec = result.eigenvalues, result.eigenvectors
        eigval, eigvec = result
        if eigval[0] > eigval[1]:
            principal = 1
        else:
            principal = 0
        angle_of_rot = -math.atan2(eigvec[principal][1], eigvec[principal][0]) * 180 / np.pi

        best_angle = angle_of_rot
        
        rotated_matrix_same_shape = ndimage.rotate(self.mask, best_angle, reshape=False, order=0)
        rotated_matrix_full = ndimage.rotate(self.mask, best_angle, reshape=True)

        if self.check_empty(rotated_matrix_same_shape):
            print("Empty")
            return self.mask, 0, None

        return rotated_matrix_same_shape, best_angle, eigvec


    def close_points_removal_sweep(self, points, target, threshold=0.4):
        """
        Prevent closely spaced sweep points from being duplicated.

        Parameters
        ----------
        points : list
            Existing path points.

        target : numpy.ndarray
            Candidate point to add.

        threshold : float
            Minimum allowable distance between points.

        Returns
        -------
        bool
            True if the target should be added.
        """
        if len(points) == 0:
            return True
        np_points = np.array(points)
        difference = np_points - target
        distances = np.linalg.norm(difference, axis=1)
        return np.all(distances > threshold)


    def sweep(self, frame, frame_shape, angle):
        """
        Generate a raster sweep trajectory over a rotated mask.

        The function traverses the mask row-by-row in a zigzag
        pattern and maps rotated coordinates back to the original
        image frame.

        Parameters
        ----------
        frame : numpy.ndarray
            Rotated binary mask.

        frame_shape : tuple
            Shape of the original frame.

        angle : float
            Rotation angle used to align the frame.

        Returns
        -------
        list
            List of planned Cartesian path points.
        """
        points = []
        center_y, center_x = frame.shape
        center_x = (center_x - 1) / 2
        center_y = (center_y - 1) / 2
        for i, arr in enumerate(frame):
            if i % 2 == 1:
                m = len(arr) - 1
                arr = arr[::-1]
            else:
                m = 0
            for j, val in enumerate(arr):
                if val == 1:
                    point = [i - center_y, abs(m - j) - center_x]
                    original_point = self.map_back_to_original(point, angle, frame_shape)
                    original_point = self.depths[original_point[0]][original_point[1]]
                    if self.close_points_removal_sweep(points, np.array(original_point)):
                        points.append(original_point)

        return points


    def map_back_to_original(self, rotated_point, angle, frame_shape):
        """
        Convert a rotated coordinate back into the original frame.

        Parameters
        ----------
        rotated_point : list
            Point in rotated coordinates.

        angle : float
            Rotation angle in degrees.

        frame_shape : tuple
            Shape of the original image.

        Returns
        -------
        list
            Original image indices [row, column].
        """
        # Center of the original matrix
        height, width = frame_shape
        y_center = (width - 1) / 2
        x_center = (height - 1) / 2
        x_prime = rotated_point[0]
        y_prime = rotated_point[1]

        # Convert theta from degrees to radians
        angle_rad = np.radians(angle)

        # Inverse rotation transformation
        x_original = x_center + (np.cos(angle_rad) * x_prime + np.sin(angle_rad) * y_prime)
        y_original = y_center + (-np.sin(angle_rad) * x_prime + np.cos(angle_rad) * y_prime)
        original = [int(round(x_original)), int(round(y_original))]

        return original  # Return row and column indices


    def sweep_planner(self):
        """
        Generate a complete sweep-based coverage path.

        Returns
        -------
        numpy.ndarray
            Planned sweep trajectory.
        """
        self.mask[self.mask > 0] = 255
        matrix, angle, _ = self.frame_direction()
        if matrix is None:
            return None
        matrix = self.adjust_values(matrix)
        points = self.sweep(matrix, matrix.shape, angle)
        return np.array(points)


    def adjust_values(self, frame):
        """
        Convert mask values to binary format.

        Parameters
        ----------
        frame : numpy.ndarray
            Input image mask.

        Returns
        -------
        numpy.ndarray
            Binary mask.
        """
        frame[frame > 0] = 1
        return frame


    def find_depth_targets(self):
        """
        Extract depth coordinates corresponding to active mask pixels.

        Returns
        -------
        numpy.ndarray
            Target depth coordinates.
        """
        condition = self.mask > 0
        return self.depths[condition]

    def locate_mass_center(self, depths):
        """
        Estimate the centre of mass of the target region.

        Parameters
        ----------
        depths : numpy.ndarray
            Target coordinates.

        Returns
        -------
        numpy.ndarray
            Median centre location.
        """
        median = np.median(depths, axis=0)
        median[2] = 0.0 # z-axis irrelevant for planning
        return median


    def close_points_removal_spiral(self, points, target, threshold=0.4):
        """
        Remove closely spaced spiral radii.

        Parameters
        ----------
        points : list
            Existing radii.

        target : float
            Candidate radius value.

        threshold : float
            Minimum allowed spacing.

        Returns
        -------
        bool
            True if radius should be added.
        """
        np_points = np.array(points)
        distances = np.abs(np_points - target)
        return np.all(distances > threshold)

    def check_radius_closeness(self, radius_array, threshold=0.4):
        """
        Remove radii that are too similar.

        Parameters
        ----------
        radius_array : numpy.ndarray
            Radius-angle pairs.

        threshold : float
            Minimum radius separation.

        Returns
        -------
        numpy.ndarray
            Filtered radius array.
        """
        if len(radius_array) == 0:
            return np.array([])
        
        radius_array = radius_array[::-1]
        index = 0
        while index < len(radius_array) - 1:
            distances = np.abs(radius_array[:, 0] - radius_array[index, 0])
            condition = distances > threshold
            condition[index] = True
            radius_array = radius_array[condition]
            index += 1

        return radius_array[::-1]


    def convert_to_polar_coord(self, distances, resolution=24):
        """
        Convert Cartesian coordinates into polar sectors.

        Parameters
        ----------
        distances : numpy.ndarray
            Cartesian coordinates relative to centre.

        resolution : int
            Number of angular sectors.

        Returns
        -------
        tuple
            mappings : dict
                Polar coordinate groupings.

            lengths : numpy.ndarray
                Number of points in each angular sector.
        """
        mappings = dict()
        lengths = []
        sections = np.array([i * np.pi * 2 / resolution for i in range(resolution)])
        for section in sections:
            mappings[section] = []
        for distance in distances:
            radius = np.linalg.norm(distance[:2])  # Euclidean Distance to center
            if distance[0] != 0:
                theta = np.arctan(distance[1] / distance[0])
                if distance[0] < 0:
                    theta += np.pi
                if theta < 0:
                    theta += 2 * np.pi  # To avoid negative values
            else:
                # Prevents division by 0
                if distance[1] > 0:
                    theta = np.pi / 2
                else:
                    theta = np.pi * 3 / 2
            differences = theta - sections  # Subtract to find the closest section to theta
            section_index = np.argmin(np.abs(differences))  # Find the smallest value (abs removes negatives) argmin returns index
            closest_section = sections[section_index]  # Get the closest section
            # if self.close_points_removal_spiral(mappings[closest_section], radius):
            #     mappings[closest_section].append(radius)
            mappings[closest_section].append([radius, distance[2]])
        
        for section in sections:
            if len(mappings[section]) == 0:
                lengths.append(0)
                continue
            sec = np.array(mappings[section])
            sec = sec[sec[:, 0].argsort()]
            mappings[section] = self.check_radius_closeness(sec).tolist()
            #mappings[section] = np.sort(mappings[section]).tolist()
            lengths.append(len(mappings[section]))

        return mappings, np.array(lengths)


    def polar_to_cartesian(self, center, points):
        """
        Convert polar coordinates back into Cartesian coordinates.

        Parameters
        ----------
        center : numpy.ndarray
            Centre offset.

        points : list
            Polar points [radius, theta].

        Returns
        -------
        numpy.ndarray
            Cartesian coordinates.
        """
        points = np.array(points).T
        x, y = points[0] * np.cos(points[1]), points[0] * np.sin(points[1])
        y += center[1]
        x += center[0]
        return np.array([x, y])


    def check_closeness(self, points, threshold=0.04):
        """
        Remove spatially redundant points from a path.

        Parameters
        ----------
        points : numpy.ndarray
            Planned path points.

        threshold : float
            Minimum allowed spacing.

        Returns
        -------
        numpy.ndarray
            Filtered path.
        """
        if len(points) == 0:
            return np.array([])
        
        index = 0
        while index < len(points) - 1:
            distances = np.linalg.norm(points - points[index], axis=1)
            condition_1 = distances == 0
            condition_2 = distances > threshold
            points = points[np.add(condition_1, condition_2)]
            index += 1

        return np.array(points)


    def spiral_path(self):
        """
        Generate a spiral coverage trajectory.

        The algorithm:
        ----------------
        1. Finds the centre of mass.
        2. Converts target points into polar coordinates.
        3. Traverses angular sectors progressively.
        4. Converts the final trajectory back to Cartesian space.

        Returns
        -------
        numpy.ndarray
            Spiral trajectory points.
        """
        if np.all(self.mask == 0):
            return []
        depths = self.find_depth_targets()  # Extract ones [n x 2] numpy array

        center_of_mass = self.locate_mass_center(depths)  # Tuple of numpy array of mass center location
        distance_matrix = depths - center_of_mass  # [n x 2] numpy array of distances from center of mass
        polar, lengths = self.convert_to_polar_coord(distance_matrix, self.res)  # A dictionary of all the positions converted to polar

        points = []
        z = []
        while not np.all(lengths == 0):
            for index, theta in enumerate(polar):
                if lengths[index] != 0:
                    value = polar[theta].pop() # [r, z]
                    point = [value[0], theta]  # Add [r, theta]
                    z.append(value[1])
                    points.append(point)
                    lengths[index] -= 1

        points = self.polar_to_cartesian(center_of_mass, points)
        points = np.array([-points[1], -points[0], z]).T
        points = self.check_closeness(points)

        return points


def plot(points):
    # Separate the points into X and Y coordinates
    x_points = [point[1] for point in points]
    y_points = [point[0] for point in points]

    # Create the plot
    #plt.figure(figsize=(8, 6))
    plt.plot(x_points, y_points, marker='o', linestyle='-', color='b')  # Line with points

    # Adding titles and labels
    plt.title('Line Plot Between Points')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')

    # Show the plot
    #plt.show()

if __name__ == "__main__":
    n = 100
    depths = np.linspace(1, 10, n)
    depths = np.array([depths, depths, np.zeros(len(depths))]).T
    targets = np.random.random(n)
    targets[targets >= 0.5] = 1
    targets[targets < 0.5] = 0

    print(f"The targets are:\n{targets.reshape(10, 10)}\n\n")
    print(f"The depths are:\n{depths.reshape(10, 10, 3)}\n\n")

    targets = targets.reshape(10, 10)
    depths = depths.reshape(10, 10, 3)

    planner = PathPlanner(
        targets=targets,
        algorithm="SPIRAL",
        depths=depths
    )

    spiral_result = planner.generate_path()
    print("The generated path looks like this:\n", spiral_result, '\n\n')

    planner = PathPlanner(
        targets=targets,
        algorithm="SWEEP",
        depths=depths
    )

    sweep_result = planner.generate_path()
    print("The generated path looks like this:\n", sweep_result, '\n\n')

    print("The depths of the targets are:\n", depths[targets > 0])

    expected = depths[targets > 0]
    deviations = []
    for arr in expected:
        distances = np.linalg.norm(spiral_result - arr, axis=1)
        deviations.append(np.min(distances))

    print(f"The largest deviation was {np.max(deviations)}\nThe total deviation is {np.sum(deviations)}")
    print("Total elements are", len(expected))
