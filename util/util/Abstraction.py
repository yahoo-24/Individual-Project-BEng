from sklearn.cluster import KMeans
import numpy as np

def abstract(obstacles, size=50):
    """
    Clusters a set of 3D obstacle points and returns cluster labels and centroids.
    
    Uses K-means clustering to partition obstacles into a specified number of clusters,
    which can be useful for spatial abstraction and reducing data complexity.
    
    Args:
        obstacles: An nx3 numpy array where n is the number of obstacles and the 3 columns represent x, y, z coordinates.
        size: The number of clusters to create (default: 50).
        
    Returns:
        tuple: (labels, centres) where labels are cluster assignments for each obstacle and centres are the cluster centroids.
    """

    # Initialize and fit K-means model with k-means++ initialization for better starting positions
    model = KMeans(size, init='k-means++')
    model.fit(obstacles)
    # Extract cluster assignment labels for each point
    labels = model.labels_
    # Extract the coordinates of each cluster centroid
    centres = model.cluster_centers_
    return labels, centres

def abstract_diff(point, obstacles, size=50):
    """
    Clusters obstacles relative to a reference point and returns cluster labels and centroids.
    
    Similar to abstract(), but translates all obstacles relative to a reference point before clustering.
    This is useful for analyzing obstacle distributions relative to a specific location.
    
    Args:
        point: A 3-element array representing the reference point (x, y, z).
        obstacles: An nx3 numpy array where n is the number of obstacles and the 3 columns represent x, y, z coordinates.
        size: The number of clusters to create (default: 50).
        
    Returns:
        tuple: (labels, centres) where labels are cluster assignments and centres are the relative cluster centroids.
    """

    # Translate all obstacles relative to the reference point
    data = obstacles - point
    # Initialize and fit K-means model with k-means++ initialization
    model = KMeans(size, init='k-means++')
    model.fit(data)
    # Extract cluster assignment labels for each point
    labels = model.labels_
    # Extract the coordinates of each cluster centroid in relative space
    centres = model.cluster_centers_
    return labels, centres

def find_distances(point, centres):
    """
    Calculates Euclidean distances from a point to all cluster centroids.
    
    Args:
        point: A 3-element array representing the query point (x, y, z).
        centres: An mx3 numpy array where m is the number of cluster centroids.
        
    Returns:
        ndarray: A 1D numpy array of distances, or -1 if no centres are provided.
    """
    # Return error value if there are no cluster centres
    if len(centres) == 0:
        return -1
    # Calculate L2 norm (Euclidean distance) along the feature axis for all centres
    return np.linalg.norm(centres - point, axis=1)

def extract_closest(point, labels, centres, obstacles):
    """
    Extracts all obstacles belonging to the cluster closest to a given point.
    
    Args:
        point: A 3-element array representing the query point (x, y, z).
        labels: A 1D numpy array of cluster assignments for each obstacle.
        centres: An mx3 numpy array of cluster centroids.
        obstacles: The original nx3 numpy array of all obstacles.
        
    Returns:
        tuple: (closest_obstacles, closest_index) where closest_obstacles are those in the nearest cluster, or (-1, -1) if inputs are empty.
    """
    # Return error values if required inputs are empty
    if len(labels) == 0 or len(centres) == 0:
        return -1, -1
    
    # Calculate distances from the query point to all cluster centroids
    distances_to_centre = find_distances(point, centres)
    # Find the index of the closest cluster centroid
    index = np.argmin(distances_to_centre)

    # Return all obstacles assigned to the closest cluster and its index
    return obstacles[labels == index], index

def extract_closest_given_distances(labels, obstacles, distances_to_centre):
    """
    Extracts all obstacles from the closest cluster given pre-computed distances.
    
    Similar to extract_closest(), but accepts pre-computed distances as input,
    useful when distances have been modified or computed differently.
    
    Args:
        labels: A 1D numpy array of cluster assignments for each obstacle.
        obstacles: The original nx3 numpy array of all obstacles.
        distances_to_centre: A 1D numpy array of distances to each cluster centroid.
        
    Returns:
        tuple: (closest_obstacles, closest_index) where closest_obstacles are those in the nearest cluster, or (-1, -1) if labels are empty.
    """
    # Return error values if labels are empty
    if len(labels) == 0:
        return -1, -1
    
    # Find the index of the cluster with minimum distance
    index = np.argmin(distances_to_centre)

    # Return all obstacles assigned to the closest cluster and its index
    return obstacles[labels == index], index

def fill_obstacles(point, labels, centres, obstacles, n=1000):
    """
    Iteratively collects obstacles from the nearest clusters until reaching a target count.
    
    Starts with obstacles from the closest cluster, then progressively adds obstacles
    from the next nearest clusters until the specified number of obstacles is reached.
    
    Args:
        point: A 3-element array representing the query point (x, y, z).
        labels: A 1D numpy array of cluster assignments for each obstacle.
        centres: An mx3 numpy array of cluster centroids.
        obstacles: The original nx3 numpy array of all obstacles.
        n: Target number of obstacles to collect (default: 1000).
        
    Returns:
        list: A list of obstacles (as nested lists) from the nearest clusters, or -1, -1 if inputs are empty.
    """
    # Return error values if required inputs are empty
    if len(labels) == 0 or len(centres) == 0:
        return -1, -1
    
    # Initialize list to store collected obstacles
    obs = []
    # Calculate distances from the query point to all cluster centroids
    distances_to_centre = find_distances(point, centres)
    # Find the index of the closest cluster centroid
    index = np.argmin(distances_to_centre)
    # Add all obstacles from the closest cluster
    obs.extend(obstacles[labels == index].tolist())
    # Track the current number of collected obstacles
    length = len(obs)

    # Continue collecting from progressively farther clusters until reaching target count
    while length < n:
        # Mark the previous cluster as infinite distance to avoid revisiting it
        distances_to_centre[index] = np.inf
        # Extract obstacles from the next nearest cluster
        o, index = extract_closest_given_distances(labels, obstacles, distances_to_centre)
        # Add these obstacles to the collection
        obs.extend(o.tolist())
        # Update the total count
        length += len(o)

    return obs

def max_pool(frame, kernel_size):
    """
    Applies max pooling operation to a 2D frame.
    
    Divides the input frame into non-overlapping kernel_size x kernel_size blocks
    and replaces each block with its maximum value.
    
    Args:
        frame: A 2D numpy array representing the input frame.
        kernel_size: The size of the pooling kernel (kernel_size x kernel_size blocks).
        
    Returns:
        ndarray: A 2D numpy array of pooled values, or -1 if frame is smaller than kernel_size.
    """
    # Return error value if frame is smaller than kernel size
    if frame.shape[0] < kernel_size or frame.shape[1] < kernel_size:
        return -1
    
    # Calculate the dimensions of the output array
    result_size_col = np.floor(frame.shape[1] / kernel_size)
    result_size_row = np.floor(frame.shape[0] / kernel_size)
    # Initialize output array with zeros
    result_arr = np.zeros((result_size_row.astype(np.int32), result_size_col.astype(np.int32)))
    
    # Iterate over rows
    row_index = 0
    while row_index < result_size_row:
        # Calculate the row bounds of the current block
        row_start = row_index * kernel_size
        row_end = row_start + kernel_size
        # Iterate over columns
        col_index = 0
        while col_index < result_size_col:
            # Calculate the column bounds of the current block
            col_start = col_index * kernel_size
            col_end = col_start + kernel_size
            # Store the maximum value from this block in the output array
            result_arr[row_index, col_index] = np.max(frame[row_start:row_end, col_start:col_end])
            col_index += 1
        row_index += 1

    return result_arr

def avg_pool(frame, kernel_size, threshold=0.75):
    """
    Applies average pooling followed by thresholding to a 2D frame.
    
    Divides the input frame into non-overlapping kernel_size x kernel_size blocks,
    computes the average of each block, and then binarizes based on a threshold.
    
    Args:
        frame: A 2D numpy array representing the input frame.
        kernel_size: The size of the pooling kernel (kernel_size x kernel_size blocks).
        threshold: The threshold value for binarization (default: 0.75); values below become 0, values >= threshold become 1.
        
    Returns:
        ndarray: A 2D binary numpy array (0s and 1s), or -1 if frame is smaller than kernel_size.
    """
    # Return error value if frame is smaller than kernel size
    if frame.shape[0] < kernel_size or frame.shape[1] < kernel_size:
        return -1
    
    # Calculate the dimensions of the output array
    result_size_col = np.floor(frame.shape[1] / kernel_size)
    result_size_row = np.floor(frame.shape[0] / kernel_size)
    # Initialize output array with zeros
    result_arr = np.zeros((result_size_row.astype(np.int32), result_size_col.astype(np.int32)))
    
    # Iterate over rows
    row_index = 0
    while row_index < result_size_row:
        # Calculate the row bounds of the current block
        row_start = row_index * kernel_size
        row_end = row_start + kernel_size
        # Iterate over columns
        col_index = 0
        while col_index < result_size_col:
            # Calculate the column bounds of the current block
            col_start = col_index * kernel_size
            col_end = col_start + kernel_size
            # Store the average value from this block in the output array
            result_arr[row_index, col_index] = np.mean(frame[row_start:row_end, col_start:col_end])
            col_index += 1
        row_index += 1

    # Binarize: set values below threshold to 0 and values at or above threshold to 1
    result_arr[result_arr < threshold] = 0
    result_arr[result_arr >= threshold] = 1
    return result_arr

def redefine_values(frame, kernel_size, dimensions):
    """
    Extracts representative values from a 2D frame by sampling from kernel-sized blocks.
    
    Divides the input frame into non-overlapping kernel_size x kernel_size blocks
    and extracts a value from the center of each block. Handles edge cases where
    the frame dimensions are not evenly divisible by kernel_size.
    
    Args:
        frame: A 2D or 3D numpy array representing the input frame.
        kernel_size: The size of blocks to partition the frame (kernel_size x kernel_size).
        dimensions: The number of feature dimensions in the output (e.g., 1 for grayscale, 3 for RGB).
        
    Returns:
        ndarray: A 3D numpy array with shape (result_rows, result_cols, dimensions) containing sampled values, or -1 if frame is smaller than kernel_size.
    """
    # Return error value if frame is smaller than kernel size
    if frame.shape[0] < kernel_size or frame.shape[1] < kernel_size:
        return -1
    
    # Calculate the theoretical centre position within each kernel block
    centre = (kernel_size + 1) / 2
    # Special case: if kernel is 1x1, return the original frame
    if kernel_size == 1:
        return frame
    
    # Calculate the dimensions of the output array
    result_size_col = np.floor(frame.shape[1] / kernel_size).astype(np.int32)
    result_size_row = np.floor(frame.shape[0] / kernel_size).astype(np.int32)

    # Calculate remainders for edge blocks that don't fit the full kernel size
    rem_col = frame.shape[1] % kernel_size
    rem_row = frame.shape[0] % kernel_size
    # Adjust centre position for remainder column blocks
    if rem_col != 0:
        rem_centre_col = np.floor(rem_col / 2) - 1
    else:
        rem_centre_col = centre
    # Adjust centre position for remainder row blocks
    if rem_row != 0:
        rem_centre_row = np.floor(rem_row / 2) - 1
    else:
        rem_centre_row = centre

    # Initialize output array with zeros
    result_arr = np.zeros((result_size_row.astype(np.int32), result_size_col.astype(np.int32), int(dimensions)))
    
    # Iterate over rows
    row_index = 0
    while row_index < result_size_row:
        # For the last row, use the adjusted centre position; otherwise use standard centre
        if row_index == result_size_row - 1:
            row_centre = row_index * kernel_size + rem_centre_row
        else:
            row_centre = row_index * kernel_size + centre
        # Iterate over columns
        col_index = 0

        while col_index < result_size_col:
            # For the last column, use the adjusted centre position; otherwise use standard centre
            if col_index == result_size_col - 1:
                col_centre = col_index * kernel_size + rem_centre_col
            else:
                col_centre = col_index * kernel_size + centre
            # Extract the value at the centre of this block
            result_arr[row_index, col_index] = frame[int(row_centre), int(col_centre)]
            col_index += 1
        row_index += 1

    return result_arr

if __name__ == "__main__":
    # Define a reference point in 3D space
    point = np.array([50, 50, 600])
    # Generate a random set of 307,200 obstacle points in a 100x100x100 space
    obstacles = np.random.random((307_200, 3)) * 100
    # Add an offset to a subset of obstacles to create spatial clustering
    obstacles[100_000:200_000, :] += 500
    # Cluster the obstacles into 200 groups and retrieve labels and centroids
    labels, centres = abstract(obstacles, size=200)
    # Alternative: cluster relative to the reference point (commented out)
    #labels, centres = abstract_diff(point, obstacles, size=200)
    # Print the cluster centroids
    print(centres)

    # Collect obstacles starting from the nearest cluster until reaching 1000 obstacles
    obs = fill_obstacles(point, labels, centres, obstacles)
    # Print the shape of the collected obstacles array
    print(np.array(obs).shape)
    # Print the total count of collected obstacles
    print(len(obs))
