from sklearn.cluster import KMeans
import numpy as np

def abstract(obstacles, size=50):
    """
    The function sorts the obstacels into (size=50) clusters and returns the centres of each cluster with them
    
    :param obstacles: This should be an nx3 numpy array where n is the number of obstacles and the 3 are the x, y, z dimensions
    :param size: The number of clusters -> int
    """

    model = KMeans(size, init='k-means++')
    model.fit(obstacles)
    labels = model.labels_
    centres = model.cluster_centers_
    return labels, centres

def abstract_diff(point, obstacles, size=50):
    """
    The function sorts the obstacels into (size=50) clusters and returns the centres of each cluster with them
    
    :param obstacles: This should be an nx3 numpy array where n is the number of obstacles and the 3 are the x, y, z dimensions
    :param size: The number of clusters -> int
    """

    data = obstacles - point
    model = KMeans(size, init='k-means++')
    model.fit(data)
    labels = model.labels_
    centres = model.cluster_centers_
    return labels, centres

def find_distances(point, centres):
    if len(centres) == 0:
        return -1
    return np.linalg.norm(centres - point, axis=1)

def extract_closest(point, labels, centres, obstacles):
    if len(labels) == 0 or len(centres) == 0:
        return -1, -1
    
    distances_to_centre = find_distances(point, centres)
    index = np.argmin(distances_to_centre) # Closest centre to the position passed into the function

    return obstacles[labels == index], index

def extract_closest_given_distances(labels, obstacles, distances_to_centre):
    if len(labels) == 0:
        return -1, -1
    
    index = np.argmin(distances_to_centre) # Closest centre to the position passed into the function

    return obstacles[labels == index], index

def fill_obstacles(point, labels, centres, obstacles, n=1000):
    if len(labels) == 0 or len(centres) == 0:
        return -1, -1
    
    obs = []
    distances_to_centre = find_distances(point, centres)
    index = np.argmin(distances_to_centre) # Closest centre to the position passed into the function
    obs.extend(obstacles[labels == index].tolist())
    length = len(obs)

    while length < n:
        distances_to_centre[index] = np.inf
        o, index = extract_closest_given_distances(labels, obstacles, distances_to_centre)
        obs.extend(o.tolist())
        length += len(o)

    return obs

def max_pool(frame, kernel_size):
    if frame.shape[0] < kernel_size or frame.shape[1] < kernel_size:
        return -1
    
    result_size_col = np.floor(frame.shape[1] / kernel_size)
    result_size_row = np.floor(frame.shape[0] / kernel_size)
    result_arr = np.zeros((result_size_row.astype(np.int32), result_size_col.astype(np.int32)))
    row_index = 0
    while row_index < result_size_row:
        row_start = row_index * kernel_size
        row_end = row_start + kernel_size
        col_index = 0
        while col_index < result_size_col:
            col_start = col_index * kernel_size
            col_end = col_start + kernel_size
            result_arr[row_index, col_index] = np.max(frame[row_start:row_end, col_start:col_end])
            col_index += 1
        row_index += 1

    return result_arr

def avg_pool(frame, kernel_size, threshold=0.75):
    if frame.shape[0] < kernel_size or frame.shape[1] < kernel_size:
        return -1
    
    result_size_col = np.floor(frame.shape[1] / kernel_size)
    result_size_row = np.floor(frame.shape[0] / kernel_size)
    result_arr = np.zeros((result_size_row.astype(np.int32), result_size_col.astype(np.int32)))
    row_index = 0
    while row_index < result_size_row:
        row_start = row_index * kernel_size
        row_end = row_start + kernel_size
        col_index = 0
        while col_index < result_size_col:
            col_start = col_index * kernel_size
            col_end = col_start + kernel_size
            result_arr[row_index, col_index] = np.mean(frame[row_start:row_end, col_start:col_end])
            col_index += 1
        row_index += 1

    result_arr[result_arr < threshold] = 0
    result_arr[result_arr >= threshold] = 1
    return result_arr

def redefine_values(frame, kernel_size, dimensions):
    if frame.shape[0] < kernel_size or frame.shape[1] < kernel_size:
        return -1
    
    centre = (kernel_size + 1) / 2
    if kernel_size == 1:
        return frame
    # print(f"Size: {kernel_size}, Centre: {centre}")
    
    result_size_col = np.floor(frame.shape[1] / kernel_size).astype(np.int32)
    result_size_row = np.floor(frame.shape[0] / kernel_size).astype(np.int32)
    # print(f"Rows: {result_size_row}")

    rem_col = frame.shape[1] % kernel_size
    rem_row = frame.shape[0] % kernel_size
    if rem_col != 0:
        rem_centre_col = np.floor(rem_col / 2) - 1
    else:
        rem_centre_col = centre
    if rem_row != 0:
        rem_centre_row = np.floor(rem_row / 2) - 1
    else:
        rem_centre_row = centre

    result_arr = np.zeros((result_size_row.astype(np.int32), result_size_col.astype(np.int32), int(dimensions)))
    row_index = 0
    while row_index < result_size_row:
        if row_index == result_size_row - 1:
            row_centre = row_index * kernel_size + rem_centre_row
        else:
            row_centre = row_index * kernel_size + centre
        col_index = 0
        # print(row_centre)

        while col_index < result_size_col:
            if col_index == result_size_col - 1:
                col_centre = col_index * kernel_size + rem_centre_col
            else:
                col_centre = col_index * kernel_size + centre
            result_arr[row_index, col_index] = frame[int(row_centre), int(col_centre)]
            col_index += 1
        row_index += 1

    return result_arr

if __name__ == "__main__":
    point = np.array([50, 50, 600])
    obstacles = np.random.random((307_200, 3)) * 100
    obstacles[100_000:200_000, :] += 500
    labels, centres = abstract(obstacles, size=200)
    #labels, centres = abstract_diff(point, obstacles, size=200)
    print(centres)

    obs = fill_obstacles(point, labels, centres, obstacles)
    print(np.array(obs).shape)
    print(len(obs))