import cv2
import numpy as np


def find_contours(mask):
    """
    Finds all contours in a binary mask image.
    
    Uses OpenCV's contour detection with tree hierarchy retrieval and simple 
    chain approximation to extract contour boundaries from the mask.
    
    Args:
        mask: A binary numpy array (2D) where contours are the boundaries between 
              foreground and background regions.
    
    Returns:
        A list of contours, where each contour is a numpy array of (x, y) coordinates 
        representing the boundary points.
    """
    contours, _ = cv2.findContours(mask, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
    return contours


def find_max_contour(mask, contours=None):
    """
    Finds the contour with the largest area in a binary mask.
    
    Args:
        mask: A binary numpy array (2D) to search for contours.
        contours: Optional. A pre-computed list of contours. If None, contours will 
                  be detected from the mask (default: None).
    
    Returns:
        The contour with the largest area, or -1 if no contours are found.
    """
    if contours is None:
        contours = find_contours(mask)

    if len(contours) > 0:
        return max(contours, key=cv2.contourArea)
    else:
        return -1
    

def find_max_contour_centre(mask, max_contour=None):
    """
    Calculates the centroid (center of mass) of the largest contour.
    
    Uses image moments to compute the centroid coordinates, which represent 
    the weighted center of the contour region.
    
    Args:
        mask: A binary numpy array (2D) used to find the largest contour if not provided.
        max_contour: Optional. A pre-computed largest contour. If None, the largest 
                     contour will be found from the mask (default: None).
    
    Returns:
        A tuple (cx, cy) representing the centroid coordinates as integers, 
        or -1 if no valid contour is found.
    """
    if max_contour is None:
        max_contour = find_max_contour(mask)
        if max_contour == -1:
            return -1

    # Calculate image moments to find centroid
    M = cv2.moments(max_contour)
    return int(M['m10']/M['m00']), int(M['m01']/M['m00'])


def find_max_contour_area(mask, max_contour=None):
    """
    Calculates the area (in pixels) of the largest contour.
    
    Args:
        mask: A binary numpy array (2D) used to find the largest contour if not provided.
        max_contour: Optional. A pre-computed largest contour. If None, the largest 
                     contour will be found from the mask (default: None).
    
    Returns:
        The area in pixels as a float, or -1 if no valid contour is found.
    """
    if max_contour is None:
        max_contour = find_max_contour(mask)
        if max_contour == -1:
            return -1

    return cv2.contourArea(max_contour)


def find_max_contour_stats(mask, max_contour=None):
    """
    Calculates both the area and centroid of the largest contour.
    
    Provides convenient access to multiple statistics of the largest contour 
    in a single function call.
    
    Args:
        mask: A binary numpy array (2D) used to find the largest contour if not provided.
        max_contour: Optional. A pre-computed largest contour. If None, the largest 
                     contour will be found from the mask (default: None).
    
    Returns:
        A tuple (area, (cx, cy)) where area is the contour area in pixels and 
        (cx, cy) are the centroid coordinates, or -1 if no valid contour is found.
    """
    if max_contour is None:
        max_contour = find_max_contour(mask)
        if max_contour == -1:
            return -1

    return cv2.contourArea(max_contour), find_max_contour_centre(mask, max_contour)
    

def find_all_contour_centres(mask, contours=None):
    """
    Calculates the centroids of all contours in a binary mask.
    
    Computes the center of mass for each contour using image moments.
    
    Args:
        mask: A binary numpy array (2D) to search for contours.
        contours: Optional. A pre-computed list of contours. If None, contours will 
                  be detected from the mask (default: None).
    
    Returns:
        A tuple (centres, contours) where centres is a list of [cx, cy] centroid 
        coordinates and contours is the list of contour objects.
    """
    if contours is None:
        contours = find_contours(mask)

    centres = []
    for contour in contours:
        # Calculate image moments and derive centroid coordinates
        M = cv2.moments(contour)
        cx, cy = int(M['m10']/M['m00']), int(M['m01']/M['m00'])
        centres.append([cx, cy])

    return centres, contours
    

def find_all_contour_areas(mask, contours=None):
    """
    Calculates the areas of all contours in a binary mask.
    
    Args:
        mask: A binary numpy array (2D) to search for contours.
        contours: Optional. A pre-computed list of contours. If None, contours will 
                  be detected from the mask (default: None).
    
    Returns:
        A tuple (areas, contours) where areas is a list of contour areas in pixels 
        and contours is the list of contour objects.
    """
    if contours is None:
        contours = find_contours(mask)

    areas = []
    for contour in contours:
        areas.append(cv2.contourArea(contour))

    return areas, contours


def isolate_contour(mask, contour):
    """
    Isolates a single contour by creating a mask and extracting its pixel coordinates.
    
    Creates a binary mask containing only the specified contour and returns both 
    the normalized mask and the (row, column) coordinates of all pixels within 
    the contour region.
    
    Args:
        mask: A binary numpy array (2D) providing the reference shape for the output mask.
        contour: A single contour object to isolate.
    
    Returns:
        A tuple (coords, filled_mask) where coords is an Nx2 numpy array of (row, col) 
        pixel coordinates and filled_mask is a binary numpy array with 1s inside the 
        contour and 0s elsewhere.
    """
    # Create empty mask with same dimensions as input
    filled_mask = np.zeros_like(mask)

    # Fill the specified contour with white (255)
    cv2.drawContours(filled_mask, contour, 0, 255, thickness=cv2.FILLED)

    # Extract coordinates of all pixels inside the contour
    ys, xs = np.where(filled_mask == 255)
    # Normalize mask values to 0 and 1
    filled_mask[filled_mask > 0] = 1

    # Stack coordinates as (row, col) pairs
    coords = np.array([ys, xs]).T
    return coords, filled_mask


def isolate_all_contours(mask, contours=None):
    """
    Isolates all contours by creating individual masks and extracting pixel coordinates.
    
    For each contour, creates a separate binary mask and extracts the pixel coordinates 
    of all points within the contour region. Useful for processing each contour 
    independently.
    
    Args:
        mask: A binary numpy array (2D) providing the reference shape for output masks.
        contours: Optional. A pre-computed list of contours. If None, contours will 
                  be detected from the mask (default: None).
    
    Returns:
        A tuple (all_contour_pixels, all_masks) where all_contour_pixels is a list of 
        Nx2 numpy arrays containing (row, col) coordinates for each contour, and 
        all_masks is a list of binary masks with 1s inside each contour and 0s elsewhere.
    """
    if contours is None:
        contours = find_contours(mask)
    
    all_contour_pixels = []
    all_masks = []

    # Process each contour individually
    for i, contour in enumerate(contours):
        # Create an empty mask for this contour
        temp_mask = np.zeros_like(mask)
        # Fill only the current contour with white (255)
        cv2.drawContours(temp_mask, contours, i, 255, thickness=cv2.FILLED)
        # Extract pixel coordinates where the contour is filled
        ys, xs = np.where(temp_mask == 255)
        # Normalize mask values to 0 and 1
        temp_mask[temp_mask > 0] = 1
        all_masks.append(temp_mask)
        # Store coordinates as (row, col) pairs
        all_contour_pixels.append(np.array([ys, xs]).T)

    return all_contour_pixels, all_masks


if __name__ == "__main__":
    dataset_location = './Testing Data/'
    import matplotlib.pyplot as plt
    
    # Load test image and convert to binary mask
    file_name = 'test_data_5.png'
    image = cv2.imread(dataset_location + file_name, 0)
    # Invert: set non-zero pixels to 2, then swap 0 and 2 to create binary mask
    image[image > 0] = 2
    image[image == 0] = 1
    image[image == 2] = 0

    # Extract contours and isolate them
    contours = find_contours(image)
    print(len(contours))
    isolated_contours = isolate_all_contours(image, contours)
    centres, _ = find_all_contour_centres(image, contours=contours)
    centres = np.array(centres)

    # Visualize the mask with contours and their centroids
    plt.imshow(image, cmap='gray')
    print(len(isolated_contours))
    # Plot each contour's pixels
    for cnts in isolated_contours:
        plt.scatter(cnts[:, 1], cnts[:, 0])
    # Plot centroids of all contours
    plt.scatter(centres[:, 0], centres[:, 1])
    plt.show()
