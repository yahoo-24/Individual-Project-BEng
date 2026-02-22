import cv2
import numpy as np


def find_contours(mask):
    contours, _ = cv2.findContours(mask, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
    return contours


def find_max_contour(mask, contours=None):
    if contours is None:
        contours = find_contours(mask)

    if len(contours) > 0:
        return max(contours, key=cv2.contourArea)
    else:
        return -1
    

def find_max_contour_centre(mask, max_contour=None):
    if max_contour is None:
        max_contour = find_max_contour(mask)
        if max_contour == -1:
            return -1

    M = cv2.moments(max_contour)
    return int(M['m10']/M['m00']), int(M['m01']/M['m00'])


def find_max_contour_area(mask, max_contour=None):
    if max_contour is None:
        max_contour = find_max_contour(mask)
        if max_contour == -1:
            return -1

    return cv2.contourArea(max_contour)


def find_max_contour_stats(mask, max_contour=None):
    if max_contour is None:
        max_contour = find_max_contour(mask)
        if max_contour == -1:
            return -1

    return cv2.contourArea(max_contour), find_max_contour_centre(mask, max_contour)
    

def find_all_contour_centres(mask, contours=None):
    if contours is None:
        contours = find_contours(mask)

    centres = []
    for contour in contours:
        M = cv2.moments(contour)
        cx, cy = int(M['m10']/M['m00']), int(M['m01']/M['m00'])
        centres.append([cx, cy])

    return centres, contours
    

def find_all_contour_areas(mask, contours=None):
    if contours is None:
        contours = find_contours(mask)

    areas = []
    for contour in contours:
        areas.append(cv2.contourArea(contour))

    return areas, contours


def isolate_contour(mask, contour):
    # Create empty mask
    filled_mask = np.zeros_like(mask)

    # Fill a specific contour (e.g., first one)
    cv2.drawContours(filled_mask, contour, 0, 255, thickness=cv2.FILLED)

    # Get coordinates of all pixels inside contour
    ys, xs = np.where(filled_mask == 255)

    # Coordinates as (row, col)
    coords = np.array([ys, xs]).T
    return coords


def isolate_all_contours(mask, contours):
    all_contour_pixels = []

    for i, contour in enumerate(contours):
        temp_mask = np.zeros_like(mask)
        cv2.drawContours(temp_mask, contours, i, 255, thickness=cv2.FILLED)
        ys, xs = np.where(temp_mask == 255)
        all_contour_pixels.append(np.array([ys, xs]).T)

    return all_contour_pixels



# x, y, w, h = cv2.boundingRect(contours[0])

# roi = binary_mask[y:y+h, x:x+w]

# ys, xs = np.where(roi == 1)

# # shift coordinates back to original image
# coords = [(y + yy, x + xx) for yy, xx in zip(ys, xs)]

if __name__ == "__main__":
    dataset_location = '/home/yahia/Downloads/Dataset/Testing Data/'
    import matplotlib.pyplot as plt
    file_name = 'test_data_5.png'
    image = cv2.imread(dataset_location + file_name, 0)
    image[image > 0] = 2
    image[image == 0] = 1
    image[image == 2] = 0

    contours = find_contours(image)
    print(len(contours))
    isolated_contours = isolate_all_contours(image, contours)
    centres, _ = find_all_contour_centres(image, contours=contours)
    centres = np.array(centres)


    plt.imshow(image, cmap='gray')
    print(len(isolated_contours))
    for cnts in isolated_contours:
        plt.scatter(cnts[:, 1], cnts[:, 0])
    plt.scatter(centres[:, 0], centres[:, 1])
    plt.show()