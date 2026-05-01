import cv2
import sys
import numpy as np


def detect_red(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    low_thr_1 = np.array([0, 150, 30])
    high_thr_1 = np.array([10, 255, 255])
    low_thr_2 = np.array([170, 150, 30])
    high_thr_2 = np.array([179, 255, 255])
    threshold_1 = cv2.inRange(frame, low_thr_1, high_thr_1)
    threshold_2 = cv2.inRange(frame, low_thr_2, high_thr_2)
    threshold = cv2.bitwise_or(threshold_1, threshold_2)
    return threshold

def blur(image, size=(13,13)):
    return cv2.blur(image, size)

"""
The code below is for testing. Use detect_red for the filter.
You can use blur if you want before passing the input to detect_red.
It is unclear if blurring has any major effects on the performance of the filter.
"""

def video():
    s = 0
    if len(sys.argv) > 1:
        s = sys.argv[1]
    source = cv2.VideoCapture(s)

    win_name = 'Camera Preview'
    filtered_name = 'Filtered View'
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.namedWindow(filtered_name, cv2.WINDOW_NORMAL)

    key = ord('2')
    show_type = 0
    while key != 27: # Escape
        has_frame, frame = source.read()
        if not has_frame:
            break
        cv2.imshow(win_name, frame)
        threshold = detect_red(cv2.blur(frame, (13, 13)))
        filtered = cv2.bitwise_and(frame, frame, mask=threshold)
        key = cv2.waitKey(1)
        if key == ord('1'):
            show_type = 0
        elif key == ord('2'):
            show_type = 1

        if show_type:
            display = filtered
        else:
            display = threshold
        cv2.imshow(filtered_name, display)

    source.release()
    cv2.destroyWindow(win_name)
    cv2.destroyWindow(filtered_name)

def path_test():
    import cpp
    import matplotlib.pyplot as plt
    s = 0
    if len(sys.argv) > 1:
        s = sys.argv[1]
    source = cv2.VideoCapture(s)

    win_name = 'Camera Preview'
    filtered_name = 'Filtered View'
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.namedWindow(filtered_name, cv2.WINDOW_NORMAL)
    a = 0

    key = ord('2')
    show_type = 0
    while key != 27: # Escape
        has_frame, frame = source.read()
        if not has_frame:
            break
        cv2.imshow(win_name, frame)
        threshold = detect_red(cv2.blur(frame, (13, 13)))
        filtered = cv2.bitwise_and(frame, frame, mask=threshold)
        key = cv2.waitKey(1)
        if key == ord('1'):
            show_type = 0
        elif key == ord('2'):
            show_type = 1

        if show_type:
            display = filtered
        else:
            display = threshold
        cv2.imshow(filtered_name, display)
        if a == 0:
            import time
            time.sleep(2)
            a += 1
            continue
        plt.imshow(threshold)
        points = cpp.spiral_path(threshold)
        cpp.plot(points)
        plt.savefig("plot.png")
        plt.close()
        graph_img = cv2.imread("plot.png")
        cv2.imshow("Line Graph in OpenCV", graph_img)

    source.release()
    cv2.destroyWindow(win_name)
    cv2.destroyWindow(filtered_name)

def image_input(source):
    image = cv2.imread(source, 1)
    display = detect_red(cv2.blur(image, (13, 13)))
    filtered = cv2.bitwise_and(image, image, mask=display)
    win_name = 'Filtered Image'
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.imshow(win_name, display)
    cv2.imshow('Original', image)
    cv2.imshow('Partial', filtered)
    cv2.waitKey(0)

    cv2.destroyWindow(win_name)
    cv2.destroyWindow('Original')
    cv2.destroyWindow('Partial')
    

if __name__ == "__main__":
    #source = "Blood.png"
    #image_input(source)
    path_test()
