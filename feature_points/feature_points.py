"""
Python code if you want to add key points manually.

Change the image path to correspond to your images.
Execute the file from this directory.
"""

import numpy as np
import cv2
import dlib

def detect_keypoints_face(img, current_image): 
    # Load the pre-trained face detector from dlib
    detector = dlib.get_frontal_face_detector()
    # Load the pre-trained shape predictor model from dlib
    predictor = dlib.shape_predictor('../shape_predictor_68_face_landmarks.dat') # https://github.com/henriklg/view-morphing/blob/master/data/shape_predictor_68_face_landmarks.dat
    n_points = 68

    # Function to convert dlib shape to numpy array
    def shape_to_np(shape, dtype="int"):
        coords = np.zeros((n_points, 2), dtype=dtype)
        for i in range(0, n_points):
            coords[i] = (shape.part(i).x, shape.part(i).y)
        return coords
    image = img.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    keypoints_list = []

    # Loop over the face detections
    for rect in rects:
        shape = predictor(gray, rect)
        shape = shape_to_np(shape)
        for (x, y) in shape:
            keypoints_list.append((int(x), int(y)))
            cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
    return keypoints_list


def automatic_face_correspondence(img1, img2):
    size = img1.shape
    # key points with automatic face detection AI
    key_points_face1 = detect_keypoints_face(img1, 1) 
    key_points_face2 = detect_keypoints_face(img2, 2)

    # Add points to take all the image
    corners = [(int(0), int(0)), (int(0), int(size[0]-1)), (int(size[1]-1), int(size[0]-1)), (int(size[1]-1) ,int(0))]
    middles = [(int(0), int(size[0] /2)), (int(size[1] /2), int(0)), (int(size[1]-1), int(size[0] /2)),(int(size[1] /2), int(size[0]-1))]
    key_points1 = key_points_face1 + corners + middles
    key_points2 = key_points_face2 + corners + middles

    return key_points1, key_points2


def read_points(file_path):
    """Read points from a text file and put them in a list"""
    with open(file_path, 'r') as file:
        content = file.read().strip()
        points = []
        content = content[1:-1]  # Remove the surrounding square brackets
        pairs = content.split('), (')
        for pair in pairs:
            pair = pair.replace('(', '').replace(')', '')
            x, y = map(int, pair.split(', '))
            points.append((x, y))
    return points


def getPointCorrespondences(im1, im2, point_list_im1, point_list_im2):
    global point, point_click, current_image, update_point
    update_point = False
    point = (-1, -1)
    point_click = (-1, -1)
    current_image = 1

    def get_coords(event, x, y, flags, param):
        global point, update_point
        if event == cv2.EVENT_LBUTTONDOWN:
            point = x, y
            print(f"Add Point Image {current_image}:", point)
            update_point = True

    cv2.namedWindow('Images', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('Images', get_coords)

    # Draw existing points
    for point in point_list_im1:
        cv2.circle(im1, point, 2, (0, 0, 255), 2)
    for point in point_list_im2:
        cv2.circle(im2, point, 2, (0, 0, 255), 2)

    while True:
        if current_image == 1:
            numpy_horizontal = np.hstack((im1,))
        else:
            numpy_horizontal = np.hstack((im2,))

        cv2.imshow('Images', numpy_horizontal)
        k = cv2.waitKey(20) & 0xFF

        if update_point:
            if current_image == 1:
                cv2.circle(im1, point, 2, (0, 0, 255), 2)
                point_list_im1.append(point)
            else:
                cv2.circle(im2, point, 2, (0, 0, 255), 2)
                point_list_im2.append(point)
            current_image = 2 if current_image == 1 else 1
            update_point = False

        if k == ord('q') or cv2.getWindowProperty('Images', cv2.WND_PROP_VISIBLE) < 1:  # Close the window
            break

    cv2.destroyAllWindows()
    return point_list_im1, point_list_im2


def main(im1, im2):
    point_list_im1, point_list_im2 = automatic_face_correspondence(im1, im2)
    print(len(point_list_im1))
    print(len(point_list_im2))
    print()
    pts1, pts2 = getPointCorrespondences(im1, im2, point_list_im1, point_list_im2)
    
    print("Points on Image 1:", pts1)
    print("Points on Image 2:", pts2)
    
    # Save key points into a text file
    with open('key_points.txt', 'w') as file:
        file.write(str(pts1) + '\n' + str(pts2))

    if (len(pts1) != len(pts2)):
        print("Be careful: The number of points selected in each image is not equal.")


if __name__ == '__main__':
    # Path to images
    img1 = cv2.imread('../input_images/Joconde.jpg')
    img2 = cv2.imread('../input_images/Joconde_flip.jpg')
    # img1 = cv2.imread('../input_images/jules.jpg')
    # img2 = cv2.imread('../input_images/jules2.jpg')

    # if you want to change the size of the image
    # img1 = cv2.resize(img1, (1125, 1500), interpolation=cv2.INTER_AREA)
    # img2 = cv2.resize(img2, (1125, 1500), interpolation=cv2.INTER_AREA)
    
    main(img1, img2)