import cv2
import numpy as np
import os
import dlib
from PIL import Image

def detect_keypoints_face(img): 
    # Load the pre-trained face detector from dlib
    detector = dlib.get_frontal_face_detector()

    # Load the pre-trained shape predictor model from dlib: download from existing repository or train one yourself
    # predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat') # https://github.com/henriklg/view-morphing/blob/master/data/shape_predictor_68_face_landmarks.dat
    predictor = dlib.shape_predictor('shape_predictor_81_face_landmarks.dat') # https://github.com/codeniko/shape_predictor_81_face_landmarks/blob/master/
    n_points = 81 # number of points of the predictor

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


def draw_triangulation(img, subdiv, delaunay_color):
    triangleList = subdiv.getTriangleList()
    size = img.shape
    rect = np.array([[0, 0], [0, size[0]], [size[1], size[0]], [size[1] ,0]], np.int32)
    for t in triangleList:
        pt = [(int(t[0]), int(t[1])), (int(t[2]), int(t[3])), (int(t[4]), int(t[5]))]
        if all(cv2.pointPolygonTest(rect, pt[i], False) >= 0 for i in range(3)):
            cv2.line(img, pt[0], pt[1], delaunay_color, 1, cv2.LINE_AA, 0)
            cv2.line(img, pt[1], pt[2], delaunay_color, 1, cv2.LINE_AA, 0)
            cv2.line(img, pt[2], pt[0], delaunay_color, 1, cv2.LINE_AA, 0)

def get_triangulation(img, points):
    size = img.shape
    rect = (0, 0, size[1], size[0])
    subdiv = cv2.Subdiv2D(rect)
    for point in points:
        subdiv.insert(point)
    return subdiv

def apply_affine_transform(warp_image, src, srcTri, dstTri):
    warpMat = cv2.getAffineTransform(np.float32(srcTri), np.float32(dstTri))
    warp_image = cv2.warpAffine(src, warpMat, (warp_image.shape[1], warp_image.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    return warp_image

def morph_triangle(img1, img2, img, t1, t2, t, alpha):
    r = cv2.boundingRect(np.float32([t]))
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))

    tRect = []
    tRectInt = []
    t1Rect = []
    t2Rect = []

    for i in range(3):
        tRect.append(((t[i][0] - r[0]), (t[i][1] - r[1])))
        tRectInt.append(((t[i][0] - r[0]), (t[i][1] - r[1])))
        t1Rect.append(((t1[i][0] - r1[0]), (t1[i][1] - r1[1])))
        t2Rect.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))

    mask = np.zeros((r[3], r[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(tRectInt), (1.0, 1.0, 1.0), 16, 0)

    img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    img2Rect = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]]

    warpImage1 = np.zeros((r[3], r[2], 3), dtype=np.float32)
    warpImage2 = np.zeros((r[3], r[2], 3), dtype=np.float32)

    warpImage1 = apply_affine_transform(warpImage1, img1Rect, t1Rect, tRect)
    warpImage2 = apply_affine_transform(warpImage2, img2Rect, t2Rect, tRect)

    imgRect = (1.0 - alpha) * warpImage1 + alpha * warpImage2

    img[r[1]:r[1] + r[3], r[0]:r[0] + r[2]] = img[r[1]:r[1] + r[3], r[0]:r[0] + r[2]] * (1 - mask) + imgRect * mask

def main():
    input_path = "input_images/"

    # Choose which images to use or import yours
    img1_path = input_path + 'Joconde.jpg'
    img2_path = input_path + 'Joconde_flip.jpg'
    # img1_path = input_path + 'einstein.jpg'
    # img2_path = input_path + 'einstein_flip.jpg'
    # img1_path = input_path + 'jules.jpg'
    # img2_path = input_path + 'jules2.jpg'

    frames = 30
    res_path = './output/'

    if not os.path.exists(res_path):
        os.makedirs(res_path)

    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    size = img1.shape
    # Resize if not the same size
    if (img1.shape != img2.shape):
        if img1.shape[0] < img2.shape[0]:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]), interpolation=cv2.INTER_AREA)
        else:
            img1 = cv2.resize(img1, (img2.shape[1], img2.shape[0]), interpolation=cv2.INTER_AREA)
    size = img1.shape
    print("Size for the output image:", size)

    # key points with automatic face detection AI
    key_points_face1 = detect_keypoints_face(img1) 
    key_points_face2 = detect_keypoints_face(img2)

    # Add points to take all the image
    corners = [(int(0), int(0)), (int(0), int(size[0]-1)), (int(size[1]-1), int(size[0]-1)), (int(size[1]-1) ,int(0))]
    middles = [(int(0), int(size[0] /2)), (int(size[1] /2), int(0)), (int(size[1]-1), int(size[0] /2)),(int(size[1] /2), int(size[0]-1))]
    key_points1 = key_points_face1 + corners + middles
    key_points2 = key_points_face2 + corners + middles

    # If you already have a list of feature points of the images, you can put them in a list and comment the lines above
    # A python file in the directory 'features_points' allow to add manually points on the images
    # Below are the key points for the Joconde images
    # key_points1 = [(122, 206), (122, 228), (125, 249), (129, 271), (135, 293), (144, 314), (156, 333), (171, 349), (192, 354), (216, 352), (242, 341), (264, 326), (283, 305), (295, 281), (301, 254), (304, 226), (306, 198), (122, 192), (130, 183), (143, 181), (157, 184), (171, 189), (196, 187), (214, 179), (234, 176), (255, 179), (271, 188), (183, 206), (181, 225), (179, 244), (177, 263), (166, 269), (173, 274), (182, 277), (193, 274), (203, 270), (137, 206), (146, 201), (157, 202), (167, 209), (156, 211), (145, 211), (215, 208), (225, 200), (238, 199), (249, 203), (239, 209), (226, 209), (161, 297), (168, 295), (176, 293), (184, 295), (193, 293), (207, 295), (225, 297), (209, 307), (195, 311), (186, 311), (177, 310), (169, 306), (166, 298), (177, 300), (185, 301), (194, 300), (219, 298), (194, 299), (185, 300), (177, 298), (131, 121), (143, 120), (164, 128), (193, 128), (242, 118), (270, 122), (293, 156), (120, 151), (126, 131), (121, 188), (302, 182), (277, 133), (228, 125), (0, 0), (0, 499), (449, 499), (449, 0), (0, 250), (225, 0), (449, 250), (225, 499), (96, 385), (75, 252), (103, 139), (155, 66), (205, 45), (252, 45), (280, 48), (313, 67), (347, 103), (367, 156), (377, 222), (387, 277), (394, 340), (399, 390), (77, 323), (84, 187), (79, 455), (314, 399), (194, 398), (140, 457), (109, 496), (313, 442), (313, 493), (402, 438), (126, 99), (190, 369), (166, 424), (298, 354), (315, 423), (314, 469), (308, 374), (179, 355)] 
    # key_points2 = [(141, 196), (142, 225), (146, 254), (152, 281), (164, 305), (184, 326), (207, 341), (233, 352), (258, 353), (279, 347), (294, 331), (306, 313), (316, 292), (322, 270), (326, 248), (329, 226), (329, 203), (175, 187), (192, 177), (213, 174), (234, 177), (252, 185), (283, 186), (296, 180), (310, 177), (323, 178), (331, 187), (268, 204), (270, 223), (272, 242), (274, 261), (247, 269), (258, 272), (268, 276), (278, 272), (285, 268), (198, 202), (211, 197), (224, 198), (235, 206), (223, 208), (210, 208), (284, 207), (294, 198), (306, 198), (315, 203), (307, 208), (296, 209), (224, 296), (243, 294), (257, 292), (266, 294), (275, 292), (283, 293), (291, 295), (282, 305), (274, 310), (265, 311), (255, 311), (241, 307), (230, 297), (256, 299), (266, 300), (274, 299), (286, 297), (274, 297), (265, 299), (256, 298), (180, 113), (203, 112), (237, 122), (269, 124), (303, 113), (317, 119), (331, 158), (161, 142), (174, 122), (142, 180), (329, 182), (325, 133), (297, 121), (0, 0), (0, 499), (449, 499), (449, 0), (0, 250), (225, 0), (449, 250), (225, 499), (58, 383), (75, 248), (89, 134), (145, 63), (202, 45), (236, 44), (272, 52), (301, 69), (327, 104), (359, 158), (370, 223), (379, 277), (374, 332), (358, 381), (59, 321), (78, 186), (22, 445), (254, 404), (143, 378), (141, 447), (144, 497), (310, 440), (354, 496), (363, 438), (115, 91), (151, 345), (140, 414), (260, 369), (285, 423), (334, 470), (257, 388), (157, 325)] 

    # Get and draw the Delaunay triangulation of the images based on the features points
    subdiv1 = get_triangulation(img1, key_points1)
    subdiv2 = get_triangulation(img2, key_points2)

    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)

    img1_copy = img1.copy()
    img2_copy = img2.copy()
    color = (255, 255, 255)

    draw_triangulation(img1_copy, subdiv1, color)
    draw_triangulation(img2_copy, subdiv2, color)
    cv2.imwrite(res_path + "tri1.jpg", img1_copy)
    cv2.imwrite(res_path + "tri2.jpg", img2_copy)

    triangles1 = subdiv1.getTriangleList()
    len_tri = len(triangles1)

    index = []
    size = img1.shape
    for i in range(len_tri):
        cur = []
        for j in range(0, 6, 2):
            pt = (triangles1[i][j], triangles1[i][j + 1])
            for k in range(len(key_points1)):
                if tuple(key_points1[k]) == pt:
                    cur.append((pt, k))
                    break
        if len(cur) == 3:
            rect = np.array([[0, 0], [0, size[0]], [size[1], size[0]], [size[1] ,0]], np.int32)
            if all(cv2.pointPolygonTest(rect, cur[k][0], False) >= 0 for k in range(3)):
                index.append([cur[k][1] for k in range(3)])

    for id in range(frames):
        alpha = 1.0 * id / (frames - 1)
        img_res = np.zeros(img1.shape, dtype=np.float32)
        key_points_res = [(int((1 - alpha) * key_points1[i][0] + alpha * key_points2[i][0]), 
                           int((1 - alpha) * key_points1[i][1] + alpha * key_points2[i][1])) for i in range(len(key_points1))]

        for i in index:
            t1 = [key_points1[i[j]] for j in range(3)]
            t2 = [key_points2[i[j]] for j in range(3)]
            t_res = [key_points_res[i[j]] for j in range(3)]
            morph_triangle(img1, img2, img_res, t1, t2, t_res, alpha)

        cv2.imwrite(res_path + "res" + str(id) + ".jpg", img_res)

    # Create GIF
    images = [Image.open(res_path + f"res{id}.jpg") for id in range(frames)]
    images[0].save(res_path + 'morphing.gif', save_all=True, append_images=images[1:], duration=100, loop=0)


if __name__ == "__main__":
    main()
