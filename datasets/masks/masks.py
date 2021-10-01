"""
Given: A set of facial landmarks
Goal: Warp and transform the image to an output coordinate space

1. face has to be centered in the image
2. rotated such that the eyes lie on a horizontal line
3. scaled such that the size of faces are appr. identical
"""
import pickle
import cv2
import os
import numpy as np

path = "bbox_landmark_mtcnn.pkl"

marks = pickle.load(open(path, "rb"))


def extract(pickle):
    rmvprefix = lambda path: path[69:]
    image_paths, bbox, landmarks = [], [], []
    c = 0
    for key in marks.items():
        image_paths.append(rmvprefix(key[0]))
        bbox.append(key[1][0])
        landmarks.append(key[1][1])
    return image_paths, bbox, landmarks


paths, boxes, ldmarks = extract(marks)
print(boxes[0][0])


def landmark_name(imgfilename):
    ldmpaths = []
    rmvapdx = lambda name: name[:len(name) - 4]
    ldmname = rmvapdx(imgfilename) + "_" + "landmarks.txt"
    return ldmname


print(landmark_name(paths[0]))


def mark_img(image, boxes):
    for box in boxes:
        for value in range(len(box)):
            for value2 in range(len(box)):
                print(value, value2)
                cv2.circle(image, (int(box[value]), int(box[value2])), 2, (0, 0, 255), 3)
    cv2.imshow("marked", image)
    cv2.waitKey()


def draw_marks(image, marks):
    image = cv2.imread(image)
    for point in marks:
        x, y = int(point[0]), int(point[1])
        image = cv2.circle(image, (x, y), 3, (0, 0, 255), -1)
    cv2.imshow("marked", image)
    cv2.waitKey()
    return image


def get_landmarks(path):
    marks = np.loadtxt(path)
    return marks


imgpath = os.path.join("..", "ckp", "images", paths[0])
markspath = os.path.join("..", "ckp", "landmarks", landmark_name(paths[0]))

marks = get_landmarks(markspath)


# image = draw_marks(imgpath, marks)


def detect_face(image):
    cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
    rects = cascade.detectMultiScale(image, 1.3, 4, cv2.CASCADE_SCALE_IMAGE, (20, 20))
    if len(rects) == 0:
        return [], image
    rects[:, 2:] += rects[:, :2]
    return rects, image


def box(rects, img):
    for x1, y1, x2, y2 in rects:
        print(x1, y1, x2, y2)
        img = cv2.rectangle(img, (x1, y1), (x2, y2), (127, 255, 0), 2)
    # cv2.imwrite('/vagrant/img/detected.jpg', img)
    return img

import math
def similarityTransform(inPoints, outPoints):
    """
    :param inpts: coordinates of the center of left and right eye
    :param outpts: destination coordinates of the center of left and right eye
    :return: affine transformation
    """
    s60 = math.sin(60 * math.pi / 180)
    c60 = math.cos(60 * math.pi / 180)
    inPts = np.copy(inPoints).tolist()
    outPts = np.copy(outPoints).tolist()

    xin = c60 * (inPts[0][0] - inPts[1][0]) - s60 * (inPts[0][1] - inPts[1][1]) + inPts[1][0]
    yin = s60 * (inPts[0][0] - inPts[1][0]) + c60 * (inPts[0][1] - inPts[1][1]) + inPts[1][1]
    inPts.append([np.int(xin), np.int(yin)])

    xout = c60 * (outPts[0][0] - outPts[1][0]) - s60 * (outPts[0][1] - outPts[1][1]) + outPts[1][0]
    yout = s60 * (outPts[0][0] - outPts[1][0]) + c60 * (outPts[0][1] - outPts[1][1]) + outPts[1][1]
    outPts.append([np.int(xout), np.int(yout)])

    tform = cv2.estimateAffinePartial2D(np.array([inPts]), np.array([outPts]))
    return cv2.UMat(tform[0])

def calc_new_marks(orig_img_shape, bb_p1, bb_p2, ld_old):
    landmarks_new = 0
    print("Orig shape: ", orig_img_shape)
    print("BBox p1: ", bb_p1)
    print("BBox p2: ", bb_p2)
    landmarks = [[ld_old[0][0], ld_old[0][5]],
                 [ld_old[0][1], ld_old[0][6]],
                 [ld_old[0][2], ld_old[0][7]],
                 [ld_old[0][3], ld_old[0][8]],
                 [ld_old[0][4], ld_old[0][9]]]
    print("Ldmarks: ", landmarks)
    tf1 = similarityTransform(bb_p1, bb_p2)
    print("Transformation: ", tf1.get())
    p2_ldm_neutral = np.reshape(np.array(landmarks), (5, 1, 2))
    p2_ldm_neutral = cv2.transform(p2_ldm_neutral, tf1).get()
    p2_ldm_neutral = np.float32(np.reshape(p2_ldm_neutral, (5, 2)))
    #
    return p2_ldm_neutral


# image = cv2.imread(imgpath)
# rects, image = detect_face(image)
# image = box(rects, image)
cv2.waitKey()
x1, y1, x2, y2 = int(boxes[0][0][0]), int(boxes[0][0][1]), int(boxes[0][0][2]), int(boxes[0][0][3])
# x, y, w, h
image = cv2.imread(imgpath)
# image = cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
# image = cv2.circle(image, (x1,y1), 2, (255, 0, 0), 5)
# image = cv2.circle(image, (x2,y2), 2, (255, 0, 0), 5)

# https://github.com/ipazc/mtcnn
# https://learnopencv.com/image-alignment-feature-based-using-opencv-c-python/
# https://stackoverflow.com/questions/59411741/python-align-two-images-according-to-specified-co-ordinates

destpoints = [(x1,y1),(x2,y2)]
srcpoints = [(0,0),(490,640)]
new_ldm = calc_new_marks(np.shape(image), destpoints, srcpoints, ldmarks[0])
print("new landmarks: ", new_ldm)
#image = image[y1:y2,x1:x2]
print("img shape new: ", np.shape(image))
"""
for idx in range(int(len(ldmarks[0][0]) / 2)):
    x, y = ldmarks[0][0][0 + idx], ldmarks[0][0][5 + idx]
    # x, y = ldmarks[0][0][idx], ldmarks[0][0][idx]
    print(x, y)
    # image = cv2.circle(image, (x, y), 2, (255, 0, 0), 5)
    image = cv2.circle(image, (x, y), 2, (255, 0, 0), 5)
"""
for mark in new_ldm:
    print(mark)
    image = cv2.circle(image, (mark[0], mark[1]), 2, (255, 255, 0), 5)

cv2.imshow("", image)
cv2.waitKey()

"""
https://github.com/ipazc/mtcnn
Es wurde das mtcnn benutzt welches 5 facial landmarks findet.
Den Mittelpunkt des linken und des rechten Auges,
die Nasenspitze und die beiden Mundwinkel.
Also die bbox landmark datei liefert eben
pro Bild die Bounding box welche,
die aus 5 Punkten besteht.
Die Koordinaten x1, y1, x2, y2. Damit lässt 
sich mit opencv ein rechteck malen.
Der fünfte punkt ist 1, wahrscheinlich
wegen der affinen transformation
"""


def crop_to_box(image, box, marks, name="default.png", visualize=False):
    """
    :param img:
    :param box: array with 5 values, 4 are coords
    :param marks:
    :return:
    """
    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    if visualize:
        image = cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        for idx in range(int(len(marks) / 2)):
            x, y = marks[0 + idx], marks[5 + idx]
            image = cv2.circle(image, (x, y), 2, (255, 0, 0), 5)
    image = image[y1:y2, + x1:x2]
    # cv2.imwrite(filename=name, img=image)
    cv2.imshow("", image)
    cv2.waitKey()
    return image


idx = 7  # sample nr., 4 images per emotion per subject

def run(num):
    for i in range(num):
        idx = i
        imgpath = os.path.join("..", "ckp", "images", paths[idx])
        image = cv2.imread(imgpath)
        crop_to_box(image, boxes[idx][0], ldmarks[idx][0], name=str(idx) + ".png", visualize=True)


# run(2)

#  S151_002_00000029.png