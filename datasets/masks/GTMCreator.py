import os
import cv2
import csv
import math
import numpy
import pickle
from tqdm import tqdm


class GTMCreator:
    """Class for creating Ground truth masks
    of the ck+ dataset"""
    imagedir = "../ckp/images"  # original images
    labels = pickle.load(open("emotion_labels.pkl", "rb")) # label = -1 means there is no label provided
    __bbldmk__ = pickle.load(open("bbox_landmark_mtcnn.pkl", "rb"))

    def __init__(self, trainset):
        """
        :param trainset: path to .csv with trainset
        imagenames like "S005_001_00000009.png"
        """
        self.trainset = trainset
        self.imagenames = self.__get_trainnames__()
        self.datadict = self.__create_dict__()
        # image = self.imagenames[0]
        # image_neutr = self.__get_neutralimagename__(image)

    def __create_dict__(self):
        """extract data from pickle files and create own dict
        The pickle files are formatted like:
        ..."""
        datadict = {}
        rmvprefix = lambda path: path[69:]
        getimgname = lambda path: path[78:]
        for key in self.__bbldmk__.items():
            labelkey = self.__get_emotionkey__((rmvprefix(key[0])))
            if labelkey in self.labels:
                emotion_label = int(self.labels[labelkey])
                # add image_path, boundingbox, landmarks and emotion to a dict
                datadict[str(getimgname(key[0]))] = [rmvprefix(key[0]), key[1][0], key[1][1], emotion_label]
        return datadict

    def run(self, emotion):
        emotions = [1, 2, 3, 4, 5, 6, 7]
        image_sum = 0
        image_counter = 0
        for name in self.imagenames[0:160]:
            imagename_neutral = self.__get_neutralimagename__(name)
            imagename_expressive = name
            # paths to images
            expressive_path = os.path.join(
                self.imagedir, self.datadict[imagename_expressive][0])
            neutral_path = os.path.join(
                self.imagedir, self.datadict[imagename_neutral][0])
            # load expressive and neutral images
            image_expressive = cv2.imread(expressive_path)
            image_neutral = cv2.imread(neutral_path)
            # bounding boxes
            box_expressive = self.datadict[imagename_expressive][1]
            box_neutral = self.datadict[imagename_neutral][1]
            # load 5 landmarks per image
            marks_expressive = self.datadict[imagename_expressive][2]
            marks_neutral = self.datadict[imagename_neutral][2]
            # get emotion label
            emotion_label = self.datadict[imagename_expressive][3]
            if emotion_label == -1:
                print("skipped: ", imagename_expressive)
                continue
            if emotion_label != emotion:
                # skip this loop if current image shows not
                # the target emotion
                continue
            image_counter += 1

            # S151_002_00000029.png has two boxes for example
            if numpy.shape(box_expressive) == (2, 5):
                box_expressive = [box_expressive[0]]
            if numpy.shape(box_neutral) == (2, 5):
                box_neutral = [box_neutral[0]]
            # image_expressive = self.__draw_box__(box_expressive, image_expressive)
            # image_neutral = self.__draw_box__(box_neutral, image_neutral)

            # cv2.imshow("expressive", image_expressive)
            # cv2.imshow("neutral", image_neutral)
            # cv2.waitKey()

            # find homography between landmarks of expr and neutr images
            if numpy.shape(marks_expressive) == (2, 10):
                # print("marks shape: ", numpy.shape(marks_expressive))
                marks_expressive = [marks_expressive[0]]
            if numpy.shape(marks_neutral) == (2, 10):
                marks_neutral = [marks_neutral[0]]

            srcPointsExpr = numpy.array([
                [int(marks_expressive[0][0]), int(marks_expressive[0][5])],
                [int(marks_expressive[0][1]), int(marks_expressive[0][6])],
                [int(marks_expressive[0][2]), int(marks_expressive[0][7])],
                [int(marks_expressive[0][3]), int(marks_expressive[0][8])],
                [int(marks_expressive[0][4]), int(marks_expressive[0][9])]
            ])
            dstPointsNeut = numpy.array([
                [int(marks_neutral[0][0]), int(marks_neutral[0][5])],
                [int(marks_neutral[0][1]), int(marks_neutral[0][6])],
                [int(marks_neutral[0][2]), int(marks_neutral[0][7])],
                [int(marks_neutral[0][3]), int(marks_neutral[0][8])],
                [int(marks_neutral[0][4]), int(marks_neutral[0][9])]
            ])
            # calculate the homography matrix
            Hom, _ = cv2.findHomography(srcPointsExpr, dstPointsNeut, cv2.RANSAC)
            dsize = numpy.shape(image_expressive)

            image_expressive_warped = cv2.warpPerspective(src=image_expressive, M=Hom, dsize=(dsize[1],dsize[0]))
            print("warped image size: ", numpy.shape(image_expressive_warped))
            # calculate absolute difference between expr and neutr image
            diff_image = self.__calc_difference__(image_expressive_warped, image_neutral)
            image_sum += diff_image
        print(image_counter)
        image_sum = 1/image_counter * image_sum
        image_sum = self.__histogram_equalization__(image_sum.astype('uint8'))
        #
        # apply adaptive histogram equalization
        #
        #
        cv2.imshow("", image_sum)
        cv2.waitKey()

    def __histogram_equalization__(self, mask):
        gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        gray_uint = numpy.uint8(gray)
        clahe = cv2.createCLAHE(clipLimit=40, tileGridSize=(8, 8))
        equ = clahe.apply(gray_uint)
        # res = np.hstack((gray_uint, equ))
        return equ

    def __std_histeq__(self, mask):
        gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        gray_uint = numpy.uint8(gray)
        equ = cv2.equalizeHist(gray_uint)
        return equ

    def __calc_difference__(self, eta_imgexpr, eta_imgneutr):
        diff_img = numpy.absolute(numpy.subtract(eta_imgexpr, eta_imgneutr))
        return diff_img

    def __get_trainnames__(self):
        """Extract all imagenames from provided training csv file.
        The .csv file just contains the imagenames like
        "S005_001_00000009.png" seperated by a new line
        """
        with open(self.trainset) as csv_trainfile:
            csv_reader = csv.reader(csv_trainfile, delimiter='\n')
            imagenames = []
            for imagename in csv_reader:
                imagenames.append(imagename[0])
            return imagenames

    def __get_emotionkey__(self, imagesubpath):
        """imagesubpath like 'S005/001/S005_001_00000001.png'
        if emotion is -1 then no label is provided"""
        return imagesubpath[9:17]

    def __get_neutralimagename__(self, imagename):
        """create the imagename for a neutral image
        corresponding to its provided expressive image imagename
        E.g.: S005_001_00000009.png -> S005_001_00000001.png
        """
        neutral = "00000001"
        nimagename = imagename[0:9] + neutral + ".png"
        return nimagename

    def __draw_box__(self, rects, image):
        """get the bounding boxes 2 points fomr boxes array"""
        for vals in rects:
            # print(x1, y1, x2, y2)
            x1, y1, x2, y2 = int(vals[0]), int(vals[1]), int(vals[2]), int(vals[3])
            image = cv2.rectangle(image, (x1, y1), (x2, y2), (127, 255, 0), 2)
        # cv2.imwrite('/vagrant/img/detected.jpg', img)
        return image

    def __landmark_name__(self, imagefilename):
        """create landmark filename"""
        rmvapdx = lambda name: name[:len(name) - 4]
        ldmname = rmvapdx(imagefilename) + "_" + "landmarks.txt"
        return ldmname

    def __draw_landmarks_by_path__(self, imagefilename, visualize_box=False):
        # left eye center:      ldmark[0][0], ldmark[0][5]
        # right eye center:     ldmark[0][1], ldmark[0][6]
        # nose tip:             ldmark[0][2], ldmark[0][7]
        # left mouth corner:    ldmark[0][3], ldmark[0][8]
        # right mouth corner:   ldmark[0][4], ldmark[0][9]
        data = self.datadict[imagefilename]
        path = os.path.join(self.imagedir, data[0])
        box, ldmarks = data[1], data[2]
        image = cv2.imread(path)

        for idx in range(int(len(ldmarks[0]) / 2)):
            x, y = ldmarks[0][0 + idx], ldmarks[0][5 + idx]
            image = cv2.circle(image, (x, y), 2, (255,0,0), 5)

        x1, y1, x2, y2 = int(box[0][0]), int(box[0][1]), int(box[0][2]), int(box[0][3])
        image = image[y1:y2,x1:x2]
        cv2.imshow(imagefilename, image)
        cv2.waitKey()

    def __draw_landmarks__(self, image, landmarks):
        for mark in landmarks:
            x, y = mark[0], mark[1]
            image = cv2.circle(image, (x, y), 2, (255, 0, 0), 5)
        return image

    def test(self, emotion, outdim=(800, 800), outpath=None):
        """Create and save the Ground Truth Mask for a
        certain emotion.
        :param emotion:
        :param out_dim:
        :return:
        """
        out_dim = (800, 800)
        w, h = out_dim
        num_images = 0
        num_landmr = 5
        img_sum = 0
        # destination where the eyes have to be placed in new image
        eyecenterdst = [(numpy.int(0.275 * w), numpy.int(h / 3.5)), (numpy.int(0.725 * w), numpy.int(h / 3.5))]

        # set some boundary points, they will be needed for delaunay triangulation
        boundarypts = numpy.array([(0,0), (w / 2, 0),
                                   (w - 1, 0), (w - 1, h / 2),
                                   (w - 1, h - 1), (w / 2, h - 1),
                                   (0, h - 1), (0, h / 2)])

        for name in self.imagenames:
            # get number of images for class emotion
            if self.datadict[name][3] == emotion:
                num_images += 1

        # determine destination landmark points
        # by applying similarity transform on the provided landmarks
        # landmarks of the first neutral face are chosen as destination points
        im1 = self.imagenames[0]
        im1_neutral = self.__get_neutralimagename__(im1)
        im1dict_neutral = self.datadict[im1_neutral]
        im1subp_neutral = im1dict_neutral[0]
        im1bbox_neutral = im1dict_neutral[1]
        im1lndm_neutral = im1dict_neutral[2]
        im1_lbl_neutral = im1dict_neutral[3]
        # im1_neutral = cv2.imread(os.path.join(self.imagedir, im1subp_neutral))

        p1_ldm_neutral = numpy.array([
            [im1lndm_neutral[0][0], im1lndm_neutral[0][5]],
            [im1lndm_neutral[0][1], im1lndm_neutral[0][6]],
            [im1lndm_neutral[0][2], im1lndm_neutral[0][7]],
            [im1lndm_neutral[0][3], im1lndm_neutral[0][8]],
            [im1lndm_neutral[0][4], im1lndm_neutral[0][9]]])


        eyecentersrc_neutralp1 = [(numpy.int(im1lndm_neutral[0][0]), numpy.int(im1lndm_neutral[0][5])),
                                  (numpy.int(im1lndm_neutral[0][1]), numpy.int(im1lndm_neutral[0][6]))]

        tf1 = self.similarityTransform(eyecentersrc_neutralp1, eyecenterdst)

        p2_ldm_neutral = numpy.reshape(numpy.array(p1_ldm_neutral), (5, 1, 2))
        p2_ldm_neutral = cv2.transform(p2_ldm_neutral, tf1).get()
        p2_ldm_neutral = numpy.float32(numpy.reshape(p2_ldm_neutral, (5, 2)))

        destination_ldms = numpy.append(p2_ldm_neutral, boundarypts, axis=0)

        rect = (0, 0, w, h)  # calculate triangles for delaunay
        dt = self.calculateDelaunayTriangles(rect, numpy.array(destination_ldms))

        for image_expressive in tqdm(self.imagenames, desc ="Creating Mask"):
            # load expressive image information from data dictionary
            imagename = image_expressive
            imagedict = self.datadict[imagename]
            imagesubp = imagedict[0]  # subpath in dataset
            imagebbox = imagedict[1]
            imagelndm = imagedict[2]
            image_lbl = imagedict[3]  # emotion label
            image_expressive = cv2.imread(os.path.join(self.imagedir, imagesubp))

            # get the corresponding neutral face from data dictionary
            imagename_neutral = self.__get_neutralimagename__(imagename)
            imagedict_neutral = self.datadict[imagename_neutral]
            imagesubp_neutral = imagedict_neutral[0]
            imagebbox_neutral = imagedict_neutral[1]
            imagelndm_neutral = imagedict_neutral[2]
            image_lbl_neutral = imagedict_neutral[3]
            image_neutral = cv2.imread(os.path.join(self.imagedir, imagesubp_neutral))

            if image_lbl != emotion:
                # skip image if it does not belong to
                # the provided emotion class
                continue

            # get eyecenter source of expressive image
            eyecentersrc = [(numpy.int(imagelndm[0][0]), numpy.int(imagelndm[0][5])),
                            (numpy.int(imagelndm[0][1]), numpy.int(imagelndm[0][6]))]
            # get eyecenter source of neutral image
            eyecentersrc_neutral = [(numpy.int(imagelndm_neutral[0][0]),
                                     numpy.int(imagelndm_neutral[0][5])),
                                    (numpy.int(imagelndm_neutral[0][1]),
                                     numpy.int(imagelndm_neutral[0][6]))]

            # determine similarity transformation matrices and transform images
            tform = self.similarityTransform(eyecentersrc, eyecenterdst)  # expressive
            tform_neutral = self.similarityTransform(eyecentersrc_neutral, eyecenterdst)  # neutral
            # apply affine transformation to both images
            img_affine = cv2.warpAffine(image_expressive, tform, (w, h)).get()
            img_affine_neutral = cv2.warpAffine(image_neutral, tform_neutral, (w, h)).get()

            # apply similarity transform on points neutral face (landmarks)
            points1_ldm_neutral = numpy.array([
                [imagelndm_neutral[0][0], imagelndm_neutral[0][5]],
                [imagelndm_neutral[0][1], imagelndm_neutral[0][6]],
                [imagelndm_neutral[0][2], imagelndm_neutral[0][7]],
                [imagelndm_neutral[0][3], imagelndm_neutral[0][8]],
                [imagelndm_neutral[0][4], imagelndm_neutral[0][9]]])

            points2_ldm_neutral = numpy.reshape(numpy.array(points1_ldm_neutral), (5, 1, 2))
            points_ldm_neutral = cv2.transform(points2_ldm_neutral, tform_neutral).get()
            points_ldm_neutral = numpy.float32(numpy.reshape(points_ldm_neutral, (5, 2)))
            points_ldm_neutral = numpy.append(points_ldm_neutral, boundarypts, axis=0)

            # apply similarity transform on expressive neutral face (landmarks)
            points1_ldm = numpy.array([
                [imagelndm[0][0], imagelndm[0][5]],
                [imagelndm[0][1], imagelndm[0][6]],
                [imagelndm[0][2], imagelndm[0][7]],
                [imagelndm[0][3], imagelndm[0][8]],
                [imagelndm[0][4], imagelndm[0][9]]])

            points2_ldm = numpy.reshape(numpy.array(points1_ldm), (5, 1, 2))
            points_ldm = cv2.transform(points2_ldm, tform).get()
            points_ldm = numpy.float32(numpy.reshape(points_ldm, (5, 2)))
            points_ldm = numpy.append(points_ldm, boundarypts, axis=0)

            # warp expressive and neutral face to destination landmarks
            output_e = numpy.zeros((h, w, 3), numpy.float32())
            output_n = numpy.zeros((h, w, 3), numpy.float32())

            image_e = numpy.zeros((h, w, 3), numpy.float32())
            image_n = numpy.zeros((h, w, 3), numpy.float32())

            # transform triangles
            for j in range(0, len(dt)):
                tin_e, tout_e = [], []
                tin_n, tout_n = [], []
                for k in range(0, 3):
                    # expressive
                    pIn_e = points_ldm[dt[j][k]]
                    pIn_e = self.constrainPoint(pIn_e, w, h)
                    pOut_e = destination_ldms[dt[j][k]]
                    pOut_e = self.constrainPoint(pOut_e, w, h)
                    tin_e.append(pIn_e)
                    tout_e.append(pOut_e)
                    # neutral
                    pIn_n = points_ldm_neutral[dt[j][k]]
                    pIn_n = self.constrainPoint(pIn_n, w, h)
                    pOut_n = destination_ldms[dt[j][k]]
                    pOut_n = self.constrainPoint(pOut_n, w, h)
                    tin_n.append(pIn_n)
                    tout_n.append(pOut_n)

                img_e = self.warpTriangle(img_affine, image_e, tin_e, tout_e)
                img_n = self.warpTriangle(img_affine_neutral, image_n, tin_n, tout_n)

            output_e = output_e + img_e
            output_n = output_n + img_n
            """
            # ########################################################################
            # try to transform the bounding boxes like the landmarks
            # and then crop the image
            # ------------------------------------------------------------------------
            # ADD CODE HERE

            # apply similarity transform on expressive neutral face bounding box
            points1_bbox = numpy.array([
                [imagebbox[0][0], imagebbox[0][1]],
                [imagebbox[0][2], imagebbox[0][3]],
                [imagebbox[0][4], imagebbox[0][4]]])
            print("coords raw: ", points1_bbox)

            points2_bbox = numpy.reshape(numpy.array(points1_bbox), (3, 1, 2))
            points_bbox = cv2.transform(points2_bbox, tform).get()
            print("coords after trafo: ", points_bbox)
            points_bbox = numpy.float32(numpy.reshape(points_bbox, (3, 2)))
            print("coords reshaped: ", points_bbox)
            # points_bbox = numpy.append(points_bbox, boundarypts, axis=0)

            image = cv2.rectangle(output_e,
                                  (points_bbox[0][0], points_bbox[0][1]),
                                  (points_bbox[1][0], points_bbox[1][1]), (0, 255, 0), 2)

            cv2.imshow("", image)
            cv2.waitKey()

            # ########################################################################
            """
            # calculate absolute difference
            img_diff = numpy.absolute(numpy.subtract(output_e, output_n))
            img_sum += img_diff

        img_sum = 1/num_images * img_sum
        # apply basic histogram equalization
        img_sum = self.__std_histeq__(img_sum)
        #
        # path for current emotion
        #
        filename = "mask_" + str(emotion) + ".png"
        destination_path = os.path.join(outpath, filename)
        img_sum = cv2.resize(img_sum, outdim)
        cv2.imwrite(destination_path, img_sum)

    def applyAffineTransform(self, src, srcTri, dstTri, size):
        # Given a pair of triangles, find the affine transform.
        warpMat = cv2.getAffineTransform(numpy.float32(srcTri), numpy.float32(dstTri))

        # Apply the Affine Transform just found to the src image
        dst = cv2.warpAffine(src, warpMat, (size[0], size[1]), None,
                            flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REFLECT_101)
        return dst

    # Warps and alpha blends triangular regions from img1 and img2 to img
    def warpTriangle(self, img1, img2, t1, t2):
        # Find bounding rectangle for each triangle
        r1 = cv2.boundingRect(numpy.float32([t1]))
        r2 = cv2.boundingRect(numpy.float32([t2]))

        # Offset points by left top corner of the respective rectangles
        t1Rect, t2Rect, t2RectInt = [], [], []
        for i in range(0, 3):
            t1Rect.append(((t1[i][0] - r1[0]), (t1[i][1] - r1[1])))
            t2Rect.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))
            t2RectInt.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))

        # Get mask by filling triangle
        mask = numpy.zeros((r2[3], r2[2], 3), dtype=numpy.float32)
        cv2.fillConvexPoly(mask, numpy.int32(t2RectInt), (1.0, 1.0, 1.0), 16, 0)

        # Apply warpImage to small rectangular patches
        img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
        size = (r2[2], r2[3])
        img2Rect = self.applyAffineTransform(img1Rect, t1Rect, t2Rect, size)
        # img2Rect = img2Rect[0]
        img2Rect = numpy.array(img2Rect)
        # img2Rect = numpy.stack((img2Rect,) * 3, axis=-1)
        img2Rect = img2Rect * mask

        # Copy triangular region of the rectangular patch to the output image
        img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] * (
                    (1.0, 1.0, 1.0) - mask)
        img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] + img2Rect
        return img2

    def constrainPoint(self, p, w, h):
        p = (min(max(p[0], 0), w - 1),
             min(max(p[1], 0), h - 1))
        return p

    def calculateDelaunayTriangles(self, rect, points):
        # Insert points into subdiv
        subdiv = cv2.Subdiv2D(rect)
        for p in points:
            subdiv.insert((p[0], p[1]))

        # List of triangles. Each triangle is a list of 3 points ( 6 numbers )
        triangleList = subdiv.getTriangleList()

        # Find the indices of triangles in the points array
        delaunayTri = []
        for t in triangleList:
            pt = []
            pt.append((t[0], t[1]))
            pt.append((t[2], t[3]))
            pt.append((t[4], t[5]))
            pt1 = (t[0], t[1])
            pt2 = (t[2], t[3])
            pt3 = (t[4], t[5])
            if self.rectContains(rect, pt1) and self.rectContains(rect, pt2) and self.rectContains(rect, pt3):
                ind = []
                for j in range(0, 3):
                    for k in range(0, len(points)):
                        if (abs(pt[j][0] - points[k][0]) < 1.0 and abs(pt[j][1] - points[k][1]) < 1.0):
                            ind.append(k)
                if len(ind) == 3:
                    delaunayTri.append((ind[0], ind[1], ind[2]))
        return delaunayTri

    # Check if a point is inside a rectangle
    def rectContains(self, rect, point):
        if point[0] < rect[0]:
            return False
        if point[1] < rect[1]:
            return False
        if point[0] > rect[2]:
            return False
        if point[1] > rect[3]:
            return False
        return True

    def similarityTransform(self, inPoints, outPoints):
        """
        :param inpts: coordinates of the center of left and right eye
        :param outpts: destination coordinates of the center of left and right eye
        :return: affine transformation
        """
        s60 = math.sin(60 * math.pi / 180)
        c60 = math.cos(60 * math.pi / 180)
        inPts = numpy.copy(inPoints).tolist()
        outPts = numpy.copy(outPoints).tolist()

        xin = c60 * (inPts[0][0] - inPts[1][0]) - s60 * (inPts[0][1] - inPts[1][1]) + inPts[1][0]
        yin = s60 * (inPts[0][0] - inPts[1][0]) + c60 * (inPts[0][1] - inPts[1][1]) + inPts[1][1]
        inPts.append([numpy.int(xin), numpy.int(yin)])

        xout = c60 * (outPts[0][0] - outPts[1][0]) - s60 * (outPts[0][1] - outPts[1][1]) + outPts[1][0]
        yout = s60 * (outPts[0][0] - outPts[1][0]) + c60 * (outPts[0][1] - outPts[1][1]) + outPts[1][1]
        outPts.append([numpy.int(xout), numpy.int(yout)])

        tform = cv2.estimateAffinePartial2D(numpy.array([inPts]), numpy.array([outPts]))
        return cv2.UMat(tform[0])


def run():
    path = "../ckp/tensplit/"
    gtms = "../masks/"
    trainfiles = [os.path.join(path, "train_ids_{i}.csv".format(i=i)) for i in range(10)]
    foldernames = [os.path.join(gtms,"train_{i}".format(i=i)) for i in range(10)]
    emotions = [str(i) for i in range(1,8)]

    for splitId in range(len(trainfiles)):
        split = trainfiles[splitId]
        print(split)
        dstp = foldernames[splitId]
        try:
            os.mkdir(dstp)
        except OSError:
            print("Creation of the directory %s failed" % dstp)
        for j in [1, 2, 3, 4, 5, 6, 7]:
            GTMC = GTMCreator(trainset=split)
            GTMC.test(emotion=j, outdim=(299, 299), outpath=dstp)


run()

# #####################################################################
# 0. Create a 10 fold train test split and save to "../ckp/tensplit/"
# 1. Once create all masks with current dataset split
# 2. Pretrain the facial mask generator for each train_id/test_id
# 3. Train the whole FMPN on train_id
#    a. Load the corresponding FMG trained on train_id
# ######################################################################

"""
splitname0 = "../ckp/tensplit/train_ids_0.csv"
splitname1 = "../ckp/tensplit/train_ids_1.csv"

GTMC = GTMCreator(trainset=splitname1)
# GTMC.run(emotion=4)
name = "S005_001_00000009.png"
# GTMC.__draw_landmarks__(imagefilename=name)

for i in [1, 2, 3, 4, 5, 6, 7]:
    GTMC.test(emotion=i)
# GTMC.test2(emotion=4)
# GTMC.run(emotion=4)

#
# :TODO: write a main method and a parser and run from terminal
#


"""






#
#
# AffectNet https://github.com/amilkh/cs230-fer/blob/master/datasets/affectnet_to_png.ipynb
#
# https://github.com/amilkh/cs230-fer
#
#

#
# Nehme train_ids_0
# laufe durch
# prüfe ob es dazu ein label gibt
# wenn nein dann gehe zu nächstem Durchlauf
# wenn ja dann
#
#
#
# https://github.com/jrosebr1/imutils/blob/master/imutils/face_utils/facealigner.py
#
