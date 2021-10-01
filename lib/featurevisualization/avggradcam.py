import glob
import numpy as np
import cv2
from matplotlib import pyplot as plt


def create_avg_gradcam(path, final_imgname):
    """create average gradcam image of a class"""
    files = glob.glob(path + "*.png")

    cams = []

    for cam_path in files:
        tmp_cam = cv2.imread(cam_path, 1)
        cams.append(tmp_cam)

    cam_avg = cams[0]

    for i in range(len(cams)):
        if i == 0:
            pass
        else:
            alpha = 1.0 / (i + 1)
            beta = 1.0 - alpha
            cam_avg = cv2.addWeighted(cams[i], alpha, cam_avg, beta, 0.0)

    cv2.imwrite(final_imgname, cam_avg)

    #  plt.imshow(cam_avg)
    #  plt.show()


if __name__ == '__main__':
    #
    # create folders "class_0, class_1, ..., class_6" and
    # inside an "vis" folder. Place the visualization image inside the vis folder.
    # Do this for all classes of the current layer.
    #
    em = ['anger', 'contempt', 'disgust', 'fear', 'happy', 'sadness', 'surprise']
    classes = [0, 1, 2, 3, 4, 5, 6]
    for i in classes:
        parentfolder = f"C:/root/uni/bachelor\gradcams\gradcam_imgs_incv_14072021_fold8\layer-16\class_{i}/vis/"
        final_imgname = f"C:/root/uni/bachelor\gradcams\gradcam_imgs_incv_14072021_fold8\layer-16/avg_gcam_layer-16_{em[i]}.png"
        create_avg_gradcam(parentfolder, final_imgname)
