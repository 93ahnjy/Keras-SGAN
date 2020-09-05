import tensorflow
import cv2
import numpy as np
import os
import itertools



"""
opencv 이용해서 labeling 후 bbox 그리기.
"""

def component_labeling(image, nlabels, stats, centroids):


    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    for i in range(nlabels):
        if i < 2:
            continue

        area = stats[i, cv2.CC_STAT_AREA]
        center_x = int(centroids[i, 0])
        center_y = int(centroids[i, 1])
        left = stats[i, cv2.CC_STAT_LEFT]
        top = stats[i, cv2.CC_STAT_TOP]
        width = stats[i, cv2.CC_STAT_WIDTH]
        height = stats[i, cv2.CC_STAT_HEIGHT]

        cv2.rectangle(image, (left, top), (left + width, top + height),
                      (0, 0, 255), 3)

    return image





def surface_detect(image, visualize=False):


    _ , th1 = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((5, 5), np.uint8)
    th1 = cv2.erode(th1, kernel, iterations=1)



    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(th1)
    #th1 = component_labeling(th1, nlabels, stats, centroids)


    x1 = stats[:, cv2.CC_STAT_LEFT]
    y1 = stats[:, cv2.CC_STAT_TOP]
    x2 = stats[:, cv2.CC_STAT_LEFT] + stats[:, cv2.CC_STAT_WIDTH]
    y2 = stats[:, cv2.CC_STAT_TOP] + stats[:, cv2.CC_STAT_HEIGHT]


    x1 = np.min(x1[x1 != 0]) -100
    y1 = np.min(y1[y1 != 0]) - 50
    x2 = np.max(x2[x2 != image.shape[1]]) + 100
    y2 = np.max(y2[y2 != image.shape[0]]) + 50


    if visualize:
        image_vis = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        cv2.rectangle(image_vis, (x1, y1), (x2, y2), (0, 255, 0), 3)

        image_vis = cv2.resize(image_vis, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)

        cv2.imshow("area detect", image_vis)
        cv2.waitKey(0)

    return image[y1:y2, x1:x2], (x1, y1), (x2, y2)





def crop_image(image, p_size, overlap_ratio, visualize=False):
    """
    :param image: 입력 이미지
    :param crop_size:
    :param overlap_ratio:
    :return:
    """

    img_H, img_W   = image.shape
    pH, pW         = p_size
    cropped_images = []


    x1_vec = [i for i in range(0, img_W - pW // 2, int((1 - overlap_ratio) * pW))]
    y1_vec = [i for i in range(0, img_H - pH // 2, int((1 - overlap_ratio) * pH))]


    i = 0
    for x1, y1 in itertools.product(x1_vec, y1_vec):
        patch = image[y1:y1 + pH, x1:x1 + pW]
        patch = cv2.resize(patch, (512, 512), interpolation=cv2.INTER_AREA)
        cropped_images.append(patch)
        # cv2.imwrite('patch_example_{}.png'.format(str(i)), patch)
        i += 1



    if visualize:
        if image.ndim < 3:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        for x1, y1 in itertools.product(x1_vec, y1_vec):
            cv2.rectangle(image, (x1, y1), (x1 + pW, y1 + pH), (0,255,0), 3)

        image = cv2.resize(image, None, fx=0.15, fy=0.15, interpolation=cv2.INTER_AREA)
        cv2.imshow("cropped area", image)
        cv2.waitKey(0)



    return cropped_images











def crop_surface(image):

    surface, p1, p2 = surface_detect(image, True)
    print(surface.shape)

    cropped_surfaces= crop_image(surface,
                               p_size=(surface.shape[0], 500),
                               overlap_ratio=0.2,
                               visualize=True)


    for i, x in enumerate(cropped_surfaces):
        print(x.shape)
        cv2.imshow("cropped areas_{}".format(str(i)), x)
        cv2.waitKey(0)


    return np.stack(cropped_surfaces, axis=0).shape
































if __name__ == '__main__':

    image_dir  = "/home/ahnjaeyoung/PycharmProjects/Surface_defect_detection/Surface_defect_dataset/Train/20190620_RawData.20190525.master_image.20190524.3.9_LEFT1_B_SCAN_IMG004_src.bmp"

    image = cv2.imread(image_dir, cv2.IMREAD_GRAYSCALE)

    # ESC 키 눌러서 다음 그림으로 넘어가기.
    dataset = crop_surface(image)
