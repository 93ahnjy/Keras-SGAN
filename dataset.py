import csv
import os
import cv2
import glob
import random


AVALIABLE_IMAGE_FORMAT_LIST = ['.png', '.jpg', '.jpeg', '.bmp', '.tif']


def get_image_list(dirname):

    filelist = os.listdir(dirname)
    imglist  = []

    for filename in filelist:
        if filename.endswith(".bmp"):
            imglist.append(filename)

    return random.shuffle(imglist)




















def make_our_csv(csvpath, datapath, csvname):


    dirname     = os.path.dirname(csvpath)
    new_csvpath = os.path.join(dirname, csvname)

    f  = open(csvpath, 'r', encoding='utf-8')
    wf = open(new_csvpath, 'w', encoding = 'utf-8')

    rdr = csv.reader(f)
    wr  = csv.writer(wf)

    for n, line in enumerate(rdr):
        endnum = line[0].rfind("\\") + 1
        new_csv = []
        new_csv.append(os.path.join(datapath, line[0][endnum:]))
        for i in range(1,6):
            new_csv.append(line[i])
        wr.writerow(new_csv)

    f.close()
    wf.close()



def check_csv(csvpath):
    f = open(os.path.join(csvpath), 'r', encoding='utf-8')
    rdr = csv.reader(f)
    for n, line in enumerate(rdr):
        print(line)
    f.close()






def visualizeBB(csvpath, result_savedir):      #일단하나씩구현

    f = open(csvpath)

    if not os.path.exists(result_savedir):
        os.mkdir(result_savedir)

    rdf = csv.reader(f)
    prev_line = " "



    for n, line in enumerate(rdf):
        endnum = line[0].rfind("/") + 1


        if prev_line != line[0]:
            org_img = cv2.imread(line[0])  # H x W x C
        else:
            org_img = cv2.imread(os.path.join(result_savedir, line[0][endnum:]))


        prev_line = line[0]
        BB = [int(line[1]), int(line[2]), int(line[3]), int(line[4]), line[5]]   # [xmin, ymin, xmax, ymax] + state


        if line[5] == 'normal':
            cv2.rectangle(org_img, (BB[0], BB[1]),(BB[2], BB[3]), (0,0,255), 3)
            cv2.putText(org_img, BB[4], (BB[2], BB[1]), cv2.FONT_ITALIC,1,(0,0,255), 1, cv2.LINE_8)
        else:
            cv2.rectangle(org_img, (BB[0], BB[1]),(BB[2], BB[3]), (255,0,0), 3)
            cv2.putText(org_img, BB[4], (BB[2], BB[1]), cv2.FONT_ITALIC,1,(255,0,0), 1, cv2.LINE_8)


        cv2.imwrite(os.path.join(result_savedir,line[0][endnum:]), org_img)
    f.close()






def Crop_Inspection(csvpath, savedir, size = -1):
    f = open(csvpath)
    rdf = csv.reader(f)
    size = int(size/2)


    if not os.path.exists(os.path.join(savedir,"normal_gray")):
        os.mkdir(os.path.join(savedir,"normal_gray"))
    if not os.path.exists(os.path.join(savedir,"fault_gray")):
        os.mkdir(os.path.join(savedir,"fault_gray"))


    for n, line in enumerate(rdf):
        endnum    = line[0].rfind("/") + 1
        BB        = [int(line[1]), int(line[2]), int(line[3]), int(line[4]), line[5]]   # [xmin, ymin, xmax, ymax, state]
        xy_len    = [round((BB[2]-BB[0])*1.5), round((BB[3] - BB[1])*1.5)]
        xy_center = [round((BB[2]+BB[0])/2), round((BB[3] + BB[1])/2)]
        org_img = cv2.imread(line[0])  # H x W x C



        if size == -1:  # crop with same size of bbox
            crop_img = org_img[xy_center[1] - xy_len[1] : xy_len[1] + xy_center[1], xy_center[0] - xy_len[0] : xy_len[0] + xy_center[0], :]
        else:
            crop_img = org_img[xy_center[1] - size : size + xy_center[1], xy_center[0] - size : size + xy_center[0], :]


        if line[5] == 'normal':
            # cv2.rectangle(crop_img, (round((xy_len[0]/3)), round((xy_len[1]/3))), (round(5*(xy_len[0]/3)), round(5*(xy_len[1]/3))), (0, 0, 255), 3)
            cv2.imwrite(os.path.join(savedir,"normal_gray",line[0][endnum:]), cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY))
        else:
            # cv2.rectangle(crop_img,(round((xy_len[0]/3)), round((xy_len[1]/3))), (round(5*(xy_len[0]/3)), round(5*(xy_len[1]/3))), (255, 0, 0), 3)
            cv2.imwrite(os.path.join(savedir, "fault_gray", line[0][endnum:]), cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY))
    f.close()





if __name__ == '__main__':
    root_path = os.path.abspath(".")                    # csv folder path

    csv_fol_name = ""                                      # csv folder name
    data_fol_name = "Test"                                 # dataset이 존재하는 folder name
    org_csv_name = "test.csv"                              # 바꾸고자 하는 csv file name
    our_csv_name = "Our_test.csv"                          # 만들고자 하는 csv file name
    visualized_folder = "test_visual"
    crop_in_fol = "crop_image"



    # 경로를 우리 컴퓨터의 경로로.
    #train_csvpath   = "/home/ahnjaeyoung/PycharmProjects/Surface_defect_detection/Surface_defect_dataset/train.csv"
    #train_data_path = "/home/ahnjaeyoung/PycharmProjects/Surface_defect_detection/Surface_defect_dataset/Train"

    #test_csvpath   = "/home/ahnjaeyoung/PycharmProjects/Surface_defect_detection/Surface_defect_dataset/test.csv"
    #test_data_path = "/home/ahnjaeyoung/PycharmProjects/Surface_defect_detection/Surface_defect_dataset/Test"

    #make_our_csv(train_csvpath, train_data_path, "Our_train.csv")
    #make_our_csv(test_csvpath,  test_data_path,  "Our_test.csv")


    # dataset의 시각화.
    our_csvpath    = "/home/ahnjaeyoung/PycharmProjects/Surface_defect_detection/Surface_defect_dataset/Our_test.csv"
    result_savedir = "/home/ahnjaeyoung/PycharmProjects/Surface_defect_detection/Surface_defect_dataset/Test_visualize"

    #visualizeBB(our_csvpath, result_savedir)

    #train_csvpath   = "/home/ahnjaeyoung/PycharmProjects/Surface_defect_detection/Surface_defect_dataset/Our_train.csv"
    #train_crop_path = "/home/ahnjaeyoung/PycharmProjects/Surface_defect_detection/Surface_defect_dataset/Train_crop"
    #Crop_Inspection(train_csvpath, train_crop_path, size=128)




    # crop 만들어 저장.
    train_csvpath   = "/home/ahnjaeyoung/PycharmProjects/Surface_defect_detection/Surface_defect_dataset/Our_train.csv"
    train_crop_path = "/home/ahnjaeyoung/PycharmProjects/Surface_defect_detection/Surface_defect_dataset/Train_crop"
    Crop_Inspection(train_csvpath, train_crop_path, size=128)

    test_csvpath   = "/home/ahnjaeyoung/PycharmProjects/Surface_defect_detection/Surface_defect_dataset/Our_test.csv"
    test_crop_path = "/home/ahnjaeyoung/PycharmProjects/Surface_defect_detection/Surface_defect_dataset/Test_crop"
    Crop_Inspection(test_csvpath, test_crop_path, size=128)