## This is an extra data logger for images (might not be real time)
## Author : Avadesh Meduri
## Date : 8/04/2022

import numpy as np
import cv2
import os
import shutil

def ImageLogger(fields, file_name, log_duration, child_conn):
    print("Initialising Image Logger")
    fields = fields ## names of the data
    file_name = file_name ## name of the file
    log_duration = log_duration
    finished_logging = False
    log_data = {"time" : []}
    for i in range(len(fields)):
        log_data[fields[i]] = []

    color_path = file_name + '_rgb.avi'
    depth_path = file_name + '_depth.avi'
    str_dir = "/home/ameduri/pydevel/ioc_qp/vision/image_data/" + file_name
    isExist = os.path.exists(str_dir)
    if isExist:
        shutil.rmtree(str_dir, ignore_errors=True)
    os.mkdir(str_dir)
    f_id = 0
    print("Finished Initialising Image Logger")

    while True:
        curr_data, ti = child_conn.recv()
        if ti < 1000*log_duration: 
            if np.linalg.norm(curr_data["position"]) != 0:
                print(str(curr_data["position"]) + str("    ") + str((ti/1000)) + "/" + str(log_duration), end='\r', flush  = True)
                log_data["time"].append(ti)
                log_data["position"].append(curr_data["position"])

                cv2.imwrite(str_dir + "/color_" + str(f_id) + ".jpg", curr_data["color_image"])
                cv2.imwrite(str_dir + "/depth_" + str(f_id) + ".jpg", curr_data["depth_image"][:,:,None])
                f_id += 1
        elif not finished_logging:
            np.savez("/home/ameduri/pydevel/ioc_qp/vision/position_data/" + file_name + '.npz', log_data, **log_data)
            finished_logging = True
            print("Finished saving image data ........")
        else:
            pass