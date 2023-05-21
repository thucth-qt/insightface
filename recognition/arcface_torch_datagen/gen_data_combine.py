from rec_builder import RecBuilder
from tqdm import tqdm 
import   glob
import numpy as np
import cv2 
from skimage import transform
import pandas as pd 
import os

builder = RecBuilder("/home/thucth/Biometrics/CombineData/MxRecord5Iden")

#=====================================================
#           INTERNAL DATA
#=====================================================
internal_data = pd.read_csv("/home/thucth/Biometrics/Internal/internal_aligned_363kIden_1306kImage.csv")
internal_data.sort_values(by=['cluster_id'], inplace=True)

print("total internal_data: ", len(internal_data))
current_label = 0.
current_clusterid = 0.
current_imgs = []
for idx in range(len(internal_data)):
# for idx in range(50):
    if idx%10000 == 0:
        print(idx)
    try:
        row = internal_data.iloc[idx]
        img_path = row['path']
        clusterid = float(row['cluster_id'])
        
        try:
            img = cv2.imread(img_path)
        except FileNotFoundError:
            print(f"File {img_path} not found.")
            continue
        except IOError:
            print(f"An error occurred while reading the file {img_path}.")
            continue
        except Exception as e:
            print("An unknown error occurred:", e)
            continue

        if img is None: continue
        
        if current_clusterid != clusterid:
            if len(current_imgs)>=5:
                #write to record
                builder.add(current_label, current_imgs)
                current_clusterid = clusterid
                current_imgs = []
                current_label+=1
            else:
                current_clusterid = clusterid
                current_imgs = []
                continue
        
        _, buffer = cv2.imencode('.jpg', img)
        str_image = buffer.tobytes()
        # with open(img_path, 'rb') as fp:
        #     str_image = fp.read()
        if len(str_image) <= 0:
            continue
        current_imgs.append(str_image)
    except Exception as e:
        print("===========================")
        print(e)
        print(img_path)
        print(idx)
        print("===========================")
	
# let part2 - webface do this modification 
# if len(current_imgs)>=5:
#     #write to record
#     builder.add(current_label, current_imgs)
#     current_clusterid = clusterid
#     current_imgs = []
#     current_label+=1
    
#=====================================================
#           WEBFACE DATA
#=====================================================
webface_data = pd.read_csv("/home/thucth/Biometrics/Webface/webface_42M.csv")
webface_data.sort_values(by=['cluster_id'], inplace=True)

print("total webface: ", len(webface_data))

for idx in range(len(webface_data)):
# for idx in range(50):
    if idx%10000 == 0:
        print(idx)
    try:
        row = webface_data.iloc[idx]
        img_path = row['path']
        clusterid = float(row['cluster_id'])
        
        try:
            img = cv2.imread(img_path)
        except FileNotFoundError:
            print(f"File {img_path} not found.")
            continue
        except IOError:
            print(f"An error occurred while reading the file {img_path}.")
            continue
        except Exception as e:
            print("An unknown error occurred:", e)
            continue

        if img is None: continue
        
        if current_clusterid != clusterid:
            if len(current_imgs)>=5:
                #write to record
                builder.add(current_label, current_imgs)
                current_clusterid = clusterid
                current_imgs = []
                current_label+=1
            else:
                current_clusterid = clusterid
                current_imgs = []
                continue
        
        _, buffer = cv2.imencode('.jpg', img)
        str_image = buffer.tobytes()
        if len(str_image) <= 0:
            continue
        current_imgs.append(str_image)
    except Exception as e:
        print("===========================")
        print(e)
        print(img_path)
        print(idx)
        print("===========================")
	
if len(current_imgs)>=5:
    #write to record
    builder.add(current_label, current_imgs)
    current_clusterid = clusterid
    current_imgs = []
    current_label+=1

builder.close()