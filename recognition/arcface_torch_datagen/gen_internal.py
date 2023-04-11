from rec_builder import RecBuilder
from tqdm import tqdm 
import   glob
import pandas as pd
import cv2

internal_data = pd.read_csv("/home/thucth/Biometrics/Internal/internal_data_available_2identitiesfiltered_1M.csv")
internal_data.head()

builder = RecBuilder("/home/thucth/Biometrics/Internal/MxRecord")

current_label = 0.
current_clusterid = 0.
current_imgs = []
print("total: ", len(internal_data))
# for idx in range(len(internal_data)):
for idx in tqdm(range(len(internal_data))):
# for idx in tqdm(range(500)):
    # if idx%10000 == 0:
    # if idx%100 == 0:
        # print(idx)
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
            if len(current_imgs)>=2:
                #write to record
                builder.add(current_label, current_imgs)
                current_clusterid = clusterid
                current_imgs = []
                current_label+=1
            else:
                print("current_img less images than 2")
                print(clusterid)
                current_clusterid = clusterid
                current_imgs = []
                continue
        
        face = img 

        _, buffer = cv2.imencode('.jpg', face)
        str_image = buffer.tobytes()
        if len(str_image) <= 0:
            continue
        current_imgs.append(str_image)
    except Exception as e:
        print("===========================")
        print(e)
        print(img_path)
        print(idx)
        # import pdb; pdb.set_trace()
        print("===========================")
        pass
    
if len(current_imgs)>=2:
    #write to record
    builder.add(current_label, current_imgs)
    current_clusterid = clusterid

builder.close()
