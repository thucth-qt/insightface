# %%
from rec_builder import RecBuilder
from tqdm import tqdm 
import   glob
# %%
builder = RecBuilder("/share/team/thucth/data/FaceReg/webface42m_rec_10faces")
# builder = RecBuilder("./sample_target/WebFace42M/")

# %%
import os

label = 0.
source = "/share/team/nhatnhm4/WebFace42M/WebFace260M/"
# source = "./sample_data/WebFace260M/"
data_folder = os.listdir(source)
data_folder.sort()
for cluster_id in tqdm(data_folder):
    imgs = []
    image_paths = glob.glob(os.path.join(source, cluster_id, '*'))
    if len(image_paths) < 10:
        # print(len(image_paths))
        continue
    for img_path in image_paths:
        with open(img_path, 'rb') as fp:
            str_image = fp.read()
        if len(str_image) <= 0:
            continue
        imgs.append(str_image)

    builder.add(label, imgs)
    label += 1.


# source = "/share/team/nhatnhm4/WebFace42M/selected_train"
# # source = "./sample_data/Internal"
# data_folder = os.listdir(source)
# data_folder.sort()
# for cluster_id in tqdm(data_folder):
#     imgs = []
#     image_paths = glob.glob(os.path.join(source, cluster_id, '*'))

#     for img_path in image_paths:
#         with open(img_path, 'rb') as fp:
#             str_image = fp.read()
#         if len(str_image) <= 0:
#             continue
#         imgs.append(str_image)

#     builder.add(label, imgs)
#     label += 1.

# %%
builder.close()