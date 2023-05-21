from easydict import EasyDict as edict

config = edict()

#loss define
config.head = "adaface" # "adaface" or "oldhead" 
config.original_margin = False
config.m = 0.4
config.h = 0.333
config.s = 64.
config.t_alpha = 0.01


config.margin_list = (1.0, 0.0, 0.4)
config.network = "ir_200"
# config.network = "ir_101"


config.resume = False
config.embedding_size = 512
config.sample_rate = 1.0
config.fp16 = True
config.momentum = 0.9
config.weight_decay = 5e-4
config.batch_size = 32
config.lr = 0.008
config.verbose = 143621 # verify each epoch
config.frequent = 10000
config.dali = False # set False beacause of the dataset issue

config.finetune_full = False
config.finetune_bb = ""


config.output = "/home/thucth/Biometrics/insightface/recognition/arcface_torch/work_dirs/combine_pfc10_ir201_adaface_5gnn"

config.rec = "/home/thucth/Biometrics/CombineData/MxRecord5Iden"
config.num_classes = 1757870
config.num_image = 41362759
config.num_epoch = 10
config.warmup_epoch = config.num_epoch//10
config.val_targets = ["lfw", "cfp_fp", "agedb_30"]

config.num_workers = 2
