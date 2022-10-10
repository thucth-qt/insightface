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
config.network = "ir_101"


config.resume = False
config.output = None
config.embedding_size = 512
config.sample_rate = 0.2
config.fp16 = True
config.momentum = 0.9
config.weight_decay = 5e-4
config.batch_size = 64
config.lr = 0.4
config.verbose = 67193 # verify 2 times per epoch
config.dali = False # set False beacause of the dataset issue

config.rec = "/mnt/data/WebFace42M_shufrec"
config.num_classes = 2130631
config.num_image = 43003987
config.num_epoch = 20
config.warmup_epoch = config.num_epoch//10
config.val_targets = ["lfw", "cfp_fp", "agedb_30"]
# config.val_targets = [ "agedb_30", "calfw", "cfp_ff", "cfp_fp", "lfw", "vgg2_fp"]
# config.val_targets = ["lfw"]

config.num_workers = 2
