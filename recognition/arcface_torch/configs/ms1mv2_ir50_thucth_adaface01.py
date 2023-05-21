from easydict import EasyDict as edict

# make training faster
# our RAM is 256G
# mount -t tmpfs -o size=140G  tmpfs /train_tmp

config = edict()

#loss define
config.head = "adaface" # "adaface" or "oldhead" 
config.m = 0.4
config.h = 0.333
config.s = 64.
config.t_alpha = 0.01

#old loss for  CombinedMarginLoss (arcface and cosface) (author)
config.margin_list = (1.0, 0.5, 0.0)


config.network = "ir_50"
config.resume = False
config.save_all_states = True
config.output = None
config.embedding_size = 512
config.sample_rate = 0.1
config.fp16 = True
config.momentum = 0.9
config.weight_decay = 5e-4
config.batch_size = 96
config.lr = 0.1
config.verbose = 5000
config.dali = False # set False beacause of the dataset issue

config.rec = "/mnt/data/MS1MV2"
config.num_classes = 85742
config.num_image = 5822653
config.num_epoch = 25
config.warmup_epoch = 0
# config.val_targets = [ "agedb_30", "calfw", "cfp_ff", "cfp_fp", "lfw", "vgg2_fp"]
# config.val_targets = ["agedb_30", "cfp_fp", "lfw"] #load 1 bin for quick debugging
config.val_targets = ["agedb_30"] #load 1 bin for quick debugging

# dataload numworkers
config.num_workers = 2