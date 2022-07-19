from easydict import EasyDict as edict

# make training faster
# our RAM is 256G
# mount -t tmpfs -o size=140G  tmpfs /train_tmp

config = edict()

#loss define
config.head = "oldhead"  # "adaface" or "oldhead"
config.m = 0.4
config.h = 0.333
config.s = 64.
config.t_alpha = 1.0


#old loss for  CombinedMarginLoss (arcface and cosface) (author)
config.margin_list = (1.0, 0.5, 0.0)


config.network = "ir_50"
config.resume = False
config.output = None
config.embedding_size = 512
config.sample_rate = 0.1
config.fp16 = True
config.momentum = 0.9
config.weight_decay = 5e-4
config.batch_size = 128
config.lr = 0.1
config.verbose = 2000
config.dali = False # set False beacause of the dataset issue

config.rec = "/share/team/thucth/data/FaceReg/faces_emore"
config.num_classes = 85742
config.num_image = 5822653
config.num_epoch = 20
config.warmup_epoch = 0
config.val_targets = ['lfw', 'cfp_fp', "agedb_30"]
