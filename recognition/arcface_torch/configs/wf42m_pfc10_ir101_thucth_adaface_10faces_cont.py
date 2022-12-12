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
config.embedding_size = 512
config.sample_rate = 1.0
config.fp16 = True
config.momentum = 0.9
config.weight_decay = 5e-4
config.batch_size = 64
config.lr = 0.004
config.verbose = 97331//2 # verify 2 times per epoch
config.dali = False # set False beacause of the dataset issue

config.finetune_full = True
# config.finetune_bb = "/mnt/data/weights/adaface/ms1mv2_ir100_thucth_adaface1/model.pt"


# config.output = "/mnt/data/weights/adaface/wf42m_pfc02_ir101_thucth_adaface_exp2_lowlr"
config.output = "/mnt/data/weights/adaface/wf42m10faces_pfc10_ir101_adaface_cont"

config.rec = "/mnt/data/webface42m10faces"
config.num_classes = 1141847
config.num_image = 37375382
config.num_epoch = 4
config.warmup_epoch = config.num_epoch/10
config.val_targets = ["lfw", "cfp_fp", "agedb_30"]
# config.val_targets = ["lfw"]

config.num_workers = 2
