from easydict import EasyDict as edict

# make training faster
# our RAM is 256G
# mount -t tmpfs -o size=140G  tmpfs /train_tmp

config = edict()

#loss define
config.head = "oldhead" # "adaface" or "oldhead" 
config.m = 0.4
config.h = 0.333
config.s = 64.
config.t_alpha = 0.01


config.margin_list = (1.0, 0.0, 0.4)



################################
# config.sample_rate = 0.00001
################################

#4gpus
# config.network = "r50" #90 GB MEM / 5370 GPU 0 / 4238 GPU 1,2,3
# config.network = "r100" #90 GB MEM / 5550 GPU 0 / 4584 GPU 1,2,3

#2gpus
# config.network = "r50" #48 GB MEM / 7414 GPU 0 / 6316 GPU 1
# config.network = "r100" #48 GB MEM / 7620 GPU 0 / 6662 GPU 1


################################
# config.sample_rate = 0.01
################################

#4gpus
# config.network = "r50" #90 GB MEM / 5340 GPU 0 / 4238 GPU 1,2,3
# config.network = "r100" #90 GB MEM / 5550 GPU 0 / 4584 GPU 1,2,3


################################
# config.sample_rate = 0.3
################################

#4gpus - batch_size = 2

# config.network = "r50" #90.3 GB MEM / 7118 GPU 0 / 6300 GPU 1,2,3
# config.network = "r100" #90.7 GB MEM / 7748 GPU 0 / 6992 GPU 1,2,3

#4gpus - batch_size = 8
# config.network = "r100" #90.7 GB MEM / 7692 GPU 0 / 7366 GPU 1,2,3

#4gpus - batch_size = 32
# config.network = "r100" #90.6 GB MEM / 7830 GPU 0 / 7918 GPU 1,2,3

#4gpus - batch_size = 64 
config.network = "r100" #91.4 GB MEM / 10808 GPU 0 / 10622 GPU 1,2,3
config.batch_size = 64


config.resume = False
config.output = None
config.embedding_size = 512  
config.sample_rate = 0.3
config.fp16 = True
config.momentum = 0.9
config.weight_decay = 5e-4
config.lr = 0.1
config.verbose = 100
config.dali = False # set False beacause of the dataset issue

config.rec = "/mnt/data/WebFace42M_shufrec"
config.num_classes = 2130631
config.num_image = 43003987

# config.rec = "/mnt/data/MS1MV2"
# config.num_classes = 85742
# config.num_image = 5822653


config.num_epoch = 20
config.warmup_epoch = 0
config.val_targets = ['lfw']
config.workers = 1
