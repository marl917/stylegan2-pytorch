# python prepare_data_bigData.py --size 256 --out /checkpoint/marlenec/lmbd_datasets/ade20k_256x256_50%TrainData /datasets01/ADE20kChallengeData2016/011719/images/training --perctg 0.5

python prepare_data_bigData.py --size 256 --out /checkpoint/marlenec/lmbd_datasets/cocostuff_256x256_50%TrainData /checkpoint/marlenec/cocostuff/train_img/ --perctg 0.5

python prepare_data_bigData.py --size 256 --out /checkpoint/marlenec/lmbd_datasets/cocostuff_256x256_20%TrainData /checkpoint/marlenec/cocostuff/train_img/ --perctg 0.2

python prepare_data_bigData.py --size 256 --out /checkpoint/marlenec/lmbd_datasets/cocostuff_256x256_10%TrainData /checkpoint/marlenec/cocostuff/train_img/ --perctg 0.1

python prepare_data_bigData.py --size 256 --out /checkpoint/marlenec/lmbd_datasets/cocostuff_256x256_5%TrainData /checkpoint/marlenec/cocostuff/train_img/ --perctg 0.05