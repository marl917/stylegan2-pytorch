import json

def sbatch(cmd, partition='learnfair', gpus=1, nodes=1, time='72:00:00'):
    return ''.join(['sbatch',
        f' --partition={partition}',
        f' --nodes={nodes}',
        f' --gpus-per-node={gpus}',
        f' --cpus-per-task={gpus * 10}',
        f' --mem={gpus * 60}G',
        f' --time={time}',
        f' --signal=USR1@60',
        f' --open-mode=append',
        f' --wrap={json.dumps("srun --label " + cmd)}'])

# print(sbatch('python -m torch.distributed.launch --nproc_per_node=8 --master_port=1234 train.py --batch 16 /checkpoint/marlenec/cityscapes_128x256/ --size 128 --output_dir /checkpoint/marlenec/cityscapes128x256_stylegan2Torch --size_ratio 2', gpus=8))
# print(sbatch('python -m torch.distributed.launch --nproc_per_node=8 --master_port=1234 train.py --batch 16 /checkpoint/marlenec/celebAHQ_256/ --size 256 --output_dir /checkpoint/marlenec/celeba256_stylegan2Torch', gpus=8))
for i in range(1,13):
    print(sbatch(f'python projectorAutoDistributed.py --ckpt /checkpoint/marlenec/stylegan2_ckpt/celeba256_trainingSet-stylegan2Torch/checkpoint/060000.pt  --nameExp celeba256_allTrainImgs_wPlus{i} --size 256 /checkpoint/marlenec/CelebAMask-HQ/training_set_img/{i} --batch 4 --step 5000 --w_plus --mse 0.00001', gpus=8, partition = 'learnlab'))