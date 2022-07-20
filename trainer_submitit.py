""" BigGAN: The Authorized Unofficial PyTorch release
    Code by A. Brock and A. Andonian
    This code is an unofficial reimplementation of
    "Large-Scale GAN Training for High Fidelity Natural Image Synthesis,"
    by A. Brock, J. Donahue, and K. Simonyan (arXiv 1809.11096).

    Let's go.
"""

from train_for_submitit import main_subm
from submitit.helpers import Checkpointable
import submitit
from argparse import ArgumentParser
import argparse



class Trainer(Checkpointable):
    def __call__(self,args):
     main_subm(args)




if __name__ == "__main__":
    device = "cuda"
    parser = argparse.ArgumentParser(description="StyleGAN2 trainer")
    parser.add_argument("--slurm", action="store_true")

    parser.add_argument(
        "--gpus", type=int, default=8, help="image sizes for the model"
    )

    parser.add_argument("path", type=str, help="path to the lmdb dataset")
    parser.add_argument("--val_path", type=str, help="path to the lmdb eval dataset")
    parser.add_argument('--arch', type=str, default='stylegan2', help='model architectures (stylegan2 | swagan)')
    parser.add_argument(
        "--iter", type=int, default=800000, help="total training iterations"
    )
    parser.add_argument(
        "--batch", type=int, default=16, help="batch sizes for each gpus"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default='.',
        help="path to the output dir",
    )
    parser.add_argument(
        "--n_sample",
        type=int,
        default=64,
        help="number of the samples generated during training",
    )
    parser.add_argument(
        "--size", type=int, default=256, help="image sizes for the model"
    )
    parser.add_argument(
        "--size_ratio", type=int, default=1, help="image size ratio for the model"
    )
    parser.add_argument(
        "--r1", type=float, default=10, help="weight of the r1 regularization"
    )
    parser.add_argument(
        "--path_regularize",
        type=float,
        default=2,
        help="weight of the path length regularization",
    )
    parser.add_argument(
        "--path_batch_shrink",
        type=int,
        default=2,
        help="batch size reducing factor for the path length regularization (reduce memory consumption)",
    )
    parser.add_argument(
        "--d_reg_every",
        type=int,
        default=16,
        help="interval of the applying r1 regularization",
    )
    parser.add_argument(
        "--g_reg_every",
        type=int,
        default=4,
        help="interval of the applying path length regularization",
    )
    parser.add_argument(
        "--mixing", type=float, default=0.9, help="probability of latent code mixing"
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="path to the checkpoints to resume training",
    )

    parser.add_argument("--lr", type=float, default=0.002, help="learning rate")
    parser.add_argument(
        "--channel_multiplier",
        type=int,
        default=2,
        help="channel multiplier factor for the model. config-f = 2, else = 1",
    )

    parser.add_argument(
        "--latent",
        type=int,
        default=512,
        help="channel multiplier factor for the model. config-f = 2, else = 1",
    )
    parser.add_argument(
        "--wandb", action="store_true", help="use weights and biases logging"
    )
    parser.add_argument(
        "--local_rank", type=int, default=0, help="local rank for distributed training"
    )
    parser.add_argument(
        "--augment", action="store_true", help="apply non leaking augmentation"
    )
    parser.add_argument(
        "--augment_p",
        type=float,
        default=0,
        help="probability of applying augmentation. 0 = use adaptive augmentation",
    )
    parser.add_argument(
        "--ada_target",
        type=float,
        default=0.6,
        help="target augmentation probability for adaptive augmentation",
    )
    parser.add_argument(
        "--ada_length",
        type=int,
        default=500 * 1000,
        help="target duraing to reach augmentation probability for adaptive augmentation",
    )
    parser.add_argument(
        "--ada_every",
        type=int,
        default=256,
        help="probability update interval of the adaptive augmentation",
    )

    args = parser.parse_args()


    trainer = Trainer()

    if not args.slurm:
        trainer(args)

    else:
        executor = submitit.SlurmExecutor(
            folder='/checkpoint/marlenec/stylegan2PyTorch_submitit_logs',
            max_num_timeout=60)
        
        partition = 'learnlab'

        executor.update_parameters(
            gpus_per_node=args.gpus, partition=partition, constraint='volta32gb', comment='cvpr, 24/11',
            nodes=1,
            cpus_per_task=args.gpus *10,
            # mem=256000,
            time=4320, job_name=args.output_dir,
            exclusive=True)
        job = executor.submit(trainer, args)
        print(job.job_id)

        # with open('/checkpoint/marlenec/StyleMapGAN/log_submitit.txt', "a") as log_file:
        #     log_file.write(f'Exp name : {args.dataset},  {args.exp_name}, job id : {job.job_id}  \n')
        # print(job.job_id)
        #
        import time
        time.sleep(1)
