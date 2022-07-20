import argparse
import math
import os
import sys
import torch
from torch import optim
from torch.nn import functional as F
from torchvision import transforms, utils
from PIL import Image
from tqdm import tqdm
from torch.utils import data
from dataset import ProjectorDataset
import signal

from distributed import (
    get_rank,
    synchronize,
    reduce_loss_dict,
    reduce_sum,
    get_world_size,
)

import lpips
from model import Generator

def sample_data(loader):
    while True:
        for batch in loader:
            yield batch


def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)


def noise_regularize(noises):
    loss = 0

    for noise in noises:
        size = noise.shape[2]

        while True:
            loss = (
                loss
                + (noise * torch.roll(noise, shifts=1, dims=3)).mean().pow(2)
                + (noise * torch.roll(noise, shifts=1, dims=2)).mean().pow(2)
            )

            if size <= 8:
                break

            noise = noise.reshape([-1, 1, size // 2, 2, size // 2, 2])
            noise = noise.mean([3, 5])
            size //= 2

    return loss


def noise_normalize_(noises):
    for noise in noises:
        mean = noise.mean()
        std = noise.std()

        noise.data.add_(-mean).div_(std)


def get_lr(t, initial_lr, rampdown=0.25, rampup=0.05):
    lr_ramp = min(1, (1 - t) / rampdown)
    lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
    lr_ramp = lr_ramp * min(1, t / rampup)

    return initial_lr * lr_ramp


def latent_noise(latent, strength):
    noise = torch.randn_like(latent) * strength

    return latent + noise


def make_image(tensor):
    return (
        tensor.detach()
        .clamp_(min=-1, max=1)
        .add(1)
        .div_(2)
        .mul(255)
        .type(torch.uint8)
        .permute(0, 2, 3, 1)
        .to("cpu")
        .numpy()
    )

def train(args, g_ema, loader, project_batch, percept, latent_mean, latent_std,out_filename, device):
    result_file = {}
    nb_pt=0
    if get_rank()==0:
        print('VALUE OF PROJECT BATCH', project_batch)
    loader = sample_data(loader)

    delattr(g_ema.module, 'style')

    noises_single = g_ema.module.make_noise(size_ratio=args.size_ratio)
    # nb_files = len(os.listdir(args.dir))
    # print(nb_files)
    for i in range(0,args.lenDataset, project_batch):
        if len(result_file)>=100:
            print('save pt file', 'name =',  f'allProjectedImg{nb_pt}_{get_rank()}.pt')
            torch.save(result_file, os.path.join(out_filename, f'allProjectedImg{nb_pt}_{get_rank()}.pt'))
            nb_pt+=1
            result_file = {}
        ##################################################################### Select project batch of images : currently trained images form dataset, of size project_batch or less
        if get_rank()==0:
            print('length of result_file', len(result_file))
            print(f'[Training for images in index [{i},{i+project_batch}]]')

        ##################################################################### Distribute images across gpus (shoud be size args.batch or less)
        train_imgs, nameToSave = next(loader)
        train_imgs = train_imgs.to(device)
        if get_rank()==0:
            print('after next loader, in pbar : size of train_imgs :', train_imgs.size())

        ##################################################################### Create noises and latent_in vectors (input to stylegan2) should be of same size astrain_imgs.size(0)
        noises = []
        for noise in noises_single:
            noises.append(noise.repeat(train_imgs.size(0), 1, 1, 1).normal_())

        latent_in = latent_mean.detach().clone().unsqueeze(0).repeat(train_imgs.size(0), 1)

        if args.w_plus:
            latent_in = latent_in.unsqueeze(1).repeat(1, g_ema.module.n_latent, 1)

        ##################################################################### Create optimizer to optimize latent_in and noise
        latent_in.requires_grad = True

        for noise in noises:
            noise.requires_grad = True

        optimizer = optim.Adam([latent_in] + noises, lr=args.lr)

        ##################################################################### Create pbar
        pbar = range(args.step)
        if get_rank() == 0:
            pbar = tqdm(pbar, initial=0, dynamic_ncols=True, smoothing=0.01)

        latent_path = []

        for k in pbar:
            t = k / args.step
            lr = get_lr(t, args.lr)
            optimizer.param_groups[0]["lr"] = lr
            noise_strength = latent_std * args.noise * max(0, 1 - t / args.noise_ramp) ** 2
            latent_n = latent_noise(latent_in, noise_strength.item())

            img_gen = g_ema([latent_n], input_is_latent=True, noise=noises, return_onlyImg=True)

            batch, channel, height, width = img_gen.shape

            if height > 256:
                factor = height // 256

                img_gen = img_gen.reshape(
                    batch, channel, height // factor, factor, width // factor, factor
                )
                img_gen = img_gen.mean([3, 5])

            p_loss = percept(img_gen, train_imgs).sum()
            n_loss = noise_regularize(noises)
            mse_loss = F.mse_loss(img_gen, train_imgs)

            loss = p_loss + args.noise_regularize * n_loss + args.mse * mse_loss

            optimizer.zero_grad()
            loss.backward()

            # for name, param in g_ema.module.named_parameters():
            #     if param.grad is None:
            #         print(name)

            optimizer.step()

            noise_normalize_(noises)

            if (k + 1) % 100 == 0:
                latent_path.append(latent_in.detach().clone())

            if get_rank() == 0:
                pbar.set_description(
                    (
                        f"perceptual: {p_loss.item():.4f}; noise regularize: {n_loss.item():.4f};"
                        f" mse: {mse_loss.item():.4f}; lr: {lr:.4f}"
                    )
                )
            # if k%100==0:
            #     print(loss)

        img_gen, _ = g_ema([latent_path[-1]], input_is_latent=True, noise=noises)


        utils.save_image(
            img_gen,
            os.path.join('/checkpoint/marlenec/projector', args.nameExp, f'gpu{get_rank()}projected_img.png'),
            nrow=8,
            normalize=True,
            range=(-1, 1),
        )

        for l, input_name in enumerate(nameToSave):
            noise_single = []
            for noise in noises:
                noise_single.append(noise[l: l + 1])

            # print('size of noise :', [noise_single[i].size() for i in range(len(noise_single))], latent_in[l].size())
            if 'leftImg8bit' in input_name:  ###cityscapes dataset
                save_name = os.path.basename(os.path.splitext(input_name)[0])[:-12]
                print("name :", save_name)
                # save_name = input_name[:-12]
                print(save_name)
            else:
                save_name = os.path.basename(os.path.splitext(input_name)[0])
                # save_name = input_name
            result_file[save_name] = {
                "latent": latent_in[l],
                "noise": noise_single,
                }

            saveImg=False
            if saveImg:
                os.makedirs(os.path.join('/checkpoint/marlenec/projector', args.nameExp, 'proj_img'), exist_ok=True)
                utils.save_image(
                    img_gen[l],
                    os.path.join('/checkpoint/marlenec/projector', args.nameExp, 'proj_img', f'{os.path.basename(os.path.splitext(input_name)[0])}.png'),
                    nrow=8,
                    normalize=True,
                    range=(-1, 1),
                )

        torch.cuda.empty_cache()
    print(result_file.keys())
    torch.save(result_file, os.path.join(out_filename, f'allProjectedImg{nb_pt}_{get_rank()}.pt'))

def main_worker(gpu, ngpus_per_node, args):
    print('gpu :', gpu)
    torch.cuda.set_device(gpu)
    device = 'cuda'
    args.gpu = gpu
    if args.distributed:
        if args.distributed == 'single_node':
            args.rank = gpu
            args.world_size = ngpus_per_node
            args.dist_url = 'tcp://127.0.0.1:58473'
        elif args.distributed == 'multi_node':
            args.rank = gpu + int(os.getenv('SLURM_NODEID')) * ngpus_per_node
            args.world_size = int(os.getenv('SLURM_NNODES')) * ngpus_per_node
            host_name = slurm_job_nodelist()[0]
            args.dist_url = f'tcp://{host_name}:58472'
        torch.distributed.init_process_group(backend='nccl', init_method=args.dist_url, world_size=args.world_size,
                                             rank=args.rank)
    else:
        args.rank = 0
        args.world_size = 1
    print(f'{args.rank}, {args.world_size}, {args.gpu}')


    ##################################################################### Load pretrained stylegan2 generator
    g_ema = Generator(args.size, 512, 8, size_ratio=args.size_ratio).to(device)
    g_ema.load_state_dict(torch.load(args.ckpt, map_location = lambda storage, loc: storage)["g_ema"], strict=False)
    g_ema.eval()



    ##################################################################### Load perceptual loss
    # percept = lpips.PerceptualLoss(
    #     model="net-lin", net="vgg", use_gpu=device.startswith("cuda")
    # )
    # percept = lpips.PerceptualLoss(
    #     model="net", net="vgg", use_gpu=device.startswith("cuda")
    # )
    percept = lpips.PerceptualLoss(
        model="net", net="img2stylegan", use_gpu=device.startswith("cuda")
    )

    # if distributed (more than 1 gpu) : create DistributedDataParallel
    if args.distributed:
        g_ema = torch.nn.parallel.DistributedDataParallel(
            g_ema,
            # device_ids=[args.local_rank],
            # output_device=args.local_rank,
            device_ids=[args.gpu],
            output_device=args.gpu,
            broadcast_buffers=False, find_unused_parameters=True)
    else:
        print('Use data parallel')
        g_ema = torch.nn.DataParallel(g_ema, device_ids=[0, 1, 2,3,4,5,6,7])

    ##################################################################### Compute latent_mean, latent_std
    n_mean_latent = 10000
    with torch.no_grad():
        noise_sample = torch.randn(n_mean_latent, 512, device=device)
        latent_out = g_ema.module.style(noise_sample)

        latent_mean = latent_out.mean(0)
        latent_std = ((latent_out - latent_mean).pow(2).sum() / n_mean_latent) ** 0.5

    ##################################################################### Hyperparameters
    # project_batch=20 #in case cannot project all img at once, train project_batch images to their projections
    project_batch = args.batch * ngpus_per_node
    out_filename = f'/checkpoint/marlenec/projector/{args.nameExp}'


    ##################################################################### Load dataset
    args.dir = args.file
    dataset = ProjectorDataset(abs_dir=args.dir, size=args.size, size_ratio=args.size_ratio)
    args.lenDataset = len(dataset)
    print('args lenDataset :', args.lenDataset)
    loader = data.DataLoader(
        dataset,
        batch_size=args.batch,
        sampler=data_sampler(dataset, shuffle=False, distributed=args.distributed)
    )

    train(args, g_ema, loader, project_batch, percept, latent_mean, latent_std,out_filename, device)

def launch_multiProc(args):
    signal.signal(signal.SIGUSR1, handle_sigusr1)
    signal.signal(signal.SIGTERM, handle_sigterm)
    ngpus_per_node = torch.cuda.device_count()
    torch.multiprocessing.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))


def handle_sigusr1(signum, frame):
    os.system(f'scontrol requeue {os.environ["SLURM_JOB_ID"]}')
    exit()

def handle_sigterm(signum, frame):
    pass

if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(
        description="Image projector to the generator latent spaces"
    )
    parser.add_argument(
        "--ckpt", type=str, required=True, help="path to the model checkpoint"
    )

    parser.add_argument(
        "--nameExp", type=str, required=True, help="name of exp file"
    )
    parser.add_argument(
        "--size", type=int, default=256, help="output image sizes of the generator"
    )
    parser.add_argument(
        "--batch", type=int, default=8, help="batch sizes for each gpus"
    )
    parser.add_argument(
        "--size_ratio", type=int, default=1, help="image size ratio for the model"
    )

    parser.add_argument(
        "--distributed", type=str, default ='single_node', help="path to the model checkpoint"
    )

    parser.add_argument(
        "--lr_rampup",
        type=float,
        default=0.05,
        help="duration of the learning rate warmup",
    )
    parser.add_argument(
        "--lr_rampdown",
        type=float,
        default=0.25,
        help="duration of the learning rate decay",
    )
    parser.add_argument("--lr", type=float, default=0.1, help="learning rate")
    parser.add_argument(
        "--noise", type=float, default=0.05, help="strength of the noise level"
    )
    parser.add_argument(
        "--noise_ramp",
        type=float,
        default=0.75,
        help="duration of the noise level decay",
    )
    parser.add_argument("--step", type=int, default=1000, help="optimize iterations")
    parser.add_argument(
        "--noise_regularize",
        type=float,
        default=1e5,
        help="weight of the noise regularization",
    )
    parser.add_argument("--mse", type=float, default=0, help="weight of the mse loss")
    parser.add_argument(
        "--w_plus",
        action="store_true",
        help="allow to use distinct latent codes to each layers",
    )
    parser.add_argument(
        "--local_rank", type=int, default=0, help="local rank for distributed training"
    )
    parser.add_argument(
        "file", metavar="FILE", help="path to image files to be projected"
    )

    args = parser.parse_args()

    os.makedirs(os.path.join('/checkpoint/marlenec/projector', args.nameExp), exist_ok=True)


    ##################################################################### If distributed : initiate it
    launch_multiProc(args)







