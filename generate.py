import argparse

import torch
from torchvision import utils
from model import Generator
from tqdm import tqdm
import fid
import numpy as np
from dataset import MultiResolutionDataset
from torchvision import transforms, utils
import sys
# from fvcore.nn import FlopCountAnalysis
import sys 

# def generate(args, g_ema, device, mean_latent):

#     with torch.no_grad():
#         g_ema.eval()
#         for i in tqdm(range(args.pics)):
#             sample_z = torch.randn(args.sample, args.latent, device=device)
            
            
#             flops = FlopCountAnalysis(g_ema, [torch.unsqueeze(sample_z[0], dim=0)])
#             print('ok')
#             print(flops.by_operator())
#             print(flops.total())
#             sys.exit()

#             sample, _ = g_ema(
#                 [sample_z], truncation=args.truncation, truncation_latent=mean_latent
#             )

#             utils.save_image(
#                 sample,
#                 f"sample_adeindoor/{str(i).zfill(6)}.png",
#                 nrow=1,
#                 normalize=True,
#                 range=(-1, 1),
#             )


if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(description="Generate samples from the generator")

    parser.add_argument(
        "--size", type=int, default=1024, help="output image size of the generator"
    )

    parser.add_argument(
        "--size_ratio", type=int, default=1, help="output image size of the generator"
    )

    parser.add_argument(
        "--fid", type=bool, default=False, help="output image size of the generator"
    )


    parser.add_argument(
        "--sample",
        type=int,
        default=1,
        help="number of samples to be generated for each image",
    )
    parser.add_argument(
        "--pics", type=int, default=20, help="number of images to be generated"
    )
    parser.add_argument("--truncation", type=float, default=1, help="truncation ratio")
    parser.add_argument(
        "--truncation_mean",
        type=int,
        default=4096,
        help="number of vectors to calculate mean for the truncation",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="stylegan2-ffhq-config-f.pt",
        help="path to the model checkpoint",
    )
    parser.add_argument(
        "--channel_multiplier",
        type=int,
        default=2,
        help="channel multiplier of the generator. config-f = 2, else = 1",
    )

    parser.add_argument(
        "--mixing", type=float, default=0.9, help="probability of latent code mixing"
    )

    parser.add_argument("--val_path", type=str, help="path to the lmdb eval dataset")

    parser.add_argument("--output_dir", type=str, help="for fid")

    args = parser.parse_args()

    args.latent = 512
    args.n_mlp = 8

    g_ema = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier, size_ratio=args.size_ratio
    ).to(device)
    checkpoint = torch.load(args.ckpt)

    g_ema.load_state_dict(checkpoint["g_ema"])

    if args.truncation < 1:
        with torch.no_grad():
            mean_latent = g_ema.mean_latent(args.truncation_mean)
    else:
        mean_latent = None


    if args.size_ratio != 1.:  # to use for cityscapes
        size_img = [args.size, args.size * args.size_ratio]
    else:
        size_img = args.size


    fid_value = []
    if args.fid:
        transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
            ]
        )
        val_dataset = MultiResolutionDataset(args.val_path, transform, size_img)
        fid_computer = fid.fid_pytorch(args, val_dataset, device=device)

        for i in range(5):
            is_best, fid_val = fid_computer.update(netEMA=g_ema, cur_iter=10000000)
            print(fid_val)
            fid_value.append(fid_val)
        print(np.mean(fid_value), np.var(fid_value))



    generate(args, g_ema, device, mean_latent)
