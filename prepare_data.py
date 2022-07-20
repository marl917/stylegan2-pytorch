import argparse
from io import BytesIO
import multiprocessing
from functools import partial
import numpy as np
import sys
from PIL import Image
import lmdb
from tqdm import tqdm
from torchvision import datasets
import torch
from torchvision.transforms import functional as trans_fn
import json


def resize_and_convert(img, size, resample, quality=100, npy=False):
    img = trans_fn.resize(img, size, resample)
    img = trans_fn.center_crop(img, size)
    #print('value of npy :', npy)
    if npy:
        # print('in resize and convert :', img.size())
        img = trans_fn.to_pil_image(img[0])
    buffer = BytesIO()
    img.save(buffer, format="jpeg", quality=quality)
    # img.save('test.png')
    # sys.exit()
    val = buffer.getvalue()
    return val


def resize_multiple(
    img, sizes=(128, 256, 512, 1024), resample=Image.LANCZOS, quality=100, npy=False
):
    imgs = []

    for size in sizes:
        imgs.append(resize_and_convert(img, size, resample, quality, npy=npy))

    return imgs


def resize_worker(img_file, sizes, resample, npy=False):
    i, file = img_file
    if not npy:
        img = Image.open(file)
        img = img.convert("RGB")
    else:
        img = torch.from_numpy(np.load(file))
        # print(img)
    out = resize_multiple(img, sizes=sizes, resample=resample, npy=npy)

    return i, out


def prepare(
    env, dataset, n_worker, sizes=(128, 256, 512, 1024), resample=Image.LANCZOS, npy=False
):

    resize_fn = partial(resize_worker, sizes=sizes, resample=resample, npy=npy)

    files = sorted(dataset.samples, key=lambda x: x[0])
    files = [(i, file) for i, (file, label) in enumerate(files)]
    total = 0

    with multiprocessing.Pool(n_worker) as pool:
        for i, imgs in tqdm(pool.imap_unordered(resize_fn, files)):
            for size, img in zip(sizes, imgs):
                key = f"{size}-{str(i).zfill(5)}".encode("utf-8")

                with env.begin(write=True) as txn:
                    txn.put(key, img)

            total += 1

        with env.begin(write=True) as txn:
            txn.put("length".encode("utf-8"), str(total).encode("utf-8"))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess images for model training")
    parser.add_argument("--out", type=str, help="filename of the result lmdb dataset")
    parser.add_argument(
        "--size",
        type=str,
        default="128,256,512,1024",
        help="resolutions of images for the dataset",
    )
    parser.add_argument(
        "--n_worker",
        type=int,
        default=8,
        help="number of workers for preparing dataset",
    )
    parser.add_argument(
        "--size_ratio",
        type=int,
        default=1,
        help="image size ratio",
    )
    parser.add_argument(
        "--resample",
        type=str,
        default="lanczos",
        help="resampling methods for resizing images",
    )
    parser.add_argument("path", type=str, help="path to the image dataset")

    args = parser.parse_args()

    resample_map = {"lanczos": Image.LANCZOS, "bilinear": Image.BILINEAR}
    resample = resample_map[args.resample]

    sizes = [int(s.strip()) for s in args.size.split(",")]

    if args.size_ratio!=1:
        tmp=[[i,i*args.size_ratio] for i in sizes]
        sizes=tmp
        # print(sizes)

    print(f"Make dataset of image sizes:", ", ".join(str(s) for s in sizes))


    npy=False
    try:
        print('before imagefolder')
        imgset = datasets.ImageFolder(args.path)
        print('after imagefoler')
    except:
        def npy_loader(path):
            sample = np.load(path)
            return sample
        imgset = datasets.DatasetFolder(
            root=args.path,
            loader=npy_loader,
            extensions='.npy'
        )
        npy=True
    print(npy)


    with lmdb.open(args.out, map_size=1024 ** 4, readahead=False) as env:
        prepare(env, imgset, args.n_worker, sizes=sizes, resample=resample, npy=npy)
