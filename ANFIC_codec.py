import argparse
import os
import random
import sys
import time
from glob import glob
from shutil import copyfile, move

import numpy as np
import torch
from absl import app
from absl.flags import argparse_flags
from skimage import io
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image

from datasets import CustomData
from entropy_models import estimate_bpp
from loss import MS_SSIM, PSNR
from metric import MultiScaleSSIM, PSNR_np
from networks import AugmentedNormalizedFlowHyperPriorCoder
from utils import Alignment, BitStreamIO

Dataset_dir = os.getenv("DATASET")
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_coder_from_args(args):
    return AugmentedNormalizedFlowHyperPriorCoder(
        args.var_filters, args.num_features, args.num_hyperpriors,
        num_layers=args.num_layers, use_QE=args.use_QE,
        init_code=args.init_code, use_affine=not args.disable_mul,
        use_mean=args.Mean, use_context=args.use_context, condition=args.condition, quant_mode=args.quant_mode)


def load_coder(args):
    coder = get_coder_from_args(args)
    align = Alignment(coder.divisor)

    # custom method for loading last checkpoint
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    print("========================================================================\n"
          "Loading model checkpoint: ", args.checkpoint,
          "\n========================================================================")

    try:
        coder.load_state_dict(ckpt['coder'])
    except RuntimeError as e:
        # Warning(e)
        print(e)
        coder.load_state_dict(ckpt['coder'], strict=False)

    coder = coder.to(DEVICE)
    if "compress" in args.command:
        coder.conditional_bottleneck.to("cpu")
    coder.eval()

    return coder, align


@torch.no_grad()
def eval(args):
    coder, align = load_coder(args)

    test_dataset = CustomData(args.source_dir, transforms.ToTensor())
    test_dataloader = DataLoader(test_dataset, 1, False, num_workers=16)

    assert args.source_dir != args.target_dir, (
        args.source_dir,  args.target_dir)
    os.makedirs(args.target_dir, exist_ok=True)

    eval_psnr_list, eval_msssim_list = [], []
    eval_rate_list, dec_time_list = [], []

    psnr = PSNR(reduction='mean', data_range=255.)
    ms_ssim = MS_SSIM(reduction='mean', data_range=255.).to(DEVICE)

    for eval_img, img_path in test_dataloader:
        img_name = os.path.basename(img_path[0][:-4])
        save_name = os.path.join(args.target_dir, img_name+"_rec.png")
        t0 = time.perf_counter()

        aligned_img = align.align(eval_img)
        img_tilde, eval_likelihoods, _ = coder(aligned_img)
        eval_img_tilde = align.resume(img_tilde)

        decode_time = time.perf_counter() - t0
        save_image(eval_img_tilde, save_name)

        eval_rate = estimate_bpp(eval_likelihoods, input=eval_img)
        eval_img = eval_img.mul(255.).clamp(0, 255).round()
        eval_img_tilde = eval_img_tilde.mul(255.).clamp(0, 255).round()

        eval_psnr = psnr(eval_img_tilde, eval_img)
        eval_msssim = ms_ssim(eval_img_tilde, eval_img)

        print("{}:: PSNR: {:2.4f}, MS-SSIM: {:.4f}, rate: {:.4f}/{:.3f}(s)".format(
            img_name, eval_psnr, eval_msssim, eval_rate.item(), decode_time
        ))

        eval_psnr_list.append(eval_psnr)
        eval_msssim_list.append(eval_msssim)
        eval_rate_list.append(eval_rate)
        dec_time_list.append(decode_time)

    if len(eval_rate_list) > 1:
        print("==========avg. performance==========")
        print("PSNR: {:.4f}, MS-SSIM: {:.4f}, rate: {:.4f}, dec_time: {:.4f}".format(
            np.mean(eval_psnr_list),
            np.mean(eval_msssim_list),
            np.mean(eval_rate_list),
            np.mean(dec_time_list)
        ))


@torch.no_grad()
def compress(args):
    coder, align = load_coder(args)

    # Create input data pipeline.
    test_dataset = CustomData(args.source_dir, transforms.ToTensor())
    test_dataloader = DataLoader(test_dataset, 1, False, num_workers=16)

    if args.target_dir is None:
        args.target_dir = args.source_dir
    os.makedirs(args.target_dir, exist_ok=True)

    enc_time_list, rate_list = [], []

    for eval_img, img_path in test_dataloader:
        img_name = os.path.basename(img_path[0][:-4])
        file_name = os.path.join(args.target_dir, img_name + ".anifc")
        sidefile_name = os.path.join(args.target_dir, img_name + ".anifcside")
        t0 = time.perf_counter()
        eval_img = eval_img.to(DEVICE)

        with BitStreamIO(sidefile_name, 'w') as fp:
            stream_list, shape_list = coder.compress(align.align(eval_img))
            if len(stream_list[0]) == 2:
                tmp, minmax = stream_list.pop(0)
                shape_list.append((1, 1, 1, minmax))
            fp.write(stream_list, [eval_img.size()]+shape_list)
            move(tmp, file_name)

        encode_time = time.perf_counter() - t0

        if args.eval:
            eval_rate = (os.path.getsize(file_name) + os.path.getsize(
                sidefile_name)) * 8 / (eval_img.size(2) * eval_img.size(3))

            print("{}:: rate: {:.4f}/{:.3f}(s)".format(
                img_name, eval_rate, encode_time))

            enc_time_list.append(encode_time)
            rate_list.append(eval_rate)

    if args.eval and len(rate_list) > 1:
        print("==========avg. performance==========")
        print("rate: {:.4f}, enc_time: {:.4f}".format(
            np.mean(rate_list),
            np.mean(enc_time_list),
        ))


@torch.no_grad()
def decompress(args):
    coder, align = load_coder(args)

    os.makedirs(args.target_dir, exist_ok=True)
    file_name_list = sorted(glob(os.path.join(args.source_dir, "*.anifc")))
    if len(file_name_list) == 0:
        print('compressed file not found in', args.source_dir)
        return

    eval_psnr_list, eval_msssim_list = [], []
    eval_rate_list, dec_time_list = [], []

    for file_name in file_name_list:
        img_name = os.path.basename(file_name)[:-6]
        sidefile_name = os.path.join(args.source_dir, img_name + ".anifcside")
        save_name = os.path.join(args.target_dir, img_name+".png")
        t0 = time.perf_counter()

        with BitStreamIO(sidefile_name, 'r') as fp:
            copyfile(file_name, "/tmp/context.tmp")
            stream_list, shape_list = fp.read_file()
            minmax = shape_list.pop(-1)[-1]
            stream_list.insert(0, ("/tmp/context.tmp", minmax))
            eval_img_tilde = coder.decompress(stream_list, shape_list[1:])
            eval_img_tilde = align.resume(
                eval_img_tilde, shape=shape_list[0])

        decode_time = time.perf_counter() - t0
        save_image(eval_img_tilde, save_name)

        if args.eval:
            eval_name = os.path.join(args.original_dir, img_name+".png")
            eval_img = io.imread(eval_name)
            eval_img_np = io.imread(save_name)

            eval_psnr = PSNR_np(eval_img, eval_img_np, data_range=255.)
            eval_msssim = MultiScaleSSIM(eval_img[None], eval_img_np[None])
            eval_rate = (os.path.getsize(file_name) + os.path.getsize(
                sidefile_name)) * 8 / (eval_img.shape[0] * eval_img.shape[1])

            print("{}:: PSNR: {:2.4f}, MS-SSIM: {:.4f}, rate: {:.4f}/{:.3f}(s)".format(
                img_name, eval_psnr, eval_msssim, eval_rate, decode_time
            ))

            eval_psnr_list.append(eval_psnr)
            eval_msssim_list.append(eval_msssim)
            eval_rate_list.append(eval_rate)
            dec_time_list.append(decode_time)

    if args.eval and len(eval_rate_list) > 1:
        print("==========avg. performance==========")
        print("PSNR: {:.4f}, MS-SSIM: {:.4f}, rate: {:.4f}, dec_time: {:.4f}".format(
            np.mean(eval_psnr_list),
            np.mean(eval_msssim_list),
            np.mean(eval_rate_list),
            np.mean(dec_time_list)
        ))


def parse_args(argv):
    """Parses command line arguments."""
    parser = argparse_flags.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    from entropy_models import __CONDITIONS__, EntropyModel

    # High-level options.
    parser.add_argument(
        "--use_context", "-UC", action="store_true",
        help="Use ContextModel.")
    parser.add_argument(
        "--condition", "-C", type=str, default="Gaussian", choices=__CONDITIONS__,
        help="Condition bottelneck.")
    parser.add_argument(
        "--num_features", "-NF", type=int, default=192,
        help="Number of filters per layer.")
    parser.add_argument(
        "--num_hyperpriors", "-NHP", type=int, default=192,
        help="Number of filters per layer.")
    parser.add_argument(
        "--Mean", "-M", action="store_true",
        help="Enable hyper-decoder to output predicted mean or not.")
    parser.add_argument(
        "--quant_mode", "-QM", type=str, default='noise', choices=EntropyModel.quant_modes,
        help="quantize with noise or round when trianing.")
    parser.add_argument(
        "--var_filters", "-VNFL", type=int, nargs='+',
        help="variable filters.")
    parser.add_argument(
        "--num_layers", "-L", type=int, default=2,
        help="Layers of ANF.")
    parser.add_argument(
        "--use_QE", "-QE", action="store_true",
        help="Use Quality Enhencement.")
    parser.add_argument(
        "--init_code", type=str, default='gaussian',
        help="init code distribution.")
    parser.add_argument(
        "--disable_mul", "-DM", action="store_true",
        help="Disable affine(multiply).")
    parser.add_argument(
        "--checkpoint_dir", "-cpdir", default="./",
        help="Directory where to save/load model checkpoints.")

    subparsers = parser.add_subparsers(
        title="commands", dest="command",
        help="What to do: \n"
             "'eval' reads an image file (lossless PNG format) and do the model forwarding to get the simulated compression result.\n"
             "'compress' reads an image file (lossless PNG format) and writes a compressed binary file.\n"
             "'decompress' reads a binary file and reconstructs the image (in PNG format).\n"
             "input and output filenames need to be provided for the latter two options.\n\n"
             "Invoke '<command> -h' for more information.")

    # 'evaluation' sub-command.
    eval_cmd = subparsers.add_parser(
        "eval",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Evaluate images with a trained model.")
    eval_cmd.add_argument(
        "--checkpoint", "-ckpt", default="model.ckpt",
        help="Model checkpoint name.")
    eval_cmd.add_argument(
        "--source_dir", "-SD",
        help="The directory of the images that are expected to compress.")
    eval_cmd.add_argument(
        "--target_dir", "-TD", default=None,
        help="The directory where the compressed files are expected to store at.")

    # 'compress' sub-command.
    compress_cmd = subparsers.add_parser(
        "compress",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Compress images with a trained model.")
    compress_cmd.add_argument(
        "--checkpoint", "-ckpt", default="model.ckpt",
        help="Model checkpoint name.")
    compress_cmd.add_argument(
        "--source_dir", "-SD",
        help="The directory of the images that are expected to compress.")
    compress_cmd.add_argument(
        "--target_dir", "-TD", default=None,
        help="The directory where the compressed files are expected to store at.")
    compress_cmd.add_argument(
        "--eval", action="store_true",
        help="Evaluate compressed images with original ones.")

    # 'decompress' sub-command.
    decompress_cmd = subparsers.add_parser(
        "decompress",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Decompress bitstreams with a trained model.")
    decompress_cmd.add_argument(
        "--checkpoint", "-ckpt", default="model.ckpt",
        help="Model checkpoint name.")
    decompress_cmd.add_argument(
        "--source_dir", "-SD",
        help="The directory of the compressed files that are expected to decompress.")
    decompress_cmd.add_argument(
        "--target_dir", "-TD",
        help="The directory where the images are expected to store at.")
    decompress_cmd.add_argument(
        "--eval", action="store_true",
        help="Evaluate decompressed images with original ones.")
    decompress_cmd.add_argument(
        "--original_dir", "-OD", nargs="?",
        help="The directory where the original images are expected to store at.")

    # Parse arguments.
    args = parser.parse_args(argv[1:])
    if args.command is None:
        parser.print_usage()
        sys.exit(2)
    return args


def main(args):
    np.set_printoptions(threshold=sys.maxsize)

    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    # Invoke subcommand.
    torch.backends.cudnn.deterministic = True
    if args.command == "eval":
        eval(args)
    if args.command == "compress":
        compress(args)
    elif args.command == "decompress":
        decompress(args)


if __name__ == "__main__":
    app.run(main, flags_parser=parse_args)
