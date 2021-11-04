import os
from shutil import move

import numpy as np
import torch
from torchvision.datasets.folder import default_loader
from torchvision.transforms.functional import to_tensor
from torchvision.utils import save_image


from entropy_models import estimate_bpp
from networks import AugmentedNormalizedFlowHyperPriorCoder
from utils import Alignment, BitStreamIO
DEVICE = torch.device("cpu")

if __name__ == "__main__":
    m = AugmentedNormalizedFlowHyperPriorCoder(
        128, 320, 192, num_layers=2, use_QE=True, use_affine=False,
        use_context=True, condition='GaussianMixtureModel')

    print(m)

    # ckpt = torch.load("./models/ANFIC_R1.ckpt", map_location=DEVICE)
    # img = to_tensor(default_loader(
    #     os.getenv("HOME")+"/Downloads/Kodak/kodim01.png"))[None]

    # ckpt = torch.load("/work/nctu0756640/log/torch_compression/ANFHyperPriorCoder_0530_1212/model.ckpt", map_location=DEVICE)
    # ckpt = torch.load(
    #     "/work/nctu0756640/log/torch_compression/ANFHyperPriorCoder_0530_1213/model.ckpt", map_location=DEVICE)
    m = m.to(DEVICE)
    # img = to_tensor(default_loader(
    #     os.getenv("DATASET")+"Kodak/kodim01.png"))[None]

    # for f in os.listdir("./models/"):
    #     if not f.endswith("ckpt"):
    #         continue
    #     ckpt = torch.load("./models/"+f, map_location=DEVICE)
    #     new_weights = {}
    #     for k in ckpt:
    #         new_weights[k.replace("DQ", "QE")] = ckpt[k]
    #     print(new_weights)
    #     torch.save({"coder": new_weights}, "./models/"+f)

    # m.load_state_dict(new_weights)

    # mode = "eval"
    # if "compress" in mode:
    #     m.conditional_bottleneck.to("cpu")

    # # print(img.shape)
    # m.eval()
    # align = Alignment(m.divisor)
    # psnrs, bpps = [], []
    # for i in range(1, 25):
    #     # img = to_tensor(default_loader(
    #     #     os.getenv("DATASET")+f"Kodak/kodim{i:02d}.png"))[None].to("cuda:0")
    #     img = torch.rand(1, 3, 256, 256).to(DEVICE)
    #     aligned_img = align.align(img)

    #     with torch.no_grad():

    #         if mode == "eval":
    #             rec, ll, _ = m(aligned_img)
    #             bpp = estimate_bpp(ll, input=img).item()

    #         elif mode == "compress":
    #             split_file = True
    #             file_name = f"./compress{i:02d}.anfic"
    #             with BitStreamIO(file_name, 'w') as fp:
    #                 rec, stream_list, shape_list = m.compress(
    #                     aligned_img, return_hat=True)
    #                 if len(stream_list[0]) == 2:
    #                     tmp, minmax = stream_list.pop(0)
    #                     split_file = True
    #                     shape_list.append((1, 1, 1, minmax))
    #                 fp.write(stream_list, [img.size()]+shape_list)

    #             bytes = os.path.getsize(file_name)
    #             if split_file:
    #                 move(file_name, f"./compress{i:02d}_side.anfic")
    #                 move(tmp, file_name)
    #                 bytes += os.path.getsize(file_name)

    #             bpp = bytes*8/img.size(-1)/img.size(-2)

    #         elif mode == "decompress":
    #             file_name = f"./compress{i:02d}.anfic"
    #             bytes = os.path.getsize(file_name)

    #             assert os.path.isfile(f"./compress{i:02d}_side.anfic")
    #             split_file = os.path.isfile(f"./compress{i:02d}_side.anfic")
    #             if split_file:
    #                 move(file_name, "/tmp/anfic.tmp")
    #                 move(f"./compress{i:02d}_side.anfic", file_name)
    #                 bytes += os.path.getsize(file_name)

    #             with BitStreamIO(file_name, 'r') as fp:
    #                 stream_list, shape_list = fp.read_file()
    #                 if split_file:
    #                     minmax = shape_list.pop(-1)[-1]
    #                     stream_list.insert(0, ("/tmp/anfic.tmp", minmax))
    #                     print(minmax)

    #                 img_tilde = m.decompress(stream_list, shape_list[1:])

    #             rec = align.resume(img_tilde, shape=shape_list[0])
    #             bpp = bytes*8/img.size(-1)/img.size(-2)

    #         rec = align.resume(rec, img.size())
    #         mse = torch.nn.functional.mse_loss(rec.mul(255).clamp(
    #             0, 255).round(), img.mul(255).clamp(0, 255).round())
    #         psnr = 20*np.log10(255) - 10*torch.log10(mse)

    #     psnrs.append(psnr.item())
    #     bpps.append(bpp)
    #     print("kodak {:02d}: {:.4f} {:.4f}".format(i, psnrs[-1], bpps[-1]))
    #     save_image(rec, f"./rec{i:02d}.png")

    #     # break

    # print(np.mean(psnrs), np.mean(bpps))
