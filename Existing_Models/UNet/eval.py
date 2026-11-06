"""
Évaluation complète : PSNR-Y, SSIM-Y et temps d'exécution par image.
Bicubique vs U-Net sur les 100 images de DIV2K valid (x4).

Usage (dans le même dossier que unet.py) :
    python eval.py --root ../../Images --ckpt checkpoints/x4/unet_x4_best.pt
Résultats : affichés + enregistrés dans results.csv (une ligne par image).
"""
import argparse
import glob
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from unet import UNetSR


# ---------------------------------------------------------------- métriques
def rgb2y(x):
    """RGB [0,1] -> luminance Y (BT.601, convention des papiers SR), dans [0,1]."""
    w = torch.tensor([65.481, 128.553, 24.966], device=x.device).view(1, 3, 1, 1) / 255.0
    return (x.clamp(0, 1) * w).sum(1, keepdim=True) + 16.0 / 255.0


def psnr(y1, y2):
    mse = (y1 - y2).pow(2).mean().item()
    return 10 * math.log10(1.0 / mse)


def ssim(y1, y2):
    """SSIM standard (fenêtre gaussienne 11x11, sigma 1.5), comme le ssim de MATLAB."""
    k = torch.arange(11, dtype=torch.float32, device=y1.device) - 5
    g = torch.exp(-k ** 2 / (2 * 1.5 ** 2))
    g = g / g.sum()
    win = (g[:, None] * g[None, :]).view(1, 1, 11, 11)
    C1, C2 = 0.01 ** 2, 0.03 ** 2

    mu1, mu2 = F.conv2d(y1, win), F.conv2d(y2, win)
    s11 = F.conv2d(y1 * y1, win) - mu1 ** 2
    s22 = F.conv2d(y2 * y2, win) - mu2 ** 2
    s12 = F.conv2d(y1 * y2, win) - mu1 * mu2
    m = ((2 * mu1 * mu2 + C1) * (2 * s12 + C2)) / ((mu1 ** 2 + mu2 ** 2 + C1) * (s11 + s22 + C2))
    return m.mean().item()


def metrics(sr, hr, shave):
    a, b = rgb2y(sr), rgb2y(hr)
    a, b = a[..., shave:-shave, shave:-shave], b[..., shave:-shave, shave:-shave]
    return psnr(a, b), ssim(a, b)


# ---------------------------------------------------------------- temps
def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def timed(fn, x, device, reps):
    sync(device)
    t0 = time.perf_counter()
    for _ in range(reps):
        out = fn(x)
    sync(device)
    return out, (time.perf_counter() - t0) / reps * 1000  # ms


def load(path, device):
    a = np.array(Image.open(path).convert("RGB"))
    return torch.from_numpy(a).permute(2, 0, 1).float().unsqueeze(0).div(255.0).to(device)


# ---------------------------------------------------------------- main
@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--scale", type=int, default=4)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--reps", type=int, default=10, help="répétitions pour mesurer le temps")
    ap.add_argument("--fp16", action="store_true", help="inférence en demi-précision")
    ap.add_argument("--csv", default="results.csv")
    ap.add_argument("--out", default=None, help="dossier des images super-résolues")
    args = ap.parse_args()

    data_root = args.root
    if not os.path.isdir(os.path.join(data_root, "HR")):
        test_root = os.path.join(data_root, "test")
        if os.path.isdir(os.path.join(test_root, "HR")):
            data_root = test_root

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    s = args.scale
    output_dir = args.out or os.path.join("results", f"x{s}")
    os.makedirs(output_dir, exist_ok=True)
    model = UNetSR(s, args.base, args.depth).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()
    dtype = torch.float16 if (args.fp16 and device.type == "cuda") else torch.float32
    model = model.to(dtype)

    def run_model(x):
        return model(x.to(dtype)).float()

    def run_bic(x):
        return F.interpolate(x, scale_factor=s, mode="bicubic", align_corners=False)

    hr_paths = sorted(glob.glob(os.path.join(data_root, "HR", "*.png")))
    if not hr_paths:
        ap.error(f"Aucune image trouvée dans {data_root}/HR")
    name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    print(f"Device : {name} | précision : {'FP16' if dtype == torch.float16 else 'FP32'} | "
          f"{len(hr_paths)} images\n")

    # échauffement (le 1er passage GPU est toujours lent)
    lr0 = load(os.path.join(data_root, f"LR_X{s}",
                            os.path.basename(hr_paths[0]).replace(".png", f"x{s}.png")), device)
    for _ in range(10):
        run_model(lr0)
        run_bic(lr0)

    rows = []
    for p in hr_paths:
        i = os.path.basename(p)[:-4]
        lr = load(os.path.join(data_root, f"LR_X{s}", f"{i}x{s}.png"), device)
        hr = load(p, device)[..., :lr.shape[-2] * s, :lr.shape[-1] * s]

        bic, t_bic = timed(run_bic, lr, device, args.reps)
        sr, t_sr = timed(run_model, lr, device, args.reps)
        sr_image = (sr[0].clamp(0, 1).mul(255).byte().permute(1, 2, 0).cpu().numpy())
        Image.fromarray(sr_image).save(os.path.join(output_dir, f"{i}.png"))
        pb, sb = metrics(bic, hr, s)
        pu, su = metrics(sr, hr, s)
        rows.append((i, hr.shape[-1], hr.shape[-2], pb, sb, pu, su, t_bic, t_sr))
        print(f"{i}  {hr.shape[-1]}x{hr.shape[-2]}  | bicubique {pb:.2f} dB / {sb:.4f} "
              f"| U-Net {pu:.2f} dB / {su:.4f} | {t_sr:.2f} ms")

    # temps sur une sortie 1080p (référence jeu vidéo)
    x1080 = torch.rand(1, 3, 1080 // s, 1920 // s, device=device)
    _, t1080_sr = timed(run_model, x1080, device, 50)
    _, t1080_bic = timed(run_bic, x1080, device, 50)

    m = np.array([r[3:] for r in rows])
    pb, sb, pu, su, tb, tu = m.mean(0)
    print("\n" + "=" * 66)
    print(f"{'Moyenne sur ' + str(len(rows)) + ' images':<24}{'PSNR-Y':>10}{'SSIM-Y':>10}"
          f"{'ms/image':>11}{'ms 1080p':>11}")
    print("-" * 66)
    print(f"{'Bicubique':<24}{pb:>10.2f}{sb:>10.4f}{tb:>11.2f}{t1080_bic:>11.2f}")
    print(f"{'U-Net':<24}{pu:>10.2f}{su:>10.4f}{tu:>11.2f}{t1080_sr:>11.2f}")
    print(f"{'Gain':<24}{pu - pb:>+10.2f}{su - sb:>+10.4f}")
    print("=" * 66)
    print(f"ms/image = temps moyen sur les images DIV2K (~2040x1356 en sortie)")
    print(f"U-Net en 1080p : {1000 / t1080_sr:.0f} images/s")

    with open(args.csv, "w") as f:
        f.write("image,largeur,hauteur,psnr_bic,ssim_bic,psnr_unet,ssim_unet,ms_bic,ms_unet\n")
        for r in rows:
            f.write(f"{r[0]},{r[1]},{r[2]},{r[3]:.3f},{r[4]:.4f},{r[5]:.3f},{r[6]:.4f},"
                    f"{r[7]:.3f},{r[8]:.3f}\n")
    print(f"Détail par image : {args.csv}")


if __name__ == "__main__":
    main()