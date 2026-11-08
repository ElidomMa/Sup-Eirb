"""
U-Net pour la super-résolution sur DIV2K (PyTorch).

Arborescence attendue dans --root :
    HR/0801.png ...
    LR_X4/0801x4.png ...

Exemples :
    python unet.py --root data/DIV2K --scale 4
    python unet.py --root data/DIV2K --scale 4 --bench   # temps par image seulement
"""
import argparse
import glob
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset


# ----------------------------------------------------------------------------
# Données
# ----------------------------------------------------------------------------
class DIV2K(Dataset):
    def __init__(self, root, split, scale, patch=48, repeat=1, n_max=None):
        hr_dir = os.path.join(root, split, "HR")
        lr_dir = os.path.join(root, split, f"LR_X{scale}")
        self.hr = sorted(glob.glob(os.path.join(hr_dir, "*.png")))[:n_max]
        self.lr = [os.path.join(lr_dir, os.path.basename(p).replace(".png", f"x{scale}.png"))
                   for p in self.hr]
        assert self.hr, f"Aucune image trouvée dans {hr_dir}"
        self.scale, self.patch, self.repeat = scale, patch, repeat
        self.train = split == "train"

    def __len__(self):
        return len(self.hr) * self.repeat

    def __getitem__(self, i):
        i %= len(self.hr)
        s = self.scale
        hr = np.array(Image.open(self.hr[i]).convert("RGB"))
        lr = np.array(Image.open(self.lr[i]).convert("RGB"))
        h, w = lr.shape[:2]

        if self.train:
            p = self.patch
            y, x = random.randint(0, h - p), random.randint(0, w - p)
            lr = lr[y:y + p, x:x + p]
            hr = hr[y * s:(y + p) * s, x * s:(x + p) * s]
            # augmentations : flips + rotation 90°
            if random.random() < 0.5:
                lr, hr = lr[:, ::-1], hr[:, ::-1]
            if random.random() < 0.5:
                lr, hr = lr[::-1], hr[::-1]
            if random.random() < 0.5:
                lr, hr = lr.transpose(1, 0, 2), hr.transpose(1, 0, 2)
        else:
            hr = hr[:h * s, :w * s]  # aligne HR sur la taille LR × scale

        def to_t(a):
            return torch.from_numpy(np.ascontiguousarray(a)).permute(2, 0, 1).float() / 255.0

        return to_t(lr), to_t(hr)


# ----------------------------------------------------------------------------
# Modèle
# ----------------------------------------------------------------------------
class ConvBlock(nn.Module):
    # Pas de BatchNorm : en SR elle dégrade les résultats (cf. EDSR)
    def __init__(self, cin, cout):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1), nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.body(x)


class UNetSR(nn.Module):
    """
    U-Net qui travaille en basse résolution (moins cher), puis upsampling par
    PixelShuffle. Le réseau prédit un résidu ajouté à l'upsampling bicubique.
    """

    def __init__(self, scale=4, base=32, depth=3):
        super().__init__()
        self.scale, self.depth = scale, depth
        chs = [base * 2 ** i for i in range(depth + 1)]

        self.head = nn.Conv2d(3, chs[0], 3, padding=1)
        self.enc = nn.ModuleList([ConvBlock(chs[i], chs[i]) for i in range(depth)])
        self.down = nn.ModuleList([nn.Conv2d(chs[i], chs[i + 1], 4, stride=2, padding=1)
                                   for i in range(depth)])
        self.mid = ConvBlock(chs[depth], chs[depth])
        self.up = nn.ModuleList([nn.ConvTranspose2d(chs[i + 1], chs[i], 2, stride=2)
                                 for i in reversed(range(depth))])
        self.dec = nn.ModuleList([ConvBlock(2 * chs[i], chs[i]) for i in reversed(range(depth))])
        self.tail = nn.Sequential(
            nn.Conv2d(chs[0], 3 * scale ** 2, 3, padding=1),
            nn.PixelShuffle(scale),
        )

    def forward(self, x):
        H, W = x.shape[-2:]
        m = 2 ** self.depth
        ph, pw = (m - H % m) % m, (m - W % m) % m
        xp = F.pad(x, (0, pw, 0, ph), mode="reflect") if (ph or pw) else x

        f = self.head(xp)
        skips = []
        for enc, down in zip(self.enc, self.down):
            f = enc(f)
            skips.append(f)
            f = down(f)
        f = self.mid(f)
        for up, dec in zip(self.up, self.dec):
            f = up(f)
            f = dec(torch.cat([f, skips.pop()], dim=1))

        res = self.tail(f)[..., :H * self.scale, :W * self.scale]
        base = F.interpolate(x, scale_factor=self.scale, mode="bicubic", align_corners=False)
        return base + res


# ----------------------------------------------------------------------------
# Métriques / benchmark
# ----------------------------------------------------------------------------
def psnr_y(sr, hr, shave):
    """PSNR sur le canal Y (convention standard des papiers SR), bords rognés de `shave` px."""
    w = torch.tensor([65.481, 128.553, 24.966], device=sr.device).view(1, 3, 1, 1) / 255.0
    diff = ((sr.clamp(0, 1) - hr) * w).sum(1)
    if shave > 0:
        diff = diff[..., shave:-shave, shave:-shave]
    mse = diff.pow(2).mean().item()
    return 10 * math.log10(1.0 / mse)


@torch.no_grad()
def evaluate(model, loader, scale, device):
    model.eval()
    s_model, s_bic = [], []
    for lr, hr in loader:
        lr, hr = lr.to(device), hr.to(device)
        bic = F.interpolate(lr, scale_factor=scale, mode="bicubic", align_corners=False)
        s_model.append(psnr_y(model(lr), hr, scale))
        s_bic.append(psnr_y(bic, hr, scale))
    model.train()
    n = len(s_model)
    print(f"     (bicubique : {sum(s_bic)/n:.2f} dB)")
    return sum(s_model) / n


@torch.no_grad()
def benchmark(model, scale, device, out_h=1080, out_w=1920, n=50):
    """Temps moyen pour produire une image 1080p (entrée = 1080p / scale)."""
    model.eval()
    x = torch.rand(1, 3, out_h // scale, out_w // scale, device=device)
    for _ in range(10):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / n * 1000
    model.train()
    return ms


# ----------------------------------------------------------------------------
# Entraînement
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--scale", type=int, default=4)
    ap.add_argument("--patch", type=int, default=48, help="taille du patch LR")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--iters", type=int, default=100_000)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--val_every", type=int, default=2000)
    ap.add_argument("--n_val", type=int, default=10, help="nb d'images de validation")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="checkpoints")
    ap.add_argument("--bench", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetSR(args.scale, args.base, args.depth).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Paramètres : {n_params / 1e6:.2f} M | device : {device}")

    if args.bench:
        print(f"Temps pour une sortie 1080p : {benchmark(model, args.scale, device):.1f} ms")
        return

    train_set = DIV2K(args.root, "train", args.scale, args.patch, repeat=20)
    val_set = DIV2K(args.root, "test", args.scale, n_max=args.n_val)
    train_loader = DataLoader(train_set, args.batch, shuffle=True, num_workers=args.workers,
                              pin_memory=True, drop_last=True, persistent_workers=args.workers > 0)
    val_loader = DataLoader(val_set, 1, num_workers=2)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.iters, eta_min=1e-6)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    os.makedirs(args.out, exist_ok=True)

    it, best, running = 0, 0.0, 0.0
    while it < args.iters:
        for lr_img, hr_img in train_loader:
            lr_img, hr_img = lr_img.to(device, non_blocking=True), hr_img.to(device, non_blocking=True)
            with torch.autocast(device.type, enabled=use_amp):
                loss = F.l1_loss(model(lr_img), hr_img)  # L1 > L2 en pratique pour la SR
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            running += loss.item()
            it += 1

            if it % 100 == 0:
                print(f"[{it}/{args.iters}] L1 {running / 100:.4f}  lr {sched.get_last_lr()[0]:.2e}")
                running = 0.0

            if it % args.val_every == 0 or it == args.iters:
                psnr = evaluate(model, val_loader, args.scale, device)
                print(f"  -> PSNR-Y validation : {psnr:.2f} dB")
                if psnr > best:
                    best = psnr
                    torch.save(model.state_dict(), os.path.join(args.out, f"unet_x{args.scale}_best.pt"))

            if it >= args.iters:
                break

    print(f"Meilleur PSNR-Y : {best:.2f} dB")
    print(f"Temps pour une sortie 1080p : {benchmark(model, args.scale, device):.1f} ms")


if __name__ == "__main__":
    main()




