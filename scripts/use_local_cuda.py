"""Réutilise une installation CUDA existante sans copier ses bibliothèques."""

import argparse
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("site_packages", type=Path, help="site-packages d'un Python 3.12 avec torch CUDA")
    args = parser.parse_args()
    source = args.site_packages.resolve()
    version = (source / "torch/version.py").read_text()
    if "+cu" not in version:
        parser.error("L'installation source ne contient pas une version CUDA de PyTorch.")
    root = Path(__file__).resolve().parents[1] / ".venv"
    site = root / f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages"
    if not site.is_dir():
        parser.error("Exécutez ce script avec le Python du projet.")
    target = root / "local-cuda"
    target.mkdir(exist_ok=True)
    items = [source / name for name in ("torch", "torchgen", "functorch", "nvidia", "cuda", "triton")]
    for pattern in ("torch-*.dist-info", "triton-*.dist-info", "nvidia*.dist-info", "cuda*.dist-info"):
        items.extend(source.glob(pattern))
    for item in items:
        destination = target / item.name
        if item.exists() and not destination.exists():
            destination.symlink_to(item, target_is_directory=item.is_dir())
        elif destination.is_symlink() and destination.resolve() != item.resolve():
            parser.error(f"Lien existant vers une autre installation : {destination}")
    pth = site / "sup_eirb_local_cuda.pth"
    pth.write_text(f"import sys; sys.path.insert(0, {str(target)!r})\n")
    (target / "source.json").write_text(json.dumps({"source": str(source), "undo": str(pth)}, indent=2))
    print(f"Installation réutilisée : {source}")
    print(f"Pour revenir à l'installation précédente, supprimer uniquement : {pth}")
    print("Relancez Python puis vérifiez scripts/check_cuda.py.")


if __name__ == "__main__":
    main()
