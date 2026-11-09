"""Diagnostic explicite du runtime et de tous les vrais checkpoints."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from super_resolution import available_model_names, create_processor, supported_scales

print("Python :", sys.executable)
print("PyTorch :", torch.__version__, torch.__file__)
print("Runtime CUDA :", torch.version.cuda)
print("CUDA disponible :", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA indisponible : version CPU de PyTorch ou pilote inaccessible. Aucun test GPU effectué.")
print("GPU :", torch.cuda.get_device_name(0))
for name in available_model_names():
    for scale in supported_scales(name):
        processor = create_processor(name, scale, device="cuda:0")
        output, elapsed = processor.upscale(np.full((24, 32, 3), 100, dtype=np.uint8))
        print(name, f"×{scale}", "paramètres", processor.verify_device(), "entrée",
              processor.last_input_device, "sortie", processor.last_output_device,
              output.shape, f"{elapsed:.1f} ms (sans échauffement)")
        processor.close()
print("Mémoire allouée après libération :", torch.cuda.memory_allocated())
