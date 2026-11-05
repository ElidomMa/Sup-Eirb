#!/bin/bash
# Lance les tests SPAN en x2, x3 et x4 et écrit la config de la machine
# et le temps moyen d'inférence par image dans results.txt

TEST_OUTPUT=$(mktemp)
trap 'rm -f "$TEST_OUTPUT"' EXIT
RESULTS_FILE="results.txt"

{
    echo "Machine:"
    echo "Date    : $(date)"
    echo "CPU     : $(lscpu | grep 'Model name' | sed 's/Model name: *//')"
    echo "Coeurs  : $(nproc)"
    echo "RAM     : $(free -h | awk '/Mem/ {print $2}')"
    if command -v nvidia-smi > /dev/null; then
        echo "GPU     : $(nvidia-smi --query-gpu=name --format=csv,noheader)"
    else
        echo "Pas de GPU"
    fi
    echo "PyTorch : $(python -c 'import torch; print(torch.__version__)')"
    echo ""
    echo "Résultats:"

        for S in 2 3 4; do
        if ! python basicsr/test.py -opt options/test/SPAN/test_SPAN_x${S}_DIV2K.yml > "$TEST_OUTPUT" 2>&1; then
            cat "$TEST_OUTPUT"
            exit 1
        fi

        TEMPS=$(grep -oP 'Average inference time: \K[0-9.]+' "$TEST_OUTPUT")
        SSIM=$(grep -oP '# ssim: \K[0-9.]+' "$TEST_OUTPUT")
        PSNR=$(grep -oP '# psnr: \K[0-9.]+' "$TEST_OUTPUT")

        echo "x${S}: ${TEMPS} ms/image, SSIM: ${SSIM}, PSNR: ${PSNR} dB"

    done
} > "$RESULTS_FILE"