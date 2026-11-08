#!/bin/bash
RACINE="$(cd "$(dirname "$0")" && pwd)"
MODELES="$RACINE/Existing_Models"

for MODELE in EDSR SPAN Real-ERSGAN; do
    echo "=== $MODELE ==="
    if (cd "$MODELES/$MODELE" && ./test.sh); then
        cat "$MODELES/$MODELE/results.txt"
    else
        echo "Erreur pendant le test de $MODELE :"
        tail -n 5 "$MODELES/$MODELE/results.txt"
    fi
    echo ""
done

echo "=== U-Net ==="
cd "$MODELES/UNet" || exit 1
for S in 2 3 4; do
    echo "x$S : test en cours..."
    if python eval.py --root ../../Images --scale "$S" \
        --ckpt "checkpoints/x$S/unet_x${S}_best.pt" \
        --csv "results_x$S.csv" > "results_x$S.txt" 2>&1; then
        grep -E "^(Bicubique|U-Net) +[0-9]" "results_x$S.txt"
    else
        echo "Erreur pendant le test du U-Net x$S (voir Existing_Models/UNet/results_x$S.txt)"
    fi
done
