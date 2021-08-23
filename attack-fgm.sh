
BATCH=128

for EPSILON in 0 0.1 0.5 1 2 4 8 16 32 64 128
do 
    python3 attack-fgm.py \
        --epsilon       ${EPSILON} \
        --batch_size    ${BATCH} \
        --device        "cuda:0" \
        --num_worker    8 \
        --pretrained    "checkpoints/noAttack.pt"
done

