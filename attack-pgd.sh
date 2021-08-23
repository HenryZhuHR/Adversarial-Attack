BATCH=64
for NUM_STEPS in 5 10 20 40
do 
    for EPSILON in 1 2 4 8 16 32 64
    do
        for ALPHA in 1 2 4 8 16 32 64
        do
        python3 attack-pgd.py \
            --epsilon       ${EPSILON} \
            --alpha         ${ALPHA} \
            --num_steps     ${NUM_STEPS} \
            --batch_size    ${BATCH} \
            --device        "cuda:0" \
            --num_worker    8 \
            --pretrained    "checkpoints/noAttack.pt"
        done
    done
done


