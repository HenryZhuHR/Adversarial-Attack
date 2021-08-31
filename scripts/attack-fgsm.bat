

@echo off
for %%E in (0, 0.1, 0.5, 1, 2, 4, 8, 16, 32, 64, 128) do (
    @echo on
    python3 test-attack.py ^
        --arch "resnet50" ^
        --checkpoint "checkpoints\resnet50-best.pt" ^
        --attack_method "fgsm" ^
        --epsilon %%E ^
        --device "cuda" ^
        --data "data/custom" ^
        --batch_size 4 ^
        --num_worker 0 ^
        --logs logs
    @echo off
)