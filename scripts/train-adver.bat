@REM for 2GB GPU
set BATCH_SIZE=4
set DEVICE= cuda
set MAX_EPOCH=2
set LR=0.001
set NUM_WORKER=0

set MDOEL_SAVE_DIR= server/checkpoints
set DATASET_DIR= data/custom
set LOG_DIR= server/runs

goto start
python3 train.py `
    --arch resnet50_nonlocal_layer2 `
    --device cuda `
    --batch_size 2 `
    --max_epoch 2 `
    --lr 0.001 `
    --num_worker 0 `
    --model_save_dir server/checkpoints `
    --data data/custom `
    --logdir server/runs `
    --adversarial_training `
    --attack_method fgsm `
    --epsilon 4
:start


python3 train.py ^
    --arch resnet50 ^
    --device %DEVICE% ^
    --batch_size %BATCH_SIZE% ^
    --max_epoch %MAX_EPOCH% ^
    --lr %LR% ^
    --num_worker %NUM_WORKER% ^
    --model_save_dir %MDOEL_SAVE_DIR% ^
    --data %DATASET_DIR% ^
    --logdir %LOG_DIR% ^
    --adversarial_training ^
    --attack_method fgsm ^
    --epsilon 2 ^