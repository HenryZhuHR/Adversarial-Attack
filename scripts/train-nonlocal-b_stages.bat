@REM for 2GB GPU
set BATCH_SIZE=8
set DEVICE= cuda
set MAX_EPOCH=1
set LR=0.001
set NUM_WORKER=0

set MDOEL_SAVE_DIR= temp/checkpoints
set DATASET_DIR= data/custom
set LOG_DIR= temp/runs

goto start
python3 train.py `
    --arch resnet50_nonlocal_10block `
    --device cuda `
    --batch_size 1 `
    --max_epoch 1 `
    --lr 0.001 `
    --num_worker 0 `
    --model_save_dir temp/checkpoints `
    --data data/custom `
    --logdir "temp/runs"
:start


python3 train.py ^
    --arch resnet34 ^
    --device %DEVICE% ^
    --batch_size %BATCH_SIZE% ^
    --max_epoch %MAX_EPOCH% ^
    --lr %LR% ^
    --num_worker %NUM_WORKER% ^
    --model_save_dir %MDOEL_SAVE_DIR% ^
    --data %DATASET_DIR% ^
    --logdir %LOG_DIR%

python3 train.py ^
    --arch resnet34_nonlocal_layer1 ^
    --device %DEVICE% ^
    --batch_size 1 ^
    --max_epoch %MAX_EPOCH% ^
    --lr %LR% ^
    --num_worker %NUM_WORKER% ^
    --model_save_dir %MDOEL_SAVE_DIR% ^
    --data %DATASET_DIR% ^
    --logdir %LOG_DIR%

python3 train.py ^
    --arch resnet34_nonlocal_layer2 ^
    --device %DEVICE% ^
    --batch_size %BATCH_SIZE% ^
    --max_epoch %MAX_EPOCH% ^
    --lr %LR% ^
    --num_worker %NUM_WORKER% ^
    --model_save_dir %MDOEL_SAVE_DIR% ^
    --data %DATASET_DIR% ^
    --logdir %LOG_DIR% 

python3 train.py ^
    --arch resnet34_nonlocal_layer3 ^
    --device %DEVICE% ^
    --batch_size %BATCH_SIZE% ^
    --max_epoch %MAX_EPOCH% ^
    --lr %LR% ^
    --num_worker %NUM_WORKER% ^
    --model_save_dir %MDOEL_SAVE_DIR% ^
    --data %DATASET_DIR% ^
    --logdir %LOG_DIR% 

python3 train.py ^
    --arch resnet34_nonlocal_layer3 ^
    --device %DEVICE% ^
    --batch_size %BATCH_SIZE% ^
    --max_epoch %MAX_EPOCH% ^
    --lr %LR% ^
    --num_worker %NUM_WORKER% ^
    --model_save_dir %MDOEL_SAVE_DIR% ^
    --data %DATASET_DIR% ^
    --logdir %LOG_DIR% 