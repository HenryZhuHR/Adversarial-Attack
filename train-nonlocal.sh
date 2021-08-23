BATCH_SIZE=128
MAX_EPOCH=100
LR=0.001
NUM_WORKER=8


python3 train.py \
    --arch resnet34 \
    --device 'cuda:0' \
    --batch_size ${BATCH_SIZE} \
    --max_epoch ${MAX_EPOCH} \
    --lr ${LR} \
    --num_worker ${NUM_WORKER} \
    --model_save_dir 'checkpoints' \
    --data 'data' \
    --logdir 'runs'

python3 train.py \
    --arch resnet34_nonlocal_layer1 \
    --device 'cuda:0' \
    --batch_size 64 \
    --max_epoch ${MAX_EPOCH} \
    --lr ${LR} \
    --num_worker ${NUM_WORKER} \
    --model_save_dir 'checkpoints' \
    --data 'data' \
    --logdir 'runs'

python3 train.py \
    --arch resnet34_nonlocal_layer2 \
    --device 'cuda:0' \
    --batch_size ${BATCH_SIZE} \
    --max_epoch ${MAX_EPOCH} \
    --lr ${LR} \
    --num_worker ${NUM_WORKER} \
    --model_save_dir 'checkpoints' \
    --data 'data' \
    --logdir 'runs' 

python3 train.py \
    --arch resnet34_nonlocal_layer3 \
    --device 'cuda:0' \
    --batch_size ${BATCH_SIZE} \
    --max_epoch ${MAX_EPOCH} \
    --lr ${LR} \
    --num_worker ${NUM_WORKER} \
    --model_save_dir 'checkpoints' \
    --data 'data' \
    --logdir 'runs' 

python3 train.py \
    --arch resnet34_nonlocal_layer3 \
    --device 'cuda:0' \
    --batch_size ${BATCH_SIZE} \
    --max_epoch ${MAX_EPOCH} \
    --lr ${LR} \
    --num_worker ${NUM_WORKER} \
    --model_save_dir 'checkpoints' \
    --data 'data' \
    --logdir 'runs' 