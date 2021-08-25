BATCH_SIZE=128
DEVICE='cuda'
MAX_EPOCH=100
LR=0.001
NUM_WORKER=8

MDOEL_SAVE_DIR='checkpoints'
DATASET_DIR='data'
LOG_DIR='runs'


python3 train.py \
    --arch resnet34 \
    --device DEVICE \
    --batch_size ${BATCH_SIZE} \
    --max_epoch ${MAX_EPOCH} \
    --lr ${LR} \
    --num_worker ${NUM_WORKER} \
    --model_save_dir ${MDOEL_SAVE_DIR} \
    --data ${DATASET_DIR} \
    --logdir ${LOG_DIR}

python3 train.py \
    --arch resnet34_nonlocal_layer1 \
    --device DEVICE \
    --batch_size 64 \
    --max_epoch ${MAX_EPOCH} \
    --lr ${LR} \
    --num_worker ${NUM_WORKER} \
    --model_save_dir ${MDOEL_SAVE_DIR} \
    --data ${DATASET_DIR} \
    --logdir ${LOG_DIR}

python3 train.py \
    --arch resnet34_nonlocal_layer2 \
    --device DEVICE \
    --batch_size ${BATCH_SIZE} \
    --max_epoch ${MAX_EPOCH} \
    --lr ${LR} \
    --num_worker ${NUM_WORKER} \
    --model_save_dir ${MDOEL_SAVE_DIR} \
    --data ${DATASET_DIR} \
    --logdir ${LOG_DIR} 

python3 train.py \
    --arch resnet34_nonlocal_layer3 \
    --device DEVICE \
    --batch_size ${BATCH_SIZE} \
    --max_epoch ${MAX_EPOCH} \
    --lr ${LR} \
    --num_worker ${NUM_WORKER} \
    --model_save_dir ${MDOEL_SAVE_DIR} \
    --data ${DATASET_DIR} \
    --logdir ${LOG_DIR} 

python3 train.py \
    --arch resnet34_nonlocal_layer3 \
    --device DEVICE \
    --batch_size ${BATCH_SIZE} \
    --max_epoch ${MAX_EPOCH} \
    --lr ${LR} \
    --num_worker ${NUM_WORKER} \
    --model_save_dir ${MDOEL_SAVE_DIR} \
    --data ${DATASET_DIR} \
    --logdir ${LOG_DIR} 