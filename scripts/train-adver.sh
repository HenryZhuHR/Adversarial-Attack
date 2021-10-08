DEVICE=cuda
BATCH_SIZE=64
MAX_EPOCH=100
LR=0.001
NUM_WORKER=8
MDOEL_SAVE_DIR=server/checkpoints
DATASET_DIR=data/custom
LOG_DIR=server/runs
ATTACK_METHOD=fgsm
EPSILON=4
for ARCH in resnet50 \
			resnet50_nonlocal_layer1 
			resnet50_nonlocal_layer2 
			resnet50_nonlocal_layer3 
			resnet50_nonlocal_layer4
do
echo ${ARCH}
	python3 train.py \
		--arch ${ARCH} \
		--device ${DEVICE} \
		--batch_size ${BATCH_SIZE} \
		--max_epoch ${MAX_EPOCH} \
		--lr ${LR} \
		--num_worker ${NUM_WORKER} \
		--model_save_dir ${MDOEL_SAVE_DIR} \
		--data ${DATASET_DIR} \
		--logdir ${LOG_DIR} \
		--model_summary \
		--adversarial_training \
		--attack_method ${ATTACK_METHOD} \

		--epsilon ${EPSILON}
done
