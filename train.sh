python3 train.py \
    --arch resnet34_nonlocal_layer1 \
    --device cuda:0 \
    --batch_size 1 \
    --max_epoch 10 \
    --lr 0.001 \
    --num_worker 0 \
    --model_save_dir checkpoints \
    --data data \
    --logdir runs \
    --model_summary
