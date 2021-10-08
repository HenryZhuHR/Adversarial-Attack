python3 train.py `
    --arch mnist_cnn `
    --device cuda `
    --batch_size 1024 `
    --max_epoch 10 `
    --lr 0.001 `
    --num_worker 0 `
    --model_save_dir server/checkpoints `
    --model_save_name mnist_cnn `
    --data ~/datasets `
    --logdir server/runs `
    --adversarial_training `
    --attack_method fgsm `
    --epsilon 4


python3 attack.py `
    --arch "resnet50" `
    --checkpoint "checkpoints/resnet50-best.pt" `
    --device "cuda" `
    --data "~/datasets/custom" `
    --batch_size 4 `
    --num_worker 0 `
    --logdir "server/runs" `
    --attack_method "cw" `
    --attack_args c=0.8 kappa=0 max_steps=1000


python3 attack.py `
    --arch "resnet50" `
    --checkpoint "checkpoints/resnet50-best.pt" `
    --device "cuda" `
    --data "~/datasets/custom" `
    --batch_size 1 `
    --num_worker 0 `
    --logdir "server/runs" `
    --attack_method "deepfool" `
    --attack_args num_classes=10 overshoot=1