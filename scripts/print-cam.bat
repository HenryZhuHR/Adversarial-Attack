
set DEVICE=cuda
set DATA=data/custom
set RES_SAVE=images

python3 cam.py ^
    --arch resnet50 ^
    --checkpoint checkpoints/resnet50.pt ^
    --device %DEVICE% ^
    --data %DATA% ^
    --result_save_dir %RES_SAVE%
python3 cam.py ^
    --arch resnet50_nonlocal_layer1 ^
    --checkpoint checkpoints/resnet50_nonlocal_layer1-best.pt ^
    --device %DEVICE% ^
    --data %DATA% ^
    --result_save_dir %RES_SAVE%
python3 cam.py ^
    --arch resnet50_nonlocal_layer2 ^
    --checkpoint checkpoints/resnet50_nonlocal_layer2-best.pt ^
    --device %DEVICE% ^
    --data %DATA% ^
    --result_save_dir %RES_SAVE%
python3 cam.py ^
    --arch resnet50_nonlocal_layer3 ^
    --checkpoint checkpoints/resnet50_nonlocal_layer3-best.pt ^
    --device %DEVICE% ^
    --data %DATA% ^
    --result_save_dir %RES_SAVE%
python3 cam.py ^
    --arch resnet50_nonlocal_layer4 ^
    --checkpoint checkpoints/resnet50_nonlocal_layer4-best.pt ^
    --device %DEVICE% ^
    --data %DATA% ^
    --result_save_dir %RES_SAVE%
python3 cam.py ^
    --arch resnet50_nonlocal_5block ^
    --checkpoint checkpoints/resnet50_nonlocal_5block-best.pt ^
    --device %DEVICE% ^
    --data %DATA% ^
    --result_save_dir %RES_SAVE%
python3 cam.py ^
    --arch resnet50_nonlocal_10block ^
    --checkpoint checkpoints/resnet50_nonlocal_10block-best.pt ^
    --device %DEVICE% ^
    --data %DATA% ^
    --result_save_dir %RES_SAVE%