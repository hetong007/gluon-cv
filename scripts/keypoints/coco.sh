python train.py \
    --model simple_pose_resnet50_v1d --mode hybrid --num-joints 17 \
    --lr 0.008 --lr-mode step --lr-decay-epoch 90,120 \
    --num-epochs 140 --batch-size 128 --num-gpus 8 -j 60 \
    --dtype float16 --no-wd --warmup-epochs 5 \
    --save-dir params_simple_pose_resnet50_v1d \
    --logging-file simple_pose_resnet50_v1d.log
