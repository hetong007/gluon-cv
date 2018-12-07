python validate.py \
    --model simple_pose_resnet50_v1b --num-joints 17 \
    --batch-size 128 --num-gpus 8 -j 60 --dtype float16 \
    --params-file params_simple_pose_resnet50_v1d/imagenet-simple_pose_resnet50_v1b-139.params
