#! usr/bin/env bash
args=(
    # Dataset options
    --headmotion None
    # --supervised_dataset_name "ixi_t1_periodic_slight"
    --supervised_dataset_name "ixi_t1_periodic_slight"
    --unsupervised_dataset_name "MR_ART"
    --epochs 500 
    --lr 5e-4
    )
device=0


CUDA_VISIBLE_DEVICES=${device} python ./training/train_models/csmri_newei_spatial.py "${args[@]}" --batch_size 10 --epochs 90  # loss stop