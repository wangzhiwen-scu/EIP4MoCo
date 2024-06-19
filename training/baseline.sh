#! usr/bin/env bash
args=(
    --headmotion None
    --unsupervised_dataset_name "MR_ART"
    --epochs 100
    --lr 2e-4
    --batch_size 16
    --load_model 0
    )
device=0

# Start--------------------------------------------Start
CUDA_VISIBLE_DEVICES=${device} python ./training/compared_models/baseline_sup_demo.py "${args[@]}" --supervised_dataset_name "ixi_t1_periodic_slight" --lr 2e-4 # 87

# CUDA_VISIBLE_DEVICES=${device} python ./training/compared_models/baseline_sup.py "${args[@]}" --supervised_dataset_name "ixi_t1_periodic_slight" --lr 2e-4 # 87

# CUDA_VISIBLE_DEVICES=${device} python ./training/compared_models/baseline_sup.py "${args[@]}" --supervised_dataset_name "ixi_t1_linear_moderate" # ⭕

# CUDA_VISIBLE_DEVICES=${device} python ./training/compared_models/baseline_sup.py "${args[@]}" --supervised_dataset_name "ixi_t1_nonlinear_moderate"  # 

# CUDA_VISIBLE_DEVICES=${device} python ./training/compared_models/baseline_sup.py "${args[@]}" --supervised_dataset_name "ixi_t1_sudden_moderate" #!

# CUDA_VISIBLE_DEVICES=${device} python ./training/compared_models/baseline_sup.py "${args[@]}" --supervised_dataset_name "ixi_t1_singleshot_moderate"  # ⭕

# CUDA_VISIBLE_DEVICES=${device} python ./training/compared_models/baseline_sup.py "${args[@]}" --supervised_dataset_name "ixi_t1_periodic_slight_rl"  # ⭕

# CUDA_VISIBLE_DEVICES=${device} python ./training/compared_models/baseline_sup.py "${args[@]}" --supervised_dataset_name "ixi_t1_periodic_moderate"

# CUDA_VISIBLE_DEVICES=${device} python ./training/compared_models/baseline_sup.py "${args[@]}" --supervised_dataset_name "ixi_t1_periodic_heavy"
