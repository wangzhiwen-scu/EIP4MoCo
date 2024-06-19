#! usr/bin/env bash
args=(
    # Dataset options
    --headmotion None
    # --supervised_dataset_name "IXI"
    # --supervised_dataset_name "ixi_t1_periodic_heavy"
    --supervised_dataset_name "ixi_t1_periodic_slight"
    --unsupervised_dataset_name "MR_ART"
    --epochs 500 
    --lr 5e-4 
    )
device=0

# Start--------------------------------------------Start

# CUDA_VISIBLE_DEVICES=${device} python ./training/compared_models/marc.py "${args[@]}" --batch_size 8 --epochs 350 # good perf. (10points)
# CUDA_VISIBLE_DEVICES=${device} python ./training/compared_models/cyclegan.py "${args[@]}" --batch_size 8 --epochs 200 # good perf. (10points)

# CUDA_VISIBLE_DEVICES=${device} python ./training/compared_models/bsa.py "${args[@]}" --batch_size 8 --epochs 100 # good perf. (10points)


args=(
    # Dataset options
    --headmotion None
    # --supervised_dataset_name "IXI"
    # --supervised_dataset_name "ixi_t1_periodic_heavy"
    # --supervised_dataset_name "ixi_t2_periodic_moderate"
    --supervised_dataset_name "ixi_t2_periodic_slight"
    --unsupervised_dataset_name "MR_ART"
    --epochs 500 
    --lr 5e-4 
    )
device=0
# CUDA_VISIBLE_DEVICES=${device} python ./training/compared_models/marc.py "${args[@]}" --batch_size 8 --epochs 350 # good perf. (10points)
# CUDA_VISIBLE_DEVICES=${device} python ./training/compared_models/cyclegan.py "${args[@]}" --batch_size 8 --epochs 200 # good perf. (10points)

# CUDA_VISIBLE_DEVICES=${device} python ./training/compared_models/bsa.py "${args[@]}" --batch_size 8 --epochs 100 # good perf. (10points)

args=(
    # Dataset options
    --headmotion None
    # --supervised_dataset_name "IXI"
    # --supervised_dataset_name "ixi_t1_periodic_heavy"
    # --supervised_dataset_name "ixi_pd_periodic_moderate"
    --supervised_dataset_name "ixi_pd_periodic_slight"
    --unsupervised_dataset_name "MR_ART"
    --epochs 500 
    --lr 5e-4 
    )
device=0
# CUDA_VISIBLE_DEVICES=${device} python ./training/compared_models/marc.py "${args[@]}" --batch_size 8 --epochs 350 # good perf. (10points)
# CUDA_VISIBLE_DEVICES=${device} python ./training/compared_models/cyclegan.py "${args[@]}" --batch_size 8 --epochs 200 # good perf. (10points)

# CUDA_VISIBLE_DEVICES=${device} python ./training/compared_models/bsa.py "${args[@]}" --batch_size 8 --epochs 100 # good perf. (10points)