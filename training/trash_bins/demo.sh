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

# Start--------------------------------------------Start
# ablation
# CUDA_VISIBLE_DEVICES=${device} python ./training/train_models/csmri_and_ei_tvloss.py "${args[@]}" --batch_size 16 --epochs 64 # good (5points) perf.
# CUDA_VISIBLE_DEVICES=${device} python ./training/train_models/ACCS_csmri_and_ei_tvloss.py "${args[@]}" --batch_size 16 --epochs 35 # good perf. (10points) 30 epc
# CUDA_VISIBLE_DEVICES=${device} python ./training/train_models/ACCS_csmri_and_ei_tvloss_disc.py "${args[@]}" --batch_size 16 --epochs 60 # good perf. (10points) 30 epc
# CUDA_VISIBLE_DEVICES=${device} python ./training/train_models/ours_cplx.py "${args[@]}" --batch_size 16 --epochs 300 # good perf. (10points) 30 epc
# CUDA_VISIBLE_DEVICES=${device} python ./training/train_models/ACCS_csmri_and_ei_tvloss_fig1_sim.py "${args[@]}" --batch_size 16 --epochs 35 # good perf. (10points) 30 epc


# CUDA_VISIBLE_DEVICES=${device} python ./training/train_models/ACCS_csmri_and_ei_tvloss_prior.py "${args[@]}" --batch_size 16 --epochs 50 # good perf. (10points) 30 epc
# CUDA_VISIBLE_DEVICES=${device} python ./training/train_models/ACCS_csmri_and_ei_tvloss_tta.py "${args[@]}" --batch_size 16 --epochs 35 # good perf. (10points) 30 epc

args=(
    # Dataset options
    --headmotion None
    # --supervised_dataset_name "ixi_t1_periodic_slight"
    --supervised_dataset_name "ixi_t1_periodic_moderate"
    --unsupervised_dataset_name "MR_ART"
    --epochs 500 
    --lr 2e-4
    )
# CUDA_VISIBLE_DEVICES=${device} python ./training/train_models/ACCS_csmri_and_ei_tvloss_tta.py "${args[@]}" --batch_size 16 --epochs 10 # good perf. (10points) 30 epc

args=(
    --headmotion None
    # --supervised_dataset_name "ixi_t1_periodic_slight"
    --supervised_dataset_name "ixi_t1_periodic_heavy"
    --unsupervised_dataset_name "MR_ART"
    --epochs 500 
    --lr 5e-4
    )
# CUDA_VISIBLE_DEVICES=${device} python ./training/train_models/ACCS_csmri_and_ei_tvloss_tta.py "${args[@]}" --batch_size 16 --epochs 35 #TODO need fintune.

args=(
    --headmotion None
    # --supervised_dataset_name "ixi_t1_periodic_slight"
    --supervised_dataset_name "ixi_t1_linear_moderate"
    --unsupervised_dataset_name "MR_ART"
    --epochs 500 
    --lr 2e-4
    )
# CUDA_VISIBLE_DEVICES=${device} python ./training/train_models/ACCS_csmri_and_ei_tvloss_tta.py "${args[@]}" --batch_size 16 --epochs 10 # 🧨🧨🧨

args=(
    --headmotion None
    # --supervised_dataset_name "ixi_t1_periodic_slight"
    --supervised_dataset_name "ixi_t1_nonlinear_moderate"
    --unsupervised_dataset_name "MR_ART"
    --epochs 500 
    --lr 2e-4
    )
# CUDA_VISIBLE_DEVICES=${device} python ./training/train_models/ACCS_csmri_and_ei_tvloss_tta.py "${args[@]}" --batch_size 16 --epochs 10 # good perf. (10points) 30 epc

args=(
    --headmotion None
    # --supervised_dataset_name "ixi_t1_periodic_slight"
    --supervised_dataset_name "ixi_t1_sudden_moderate"
    --unsupervised_dataset_name "MR_ART"
    --epochs 500 
    --lr 5e-4
    )
# CUDA_VISIBLE_DEVICES=${device} python ./training/train_models/ACCS_csmri_and_ei_tvloss_tta.py "${args[@]}" --batch_size 16 --epochs 35 #TODO need fintune.

args=(
    --headmotion None
    # --supervised_dataset_name "ixi_t1_periodic_slight"
    --supervised_dataset_name "ixi_t1_singleshot_moderate"
    --unsupervised_dataset_name "MR_ART"
    --epochs 500 
    --lr 5e-4
    )
# CUDA_VISIBLE_DEVICES=${device} python ./training/train_models/ACCS_csmri_and_ei_tvloss_tta.py "${args[@]}" --batch_size 16 --epochs 35 # #TODO need fintune.

args=(
    --headmotion None
    # --supervised_dataset_name "ixi_t1_periodic_slight"
    --supervised_dataset_name "ixi_t2_periodic_moderate"
    --unsupervised_dataset_name "MR_ART"
    --epochs 500 
    --lr 5e-4
    )
# CUDA_VISIBLE_DEVICES=${device} python ./training/train_models/ACCS_csmri_and_ei_tvloss_tta.py "${args[@]}" --batch_size 16 --epochs 35 # #TODO need fintune.
# maybe add tv loss can improve psnr/ssim. or add motion-corrupted to cs-ei; and image merge using image consistent.

args=(
    --headmotion None
    --supervised_dataset_name "ixi_pd_periodic_moderate"
    --unsupervised_dataset_name "MR_ART"
    --epochs 500 
    --lr 5e-4
    )
# CUDA_VISIBLE_DEVICES=${device} python ./training/train_models/ACCS_csmri_and_ei_tvloss_tta.py "${args[@]}" --batch_size 16 --epochs 15

args=(
    --headmotion None
    --supervised_dataset_name "stanford_knee_axial_pd_periodic_slight"
    --unsupervised_dataset_name "MR_ART"
    --epochs 500 
    --lr 5e-4
    )
# CUDA_VISIBLE_DEVICES=${device} python ./training/train_models/ACCS_csmri_and_ei_tvloss_tta.py "${args[@]}" --batch_size 16 --epochs 15  # ! bad quan, but good qua

args=(
    --headmotion None
    --supervised_dataset_name "fastmribrain_t1_periodic_slight"
    --unsupervised_dataset_name "MR_ART"
    --epochs 500 
    --lr 5e-4
    )
# CUDA_VISIBLE_DEVICES=${device} python ./training/train_models/ACCS_csmri_and_ei_tvloss_tta.py "${args[@]}" --batch_size 16 --epochs 15  

args=(
    --headmotion None
    --supervised_dataset_name "mrb13_t1_sudden_sslight"
    --unsupervised_dataset_name "MR_ART"
    --epochs 500 
    --lr 5e-4
    )
# CUDA_VISIBLE_DEVICES=${device} python ./training/train_models/ACCS_csmri_and_ei_tvloss_tta.py "${args[@]}" --batch_size 16 --epochs 15  # ! data need.

args=(
    --headmotion None
    --supervised_dataset_name "mrb13_t1_sudden_sslight"
    --unsupervised_dataset_name "MR_ART"
    --epochs 500 
    --lr 5e-4
    )
# CUDA_VISIBLE_DEVICES=${device} python ./training/train_models/ACCS_csmri_and_ei_tvloss_tta_maart.py "${args[@]}" --batch_size 16 --epochs 15  # ! data need.
