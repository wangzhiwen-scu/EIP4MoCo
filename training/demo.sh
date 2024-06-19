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

# testing ei
# CUDA_VISIBLE_DEVICES=${device} python ./training/train_models/ei_baseline.py "${args[@]}" --batch_size 10 --epochs 60
# CUDA_VISIBLE_DEVICES=${device} python ./training/train_models/csmri_ei.py "${args[@]}" --batch_size 10 --epochs 40
# CUDA_VISIBLE_DEVICES=${device} python ./training/train_models/csmri_ei_bezier.py "${args[@]}" --batch_size 10 --epochs 57  # loss stop
# CUDA_VISIBLE_DEVICES=${device} python ./training/train_models/csmri_ei_bezier.py "${args[@]}" --batch_size 10 --epochs 30 --load_model 1  # loss stop
# CUDA_VISIBLE_DEVICES=${device} python ./training/train_models/csmri_ei_bezier.py "${args[@]}" --batch_size 10 --epochs 77 # loss stop psnr ssim so high
# CUDA_VISIBLE_DEVICES=${device} python ./training/train_models/csmri_ei.py "${args[@]}" --batch_size 10 --epochs 120 # loss stop
# CUDA_VISIBLE_DEVICES=${device} python ./training/train_models/csmri_newei.py "${args[@]}" --batch_size 10 --epochs 90 # loss stop

# CUDA_VISIBLE_DEVICES=${device} python ./training/train_models/csmri_newei_singleshot.py "${args[@]}" --batch_size 10 --epochs 60 --load_model 1 # loss stop

# CUDA_VISIBLE_DEVICES=${device} python ./training/train_models/csmri_newei_bezier.py "${args[@]}" --batch_size 10 --epochs 90 # loss stop

# CUDA_VISIBLE_DEVICES=${device} python ./training/train_models/csmri_newei_bezier.py "${args[@]}" --batch_size 10 --epochs 90 # loss stop

# CUDA_VISIBLE_DEVICES=${device} python ./training/train_models/csmri_newei_spatial.py "${args[@]}" --batch_size 10 --epochs 90  # loss stop

# CUDA_VISIBLE_DEVICES=${device} python ./training/train_models/csmri_newei_spatial_lessepoch.py "${args[@]}" --batch_size 10 --epochs 70  # loss stop

# CUDA_VISIBLE_DEVICES=${device} python ./training/train_models/csmri_newei_spatial_rotate.py "${args[@]}" --batch_size 10 --epochs 90  # loss stop
CUDA_VISIBLE_DEVICES=${device} python ./training/train_models/csmri_newei_spatial_norotate.py "${args[@]}" --batch_size 10 --epochs 90  # loss stop



# ablation
# CUDA_VISIBLE_DEVICES=${device} python ./training/train_models/csmri_and_ei_tvloss.py "${args[@]}" --batch_size 16 --epochs 64 # good (5points) perf.
# CUDA_VISIBLE_DEVICES=${device} python ./training/train_models/ACCS_csmri_and_ei_tvloss.py "${args[@]}" --batch_size 16 --epochs 35 # good perf. (10points) 30 epc
# CUDA_VISIBLE_DEVICES=${device} python ./training/train_models/ACCS_csmri_and_ei_tvloss_disc.py "${args[@]}" --batch_size 16 --epochs 60 # good perf. (10points) 30 epc
# CUDA_VISIBLE_DEVICES=${device} python ./training/train_models/ours_cplx.py "${args[@]}" --batch_size 16 --epochs 300 # good perf. (10points) 30 epc
# CUDA_VISIBLE_DEVICES=${device} python ./training/train_models/ACCS_csmri_and_ei_tvloss_fig1_sim.py "${args[@]}" --batch_size 16 --epochs 35 # good perf. (10points) 30 epc


# CUDA_VISIBLE_DEVICES=${device} python ./training/train_models/ACCS_csmri_and_ei_tvloss_prior.py "${args[@]}" --batch_size 16 --epochs 50 # good perf. (10points) 30 epc
# CUDA_VISIBLE_DEVICES=${device} python ./training/train_models/ACCS_csmri_and_ei_tvloss_tta.py "${args[@]}" --batch_size 16 --epochs 35 # good perf. (10points) 30 epc

# CUDA_VISIBLE_DEVICES=${device} python ./training/train_models/ACCS_csmri_and_ei_tvloss_prior_rlandap.py "${args[@]}" --batch_size 16 --epochs 50 # good perf. (10points) 30 epc

# CUDA_VISIBLE_DEVICES=${device} python ./training/train_models/ACCS_csmri_and_ei_tvloss_prior_2.py "${args[@]}" --batch_size 16 --epochs 50 # good perf. (10points) 30 epc
# CUDA_VISIBLE_DEVICES=${device} python ./training/train_models/ACCS_csmri_and_ei_tvloss_prior_rlandap_csmri.py "${args[@]}" --batch_size 16 --epochs 50 --load_model 1 # good perf. (10points) 30 epc

# good perf. Need 120 training when T1->T2 (remaining artifacts), but 40 good for MR-ART.
# CUDA_VISIBLE_DEVICES=${device} python ./training/train_models/ACCS_csmri_and_ei_tvloss_prior_bezier.py "${args[@]}" --batch_size 16 --epochs 40  
# CUDA_VISIBLE_DEVICES=${device} python ./training/train_models/ACCS_csmri_and_ei_tvloss_prior_bezier_mse.py "${args[@]}" --batch_size 16 --epochs 130  

# CUDA_VISIBLE_DEVICES=${device} python ./training/train_models/ACCS_csmri_and_ei_tvloss_prior_sampcenter.py "${args[@]}" --batch_size 16 --epochs 18 --lr 4e-4
# CUDA_VISIBLE_DEVICES=${device} python ./training/train_models/ACCS_csmri_and_ei_tvloss_prior_sampcenter.py "${args[@]}" --batch_size 16 --epochs 20 --lr 3e-4 --load_model 1 

# CUDA_VISIBLE_DEVICES=${device} python ./training/train_models/ACCS_csmri_and_ei_tvloss_prior_novgg.py "${args[@]}" --batch_size 16 --epochs 40

# CUDA_VISIBLE_DEVICES=${device} python ./training/train_models/ACCS_csmri_and_ei_tvloss_prior_csonly.py "${args[@]}" --batch_size 16 --epochs 40 --load_model 1

# CUDA_VISIBLE_DEVICES=${device} python ./training/train_models/ACCS_csmri_and_ei_tvloss_prior_unet.py "${args[@]}" --batch_size 16 --epochs 40 --load_model 1 

# CUDA_VISIBLE_DEVICES=${device} python ./training/train_models/ACCS_csmri_and_ei_tvloss_prior_MSTPP.py "${args[@]}" --batch_size 16 --epochs 40 

# CUDA_VISIBLE_DEVICES=${device} python ./training/train_models/ACCS_csmri_and_ei_tvloss_prior_restormer.py "${args[@]}" --batch_size 4 --epochs 15
# CUDA_VISIBLE_DEVICES=${device} python ./training/train_models/ACCS_csmri_and_ei_tvloss_prior_stripformer.py "${args[@]}" --batch_size 4 --epochs 40


# CUDA_VISIBLE_DEVICES=${device} python ./training/train_models/ACCS_csmri_and_ei_tvloss_prior_csonly_bezier.py "${args[@]}" --batch_size 16 --epochs 80
# CUDA_VISIBLE_DEVICES=${device} python ./training/train_models/ACCS_csmri_and_ei_tvloss_prior_newei_mse.py "${args[@]}" --batch_size 16 --epochs 50

args=(
    # Dataset options
    --headmotion None
    # --supervised_dataset_name "ixi_t1_periodic_slight"
    --supervised_dataset_name "ixi_t1_linear_moderate"
    --unsupervised_dataset_name "MR_ART"
    --epochs 500 
    --lr 5e-4
    )
device=0

# Start--------------------------------------------Start

# testing ei
# CUDA_VISIBLE_DEVICES=${device} python ./training/train_models/ei_baseline_motion_sslmotion.py "${args[@]}" --batch_size 12 --epochs 60
# CUDA_VISIBLE_DEVICES=${device} python ./training/train_models/csmri_ei_ft.py "${args[@]}" --batch_size 10 --epochs 40 --load_model 1 --lr 1e-4
# CUDA_VISIBLE_DEVICES=${device} python ./training/train_models/csmri_ei_tta.py "${args[@]}" --batch_size 10 --epochs 40 --load_model 1

args=(
    # Dataset options
    --headmotion None
    # --supervised_dataset_name "ixi_t1_periodic_slight"
    --supervised_dataset_name "ixi_t1_sudden_moderate"
    --unsupervised_dataset_name "MR_ART"
    --epochs 500 
    --lr 5e-4
    )
device=0

# Start--------------------------------------------Start

# testing ei
# CUDA_VISIBLE_DEVICES=${device} python ./training/train_models/ei_baseline_motion_sslmotion.py "${args[@]}" --batch_size 12 --epochs 60
# CUDA_VISIBLE_DEVICES=${device} python ./training/train_models/csmri_ei_ft.py "${args[@]}" --batch_size 10 --epochs 40 --load_model 1
# CUDA_VISIBLE_DEVICES=${device} python ./training/train_models/csmri_ei_tta.py "${args[@]}" --batch_size 10 --epochs 40 --load_model 1

args=(
    # Dataset options
    --headmotion None
    # --supervised_dataset_name "ixi_t1_periodic_slight"
    --supervised_dataset_name "ixi_t1_periodic_slight_rl"
    --unsupervised_dataset_name "MR_ART"
    --epochs 500 
    --lr 5e-4
    )
device=0

# Start--------------------------------------------Start

# testing ei
# CUDA_VISIBLE_DEVICES=${device} python ./training/train_models/ei_baseline_motion_sslmotion.py "${args[@]}" --batch_size 12 --epochs 60
# CUDA_VISIBLE_DEVICES=${device} python ./training/train_models/csmri_ei_tta.py "${args[@]}" --batch_size 10 --epochs 40 --load_model 1
# CUDA_VISIBLE_DEVICES=${device} python ./training/train_models/csmri_ei_tta.py "${args[@]}" --batch_size 10 --epochs 40 --load_model 1


args=(
    # Dataset options
    --headmotion None
    # --supervised_dataset_name "ixi_t1_periodic_slight"
    --supervised_dataset_name "ixi_t1_singleshot_moderate"
    --unsupervised_dataset_name "MR_ART"
    --epochs 500 
    --lr 5e-4
    )
device=0

# Start--------------------------------------------Start

# testing ei
# CUDA_VISIBLE_DEVICES=${device} python ./training/train_models/ei_baseline_motion_sslmotion.py "${args[@]}" --batch_size 12 --epochs 60
# CUDA_VISIBLE_DEVICES=${device} python ./training/train_models/csmri_ei_tta.py "${args[@]}" --batch_size 10 --epochs 40 --load_model 1

args=(
    # Dataset options
    --headmotion None
    # --supervised_dataset_name "ixi_t1_periodic_slight"
    --supervised_dataset_name "ixi_t1_periodic_moderate"
    --unsupervised_dataset_name "MR_ART"
    --epochs 500 
    --lr 5e-4
    )
device=0

# Start--------------------------------------------Start

# testing ei
# CUDA_VISIBLE_DEVICES=${device} python ./training/train_models/ei_baseline_motion_sslmotion.py "${args[@]}" --batch_size 12 --epochs 60
# CUDA_VISIBLE_DEVICES=${device} python ./training/train_models/csmri_ei_tta.py "${args[@]}" --batch_size 10 --epochs 40 --load_model 1

args=(
    # Dataset options
    --headmotion None
    # --supervised_dataset_name "ixi_t1_periodic_slight"
    --supervised_dataset_name "ixi_t1_periodic_heavy"
    --unsupervised_dataset_name "MR_ART"
    --epochs 500 
    --lr 5e-4
    )
device=0

# Start--------------------------------------------Start

# testing ei
# CUDA_VISIBLE_DEVICES=${device} python ./training/train_models/ei_baseline_motion_sslmotion.py "${args[@]}" --batch_size 12 --epochs 60
# CUDA_VISIBLE_DEVICES=${device} python ./training/train_models/csmri_ei_tta.py "${args[@]}" --batch_size 10 --epochs 40 --load_model 1


args=(
    # Dataset options
    --headmotion None
    # --supervised_dataset_name "ixi_t1_periodic_slight"
    --supervised_dataset_name "ixi_t2_periodic_slight"
    --unsupervised_dataset_name "MR_ART"
    --epochs 500 
    --lr 5e-4
    )
device=0
# CUDA_VISIBLE_DEVICES=${device} python ./training/train_models/ACCS_csmri_and_ei_tvloss_prior.py "${args[@]}" --batch_size 16 --epochs 50  # good perf. (10points) 30 epc
# CUDA_VISIBLE_DEVICES=${device} python ./training/train_models/ACCS_csmri_and_ei_tvloss_prior_bezier.py "${args[@]}" --batch_size 16 --epochs 200  # good perf. (10points) 30 epc
# CUDA_VISIBLE_DEVICES=${device} python ./training/train_models/ACCS_csmri_and_ei_tvloss_prior_csonly_bezier.py "${args[@]}" --batch_size 16 --epochs 80
# CUDA_VISIBLE_DEVICES=${device} python ./training/train_models/csmri_ei_tta.py "${args[@]}" --batch_size 10 --epochs 40 --load_model 1
# CUDA_VISIBLE_DEVICES=${device} python ./training/train_models/csmri_ei_bezier.py "${args[@]}" --batch_size 10 --epochs 80  # loss stop
# CUDA_VISIBLE_DEVICES=${device} python ./training/train_models/csmri_ei_bezier.py "${args[@]}" --batch_size 10 --epochs 30  # loss stop
# CUDA_VISIBLE_DEVICES=${device} python ./training/train_models/csmri_ei_bezier.py "${args[@]}" --batch_size 10 --epochs 50 # loss stop psnr ssim so high

# CUDA_VISIBLE_DEVICES=${device} python ./training/train_models/csmri_newei_bezier.py "${args[@]}" --batch_size 10 --epochs 90 # loss stop

args=(
    # Dataset options
    --headmotion None
    # --supervised_dataset_name "ixi_t1_periodic_slight"
    --supervised_dataset_name "ixi_pd_periodic_slight"
    --unsupervised_dataset_name "MR_ART"
    --epochs 500 
    --lr 5e-4
    )
device=0
# CUDA_VISIBLE_DEVICES=${device} python ./training/train_models/ACCS_csmri_and_ei_tvloss_prior.py "${args[@]}" --batch_size 16 --epochs 50  # good perf. (10points) 30 epc
# CUDA_VISIBLE_DEVICES=${device} python ./training/train_models/ACCS_csmri_and_ei_tvloss_prior_bezier.py "${args[@]}" --batch_size 16 --epochs 40  # good perf. (10points) 30 epc
#! find t1 bezier, what happen...
# CUDA_VISIBLE_DEVICES=${device} python ./training/train_models/ACCS_csmri_and_ei_tvloss_prior_csonly_bezier.py "${args[@]}" --batch_size 16 --epochs 80
# CUDA_VISIBLE_DEVICES=${device} python ./training/train_models/csmri_ei_tta.py "${args[@]}" --batch_size 10 --epochs 40 --load_model 1
# CUDA_VISIBLE_DEVICES=${device} python ./training/train_models/csmri_ei_bezier.py "${args[@]}" --batch_size 10 --epochs 80  # loss stop
# CUDA_VISIBLE_DEVICES=${device} python ./training/train_models/csmri_ei_bezier.py "${args[@]}" --batch_size 10 --epochs 30  # loss stop
# CUDA_VISIBLE_DEVICES=${device} python ./training/train_models/csmri_ei_bezier.py "${args[@]}" --batch_size 10 --epochs 50 # loss stop psnr ssim so high

# CUDA_VISIBLE_DEVICES=${device} python ./training/train_models/csmri_newei_bezier.py "${args[@]}" --batch_size 10 --epochs 90 # loss stop
