#! usr/bin/env bash
args=(
    # Dataset options
    --headmotion None
    # --supervised_dataset_name "ixi_t1_periodic_slight"
    --supervised_dataset_name "ixi_t1_periodic_slight"
    --unsupervised_dataset_name "MR_ART"
    --epochs 40 
    --lr 5e-4
    --batch_size 10
    )
device=0


#TODO abaltion 1 (a) sampling ratio
# ablation_which: ablation1, ablation2, ablation3
# ablation_var: ablation1-[20, 30, 40, 50, 70, 80, 90]; ablation2-['equidistant', 'gaussian']; ablation3-[24, 34, 54, 64, 74, 84]
# ablation1 60 is our; ablation3 44 is our.

# @ ablation1
# CUDA_VISIBLE_DEVICES=${device} python ./training/train_models/ablation1_ei/csmri_rotcsmri_maskablation1.py "${args[@]}" --ablation_which 'ablation1' --ablation_var '20'
# CUDA_VISIBLE_DEVICES=${device} python ./training/train_models/ablation1_ei/csmri_rotcsmri_maskablation1.py "${args[@]}" --ablation_which 'ablation1' --ablation_var '30'
# CUDA_VISIBLE_DEVICES=${device} python ./training/train_models/ablation1_ei/csmri_rotcsmri_maskablation1.py "${args[@]}" --ablation_which 'ablation1' --ablation_var '40'
# CUDA_VISIBLE_DEVICES=${device} python ./training/train_models/ablation1_ei/csmri_rotcsmri_maskablation1.py "${args[@]}" --ablation_which 'ablation1' --ablation_var '50'
# CUDA_VISIBLE_DEVICES=${device} python ./training/train_models/ablation1_ei/csmri_rotcsmri_maskablation1.py "${args[@]}" --ablation_which 'ablation1' --ablation_var '70'
# CUDA_VISIBLE_DEVICES=${device} python ./training/train_models/ablation1_ei/csmri_rotcsmri_maskablation1.py "${args[@]}" --ablation_which 'ablation1' --ablation_var '80'
# CUDA_VISIBLE_DEVICES=${device} python ./training/train_models/ablation1_ei/csmri_rotcsmri_maskablation1.py "${args[@]}" --ablation_which 'ablation1' --ablation_var '90'


#TODO ablation 2 (b) sampling strategy
# @ ablation2
# CUDA_VISIBLE_DEVICES=${device} python ./training/train_models/ablation1_ei/csmri_rotcsmri_maskablation1.py "${args[@]}" --ablation_which 'ablation2' --ablation_var 'equidistant'
# CUDA_VISIBLE_DEVICES=${device} python ./training/train_models/ablation1_ei/csmri_rotcsmri_maskablation1.py "${args[@]}" --ablation_which 'ablation2' --ablation_var 'gaussian'

# @ ablation3
# CUDA_VISIBLE_DEVICES=${device} python ./training/train_models/ablation1_ei/csmri_rotcsmri_maskablation1.py "${args[@]}" --ablation_which 'ablation3' --ablation_var '24'
# CUDA_VISIBLE_DEVICES=${device} python ./training/train_models/ablation1_ei/csmri_rotcsmri_maskablation1.py "${args[@]}" --ablation_which 'ablation3' --ablation_var '34'
# CUDA_VISIBLE_DEVICES=${device} python ./training/train_models/ablation1_ei/csmri_rotcsmri_maskablation1.py "${args[@]}" --ablation_which 'ablation3' --ablation_var '54'
# CUDA_VISIBLE_DEVICES=${device} python ./training/train_models/ablation1_ei/csmri_rotcsmri_maskablation1.py "${args[@]}" --ablation_which 'ablation3' --ablation_var '64'
# CUDA_VISIBLE_DEVICES=${device} python ./training/train_models/ablation1_ei/csmri_rotcsmri_maskablation1.py "${args[@]}" --ablation_which 'ablation3' --ablation_var '74'
# CUDA_VISIBLE_DEVICES=${device} python ./training/train_models/ablation1_ei/csmri_rotcsmri_maskablation1.py "${args[@]}" --ablation_which 'ablation3' --ablation_var '84'

#TODO ablation 3 (c) EIP learning
# CUDA_VISIBLE_DEVICES=${device} python ./training/train_models/ablation3_ei/csmri.py "${args[@]}" --batch_size 10 --epochs 30 # loss stop
# CUDA_VISIBLE_DEVICES=${device} python ./training/train_models/ablation3_ei/ei.py "${args[@]}" --batch_size 10 --epochs 43 # loss stop
# CUDA_VISIBLE_DEVICES=${device} python ./training/train_models/ablation3_ei/rotcsmri.py "${args[@]}" --batch_size 10 --epochs 47 # loss stop

# CUDA_VISIBLE_DEVICES=${device} python ./training/train_models/ablation3_ei/csmri_ei.py "${args[@]}" --batch_size 10 --epochs 40 # loss stop
# CUDA_VISIBLE_DEVICES=${device} python ./training/train_models/ablation3_ei/csmri_rotcsmri.py "${args[@]}" --batch_size 10 --epochs 40 # loss stop
# CUDA_VISIBLE_DEVICES=${device} python ./training/train_models/ablation3_ei/ei_rotcsmri.py "${args[@]}" --batch_size 10 --epochs 40 # loss stop
CUDA_VISIBLE_DEVICES=${device} python ./training/train_models/ablation3_ei/ei_rotcsmri.py "${args[@]}" --batch_size 10 --epochs 20 --load_model 1 # loss stop

# CUDA_VISIBLE_DEVICES=${device} python ./training/train_models/ablation3_ei/csmri_ei_rotcsmri.py "${args[@]}" --batch_size 10 --epochs 90 --load_model 1 # loss stop

#TODO ablation 4 (d) backbone

#TODO ablation 5 (e) modality augmentation

#TODO ablation 6 (f) loss function
# CUDA_VISIBLE_DEVICES=${device} python ./training/train_models/ablation6_vgg/csmri_newei_spatial.py "${args[@]}" --batch_size 10 --epochs 90 --vggloss_weight 0 # loss stop
# CUDA_VISIBLE_DEVICES=${device} python ./training/train_models/ablation6_vgg/csmri_newei_spatial.py "${args[@]}" --batch_size 10 --epochs 90 --vggloss_weight 1e-3 # loss stop
# CUDA_VISIBLE_DEVICES=${device} python ./training/train_models/ablation6_vgg/csmri_newei_spatial.py "${args[@]}" --batch_size 10 --epochs 90 --vggloss_weight 1e-4 # loss stop

