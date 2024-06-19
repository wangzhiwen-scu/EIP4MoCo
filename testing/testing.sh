#! usr/bin/env bash
device=0

# CUDA_VISIBLE_DEVICES=${device} python ./testing/fig1a1.py # done
# CUDA_VISIBLE_DEVICES=${device} python ./testing/fig1a2.py # done
# CUDA_VISIBLE_DEVICES=${device} python ./testing/fig1a3.py # psnr... random sudden 
# CUDA_VISIBLE_DEVICES=${device} python ./testing/fig1a4.py # done
# CUDA_VISIBLE_DEVICES=${device} python ./testing/fig1a5.py
# CUDA_VISIBLE_DEVICES=${device} python ./testing/fig1a6.py # done
# CUDA_VISIBLE_DEVICES=${device} python ./testing/fig1a7.py # done


# CUDA_VISIBLE_DEVICES=${device} python ./testing/fig2b1.py
# CUDA_VISIBLE_DEVICES=${device} python ./testing/fig2b2.py
# CUDA_VISIBLE_DEVICES=${device} python ./testing/fig2b3.py
# CUDA_VISIBLE_DEVICES=${device} python ./testing/fig2b4.py

# CUDA_VISIBLE_DEVICES=${device} python ./testing/fig2c1.py
# CUDA_VISIBLE_DEVICES=${device} python ./testing/fig2c2.py
# CUDA_VISIBLE_DEVICES=${device} python ./testing/fig2c3.py
# CUDA_VISIBLE_DEVICES=${device} python ./testing/fig2c4.py


CUDA_VISIBLE_DEVICES=${device} python ./testing/fig3a1.py
CUDA_VISIBLE_DEVICES=${device} python ./testing/fig3a2.py

## arch ablation
# CUDA_VISIBLE_DEVICES=${device} python ./testing/ablation/fig1a1_restormer.py
# CUDA_VISIBLE_DEVICES=${device} python ./testing/ablation/fig1a2_restormer.py
# CUDA_VISIBLE_DEVICES=${device} python ./testing/ablation/fig1a3_restormer.py
# CUDA_VISIBLE_DEVICES=${device} python ./testing/ablation/fig1a4_restormer.py
# CUDA_VISIBLE_DEVICES=${device} python ./testing/ablation/fig1a5_restormer.py

# CUDA_VISIBLE_DEVICES=${device} python ./testing/ablation/fig1b1_restormer.py
# CUDA_VISIBLE_DEVICES=${device} python ./testing/ablation/fig1b2_restormer.py

# CUDA_VISIBLE_DEVICES=${device} python ./testing/ablation/fig1a1_csonly.py
# CUDA_VISIBLE_DEVICES=${device} python ./testing/ablation/fig1a2_csonly.py
# CUDA_VISIBLE_DEVICES=${device} python ./testing/ablation/fig1a3_csonly.py
# CUDA_VISIBLE_DEVICES=${device} python ./testing/ablation/fig1a4_csonly.py
# CUDA_VISIBLE_DEVICES=${device} python ./testing/ablation/fig1a5_csonly.py

# CUDA_VISIBLE_DEVICES=${device} python ./testing/ablation/fig1b1_csonly.py
# CUDA_VISIBLE_DEVICES=${device} python ./testing/ablation/fig1b2_csonly.py


# # CUDA_VISIBLE_DEVICES=${device} python ./testing/ablation/fig1a1_cs_newei.py
# CUDA_VISIBLE_DEVICES=${device} python ./testing/ablation/fig1a2_cs_newei.py
# CUDA_VISIBLE_DEVICES=${device} python ./testing/ablation/fig1a3_cs_newei.py
# # CUDA_VISIBLE_DEVICES=${device} python ./testing/ablation/fig1a4_cs_newei.py
# CUDA_VISIBLE_DEVICES=${device} python ./testing/ablation/fig1a5_cs_newei.py

# CUDA_VISIBLE_DEVICES=${device} python ./testing/ablation/fig1b1_cs_newei.py
# CUDA_VISIBLE_DEVICES=${device} python ./testing/ablation/fig1b2_cs_newei.py

# CUDA_VISIBLE_DEVICES=${device} python ./testing/ablation2/fig2b1_cs_newei.py
# CUDA_VISIBLE_DEVICES=${device} python ./testing/ablation2/fig2b2_cs_newei.py
# CUDA_VISIBLE_DEVICES=${device} python ./testing/ablation2/fig2b3_cs_newei.py
# CUDA_VISIBLE_DEVICES=${device} python ./testing/ablation2/fig2b4_cs_newei.py

# CUDA_VISIBLE_DEVICES=${device} python ./testing/ablation2/fig2c1_cs_newei.py
# CUDA_VISIBLE_DEVICES=${device} python ./testing/ablation2/fig2c2_cs_newei.py
# CUDA_VISIBLE_DEVICES=${device} python ./testing/ablation2/fig2c3_cs_newei.py
# CUDA_VISIBLE_DEVICES=${device} python ./testing/ablation2/fig2c4_cs_newei.py



# CUDA_VISIBLE_DEVICES=${device} python ./testing/ablation1_sampling/ablation1_samplingratio.py
# CUDA_VISIBLE_DEVICES=${device} python ./testing/ablation1_sampling/ablation2a_samplingtypes.py
# CUDA_VISIBLE_DEVICES=${device} python ./testing/ablation1_sampling/ablation2b_centerratio.py
