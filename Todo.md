1. Read dataset 'IXI-simulation', 'fastMRI-simulation', 'MR-ART-real'
2. training 'IXI'; 
3. double-training 'IXI' & 'fastMRI'; 
4. double-training 'IXI' & 'MR-ART'

model: unet; SADNet; Transformer; physic-driven admm.
Todo: 
1. dataset: zerofilling-image to (256, 256), then resize (240, 240)
2. 2D motion and 3d motion hybrid.
3. 
4. Need SOTA blur network. xx hold on.
5. need same type motion blur. only degree(), transformation(), num_(); or only 2d.
6. padding???
7. Download SOTA motion correction github to run. or debulr model. regestrationnet + MNI net + motioncerrectionnet.
8. using kspace to motion correction.
9. The reason of low ssim, but high psnr is that (still, with motion) is unregestrated. and that may lead to network unable to learning.
10. Search: unregistration denoising, or deblur.
11. CSMRI as a proxy task. random-80% undersampling, 
12. Change SOTA GAN or what ever demotion. unparied denoising, unregistration denoising, semi-denoising, zero-shot denoising.
13. Motion2Motion.
14. Horizon and Vertical Mask , Hybrid training them.
15. CS-MRI and 2D simulation hybrid training them.
16. Reimplement CycleGAN med v.2; JCY cs-mri; MARC; Stack-Unets motion correction.
17. Ablation: simulation without tv-loss unsup; new backbone, such as (k-space img two domain, sampling rate decrease progressively, or 95% motion sampling); vgg16; seg+ADAIN (for reliebal tissue);
18. simulation: psnr;ssim; intro entropy; real: doctor-bulr, artifacts, tissue, and so on (JCY).
19. must to motion estimation: theta1,theata2, x1, y1, ; try duncan: >0.89, we can success.
https://arxiv.org/pdf/2301.01106.pdf 
https://aapm.onlinelibrary.wiley.com/doi/pdfdirect/10.1002/mp.16119 k-space..

https://www.researchgate.net/profile/Zhaolin-Chen-3/publication/338118261_Suppressing_motion_artefacts_in_MRI_using_an_Inception-ResNet_network_with_motion_simulation_augmentation/links/5e040875a6fdcc28373efe87/Suppressing-motion-artefacts-in-MRI-using-an-Inception-ResNet-network-with-motion-simulation-augmentation.pdf 

20. https://github.com/chandrakanth-jp/mri-motion-artifact-simulation/blob/master/utilities_motion_corruption.py 3D simulation.
21. 自己模拟， https://www.sciencedirect.com/science/article/pii/S1053811922005286?via%3Dihub#bib0022  2D simulation

22. Based on retroMOCO, 2d simulation and 3D simulation. add it to our framework. Most important. Addition motion artifacts through retroMOCO, or registration netwrok between slice.
23. Based on jcy cs-mri pretrain, we can get sum of different event-bulr image, so we can using event-debulr it.
24. Based on Stack-Unets motion correction. we can better than it in real motion and unseen sythesis data.
25. Based on MARC, we can use pactch-csmri to pretrain, then to de-artifacts. Sup using artifacts as lable, and unsup using clear image to tvloss.
25. https://github.com/nalinimsingh/neuroMoCo berkely
26. Add shapness using adin(segmentation).
27. cs-mri using ssdu, ie., self2self mri. wss.
28. 模拟的运动都不是真实运动，可以用图像域的方法来处理。模拟的旋转中心的假设是图中心，真实旋转不一定是图中心，因此受到一定限制；可以使用图像域的方法来做；并且引入了ssdu pretrain，所以不需要高清图像。
29. 使用1D MRI 训练，或者automap，自动k空间校正，在加上k空间下采样技术（nips22）。这样就不存在预测运动参数了。
30. 1st idea: 先设计3slice 的 dataset。有效果了再说。先去伪影（3D 或者 2D），然后整个volume配准（voxmorph），然后refine。
31. 2nd idea: dual dc motion: using trusted k-point to dc; untrusted k-point using acs calibration; two-channel refer to vedio 3-channel process (3frame).
32. best score: cs-mri + real ei + real motion tv.
33. recommended k-point, throught a convLayer; unrecommended throughout acs layer. trusted thresh in 0.5. or learned. pretrain-csmri is also do it (i.e., do not using mask, using conv(k_us)->trusted k-space, ).
34. change backbone, 3 attention, or transformer. (supersolition)
35. cycleGAN training, simulation trainning (2D). -> DUNCAN.
36. ⭕1: SSL when TTA, ⭕2: stochastic restore, 🧷3: aver. label, ⭕4: aver. ema teacher update. 🧷5. Data augumentation . Ablation study.