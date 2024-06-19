import matlab.engine
import scipy.io as sp
import cv2
import numpy as np

def NormalizeData(data, datarange=1):
    return datarange*(data - np.min(data)) / (np.max(data) - np.min(data))

eng = matlab.engine.start_matlab()

path = "./results/fig6ploseone/MR_ART_Axial_None/rec_motion_paried_IXI054-Guys-0707-T1.nii_753_mask.png.mat"
mat_contents = sp.loadmat(path)
img = mat_contents['img']
mask = mat_contents['mask']

img = img.astype(float)
mask = mask.astype(float)


eng.cd(r'./utils/matlab_splitbregman', nargout=0)

rec_img = eng.mysplitbregman(img, mask)
rec_img = np.array(rec_img)
rec_img = NormalizeData(rec_img, 255)

cv2.imwrite('./rec_img.png', rec_img)
