import numpy as np
from scipy.stats import entropy

# https://github.com/nipreps/mriqc





def calculate_snr(image_data):
    """_summary_

    Calculate the SNR. The estimation may be provided with only one foreground 
    region in which the noise is computed as follows: SNR = \mu_F / (sigma_F*sqrt(n/(n-1))), 
    where mu_F is the mean intensity of the foreground and sigma_F  is the standard deviation 
    of the same region.  n (int) – number of voxels in foreground mask.

    """
    # create by gpt-4 from https://mriquestions.com/signal-to-noise.html
    # and https://www.nature.com/articles/s41597-022-01694-8#Sec2

    # https://mriqc.readthedocs.io/en/latest/iqms/t1w.html choose a patch to compute.

    image_size = 240

    # Define the top-mid region boundaries
    # top = 0
    # bottom = image_size // 2
    # left = image_size // 4
    # right = 3 * image_size // 4


    # Extract the patch
    top, bottom, left, right = 80, 80+40, 160, 160+40
    patch = image_data[top:bottom, left:right]

    # Compute signal and noise
    signal = np.mean(patch)
    noise = np.std(patch)

    # Compute SNR in dB
    snr = 10 * np.log10(signal / noise)

    return snr

def calculate_efc(image_slice):
    """_summary_ input below to GPT4 FROM https://mriqc.readthedocs.io/en/latest/iqms/t1w.html
        Calculate the EFC [Atkinson1997]. Uses the Shannon entropy of voxel intensities as an indication of ghosting and blurring induced by head motion. A range of low values is better, with EFC = 0 for all the energy concentrated in one pixel.
        E = -sum^N_j=1 = (x_j/x_max * ln(x_j/x_max)), with x_max=sqrt(sum^N_j=1 x^2_j),
        The original equation is normalized by the maximum entropy, so that the EFC can be compared across images with different dimensions:
        EFC = (N/sqrt(N) log sqrt(N)^(-1))*E
        img (numpy.ndarray) – input data
        framemask (numpy.ndarray) – a mask of empty voxels inserted after a rotation of data
    """
    # range into 0~256
    # image_slice = 255.0 * (image_slice - np.min(image_slice)) / (np.max(image_slice) - np.min(image_slice))

    # Define the frame mask (replace with your own mask)
    frame_mask = np.ones_like(image_slice, dtype=bool)

    # Apply the frame mask to the image slice
    masked_image_slice = image_slice * frame_mask

    # Calculate the square of the masked image slice
    squared_masked_image = masked_image_slice ** 2

    # Calculate x_max
    x_max = np.sqrt(np.sum(squared_masked_image))

    # Normalize the masked image slice by x_max
    normalized_masked_image = masked_image_slice / x_max

    # Calculate the entropy (E)
    E = -np.sum(normalized_masked_image * np.log(normalized_masked_image + np.finfo(float).eps))

    # Calculate the number of non-zero voxels (N)
    N = np.sum(frame_mask)

    # Calculate the normalized EFC using the provided formula
    efc_normalized = (N / np.sqrt(N) * np.log(np.sqrt(N))) * E

    return efc_normalized


def calculate_cjv(image_slice, wm_mask=None, gm_mask=None):

    if not wm_mask.any():
        # Define the intensity thresholds for white-matter and gray-matter
        wm_threshold = 0.7
        gm_threshold = 0.4
        # # Create the WM and GM masks using the intensity thresholds
        wm_mask, gm_mask = create_wm_gm_masks(image_slice, wm_threshold, gm_threshold)

    # Calculate the mean and standard deviation of signal intensities within each mask
    mu_wm, sigma_wm = np.mean(image_slice[np.where(wm_mask)]), np.std(image_slice[np.where(wm_mask)])
    mu_gm, sigma_gm = np.mean(image_slice[np.where(gm_mask)]), np.std(image_slice[np.where(gm_mask)])


    cjv = (sigma_wm + sigma_gm) / abs(mu_wm - mu_gm)
    return cjv

def create_wm_gm_masks(image_slice, wm_threshold, gm_threshold):
    # Create the binary masks based on the intensity thresholds

    
    wm_mask = image_slice > wm_threshold
    gm_mask = (image_slice > gm_threshold) & (image_slice <= wm_threshold)
    return wm_mask, gm_mask

# # Define the intensity thresholds for white-matter and gray-matter
# wm_threshold = 0.7
# gm_threshold = 0.4

# # Create the WM and GM masks using the intensity thresholds
# wm_mask, gm_mask = create_wm_gm_masks(image_slice, wm_threshold, gm_threshold)