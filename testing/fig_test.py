# generate from GPT4.
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import ascent

# Load the ascent dataset
phantom = ascent()

# Create 6 copies of the phantom image with different rotation angles

def plot_images(images, rows, cols):
    fig, axes = plt.subplots(rows, cols, figsize=(12, 6))
    
    for i, ax in enumerate(axes.flat):
        row = i // cols
        col = i % cols

        if row == 0:  # First row
            img = images[col]
            cmap = None
        elif row == 1:  # Second row (zoomed-in region of the corresponding image in the first row)
            img = images[col][100:200, 100:200]
            cmap = None
        elif row == 2:  # Last row
            img = images[col]
            cmap = 'jet'
        else:
            ax.axis('off')
            continue

        ax.imshow(img, cmap=cmap)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.axis('off')

    plt.subplots_adjust(wspace=0, hspace=0)

    save_path = './results/test.png'
    plt.savefig(save_path, bbox_inches='tight')

def plot_images_bbox(images, rows, cols):
    fig, axes = plt.subplots(rows, cols, figsize=(12, 6))

    # Define bounding box coordinates for the zoomed-in region
    x1, y1, x2, y2 = 100, 100, 200, 200

    for i, ax in enumerate(axes.flat):
        row = i // cols
        col = i % cols

        if row == 0:  # First row
            img = images[col]
            cmap = None
            ax.imshow(img, cmap=cmap)
            # Draw bounding box
            rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, edgecolor='red', linewidth=2, fill=False)
            ax.add_patch(rect)
        elif row == 1:  # Second row (zoomed-in region of the corresponding image in the first row)
            img = images[col][y1:y2, x1:x2]
            cmap = None
            ax.imshow(img, cmap=cmap)
        elif row == 2:  # Last row
            img = images[col]
            cmap = 'jet'
            ax.imshow(img, cmap=cmap)
        else:
            ax.axis('off')
            continue

        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.axis('off')
    plt.subplots_adjust(wspace=0, hspace=0)

    save_path = './results/test_bbox.png'
    plt.savefig(save_path, bbox_inches='tight')

def plot_images_bbox_1and2_is_gray(images, rows, cols):
    fig, axes = plt.subplots(rows, cols, figsize=(12, 6))

    # Define bounding box coordinates for the zoomed-in region
    x1, y1, x2, y2 = 100, 100, 200, 200

    for i, ax in enumerate(axes.flat):
        row = i // cols
        col = i % cols

        if row == 0:  # First row
            img = images[col]
            cmap = 'gray'
            ax.imshow(img, cmap=cmap)
            # Draw bounding box
            if col == 0:
                rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, edgecolor='red', linewidth=1, fill=False)
                ax.add_patch(rect)
        elif row == 2:  # Second row (zoomed-in region of the corresponding image in the first row)
            img = images[col][y1:y2, x1:x2]
            cmap = 'gray'
            vmin, vmax = np.min(img), np.max(img)
            ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
        elif row == 1:  # Last row
            img = images[col]
            cmap = 'seismic'
            ax.imshow(img, cmap=cmap)
        else:
            ax.axis('off')
            continue

        # elif row == 2:  # Last row (zoomed-in NRMSE)
        #     ref_img = images[0][y1:y2, x1:x2]  # Reference ROI from the first column
        #     cur_img = images[col][y1:y2, x1:x2]  # Current

        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.axis('off')
    plt.subplots_adjust(wspace=0, hspace=0)

    save_path = './results/test_bbox_1and2_isgray.png'
    plt.savefig(save_path, bbox_inches='tight')

def plot_images_bbox_1and2_is_gray_colorbar(images, rows, cols):
    fig, axes = plt.subplots(rows, cols+1, figsize=(12, 6), gridspec_kw={'width_ratios': [1]*cols + [0.05]})

    # Define bounding box coordinates for the zoomed-in region
    x1, y1, x2, y2 = 100-20, 100-20, 200, 200

    for i, ax in enumerate(axes.flat[:-1]):
        row = i // (cols + 1 )
        col = i % (cols + 1)

        if row == 0 and col < len(images):  # First row:  # First row
            img = images[col]
            cmap = 'gray'
            im = ax.imshow(img, cmap=cmap)
            # Draw bounding box
            if col == 0:
                rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, edgecolor='red', linewidth=1, fill=False)
                ax.add_patch(rect)
        elif row == 2 and col < len(images):  # Second row (zoomed-in region of the corresponding image in the first row)
            img = images[col][y1:y2, x1:x2]
            cmap = 'gray'
            vmin, vmax = np.min(img), np.max(img)
            im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
            # Add arrow to reference region
            ax.arrow(40, 40, 20, 20, color='white', width=0.01, head_width=5, length_includes_head=True)
            # ax.arrow(120, 80, -20, 20, color='white', width=0.01, head_width=5, length_includes_head=True)
            # ax.arrow(80, 120, 20, -20, color='white', width=0.01, head_width=5, length_includes_head=True)
            # ax.arrow(120, 120, -20, -20, color='white', width=0.01, head_width=5, length_includes_head=True)

        elif row == 1 and col < len(images):  # Last row
            img = images[col]
            cmap = 'seismic'
            im = ax.imshow(img, cmap=cmap)
        else:
            ax.axis('off')
            continue

        # elif row == 2:  # Last row (zoomed-in NRMSE)
        #     ref_img = images[0][y1:y2, x1:x2]  # Reference ROI from the first column
        #     cur_img = images[col][y1:y2, x1:x2]  # Current

        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.axis('off')

    # Add colorbars
    # for row in range(rows):
    #     cax = axes[row, -1]
    #     plt.colorbar(im, cax=cax, orientation='vertical')
    # Add colorbars for each row
    # Add colorbars for each row
    for row in range(rows):
        # Create a new axes object for the colorbar
        cax = fig.add_axes([0.92, 0.15 + (0.27 * row), 0.02, 0.05])
        # Plot the colorbar and set the orientation and labels
        plt.colorbar(im, cax=cax, orientation='vertical')
        cax.set_title('Row ' + str(row+1))
        cax.tick_params(labelsize=8)

    # fig.tight_layout()
        
    plt.subplots_adjust(wspace=0, hspace=0)
    save_path = './results/test_bbox_1and2_isgray_colorbar_updown.png'
    plt.savefig(save_path, bbox_inches='tight')

sample_images = [np.rot90(phantom, k=i) for i in range(6)]
plot_images_bbox_1and2_is_gray_colorbar(sample_images, 3, 6)

# img1 = np.random.randn(8,8)
# img2 = np.random.randn(8,8)
# img3 = np.random.randn(8,8)

# b_max = np.sqrt(np.sum(np.square(img1)))
# E_corrected = -np.sum( (img1/b_max) * np.log2(img/b_max))

# E_motion = -np.sum( (img/b_max) * np.log2(img/b_max))
# E_still = -np.sum( (img/b_max) * np.log2(img/b_max))

# Q_motion = (E_corrected - E_motion) / (E_still -  E_motion)
