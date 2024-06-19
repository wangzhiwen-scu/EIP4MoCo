# https://github.com/zzzqzhou/Dual-Normalization/blob/16d6c307b6b16b950db1b37ef321cb376aabb9cb/bezier_curve.py
import numpy as np
import random
import matplotlib.pyplot as plt
try:
    from scipy.special import comb
except:
    from scipy.misc import comb


def bernstein_poly(i, n, t):
    """
     The Bernstein polynomial of n, i as a function of t
    """
    return comb(n, i) * ( t**(n-i) ) * (1 - t)**i


def bezier_curve(points, nTimes=1000):
    """
       Given a set of control points, return the
       bezier curve defined by the control points.
       Control points should be a list of lists, or list of tuples
       such as [ [1,1], 
                 [2,3], 
                 [4,5], ..[Xn, Yn] ]
        nTimes is the number of time steps, defaults to 1000
        See http://processingjs.nihongoresources.com/bezierinfo/
    """

    nPoints = len(points)
    xPoints = np.array([p[0] for p in points])
    yPoints = np.array([p[1] for p in points])

    t = np.linspace(0.0, 1.0, nTimes)

    polynomial_array = np.array([bernstein_poly(i, nPoints-1, t) for i in range(0, nPoints)])

    xvals = np.dot(xPoints, polynomial_array)
    yvals = np.dot(yPoints, polynomial_array)

    return xvals, yvals


def nonlinear_transformation(x, prob=0.5):
    if random.random() >= prob:
        return x
    points = [[0, 0], [random.random(), random.random()], [random.random(), random.random()], [1, 1]]
    xvals, yvals = bezier_curve(points, nTimes=100000)
    if random.random() < 0.5:
        # Half change to get flip
        xvals = np.sort(xvals)
    else:
        xvals, yvals = np.sort(xvals), np.sort(yvals)
    nonlinear_x = np.interp(x, xvals, yvals)
    return nonlinear_x


# https://github.com/zzzqzhou/Dual-Normalization/blob/8cac04774664f90f37b0d342305c8051bc141589/preprocess_func.py

def norm(slices):
    max = np.max(slices)
    min = np.min(slices)
    slices = 2 * (slices - min) / (max - min) - 1
    return slices

def norm_1(slices):
    max = np.max(slices)
    min = np.min(slices)
    slices = 1.0 * (slices - min) / (max - min)
    return slices

# slices = norm(slices)
# slices, nonlinear_slices_1, nonlinear_slices_2, \
# nonlinear_slices_3, nonlinear_slices_4, nonlinear_slices_5 = nonlinear_transformation(slices)

def nonlinear_transformation(slices):
    slices = norm(slices)
    points_1 = [[-1, -1], [-1, -1], [1, 1], [1, 1]]
    xvals_1, yvals_1 = bezier_curve(points_1, nTimes=100000)
    xvals_1 = np.sort(xvals_1)

    points_2 = [[-1, -1], [-0.5, 0.5], [0.5, -0.5], [1, 1]]
    xvals_2, yvals_2 = bezier_curve(points_2, nTimes=100000)
    xvals_2 = np.sort(xvals_2)
    yvals_2 = np.sort(yvals_2)

    points_3 = [[-1, -1], [-0.5, 0.5], [0.5, -0.5], [1, 1]]
    xvals_3, yvals_3 = bezier_curve(points_3, nTimes=100000)
    xvals_3 = np.sort(xvals_3)

    points_4 = [[-1, -1], [-0.75, 0.75], [0.75, -0.75], [1, 1]]
    xvals_4, yvals_4 = bezier_curve(points_4, nTimes=100000)
    xvals_4 = np.sort(xvals_4)
    yvals_4 = np.sort(yvals_4)

    points_5 = [[-1, -1], [-0.75, 0.75], [0.75, -0.75], [1, 1]]
    xvals_5, yvals_5 = bezier_curve(points_5, nTimes=100000)
    xvals_5 = np.sort(xvals_5)

    """
    slices, nonlinear_slices_2, nonlinear_slices_4 are source-similar images
    nonlinear_slices_1, nonlinear_slices_3, nonlinear_slices_5 are source-dissimilar images
    """
    nonlinear_slices_1 = np.interp(slices, xvals_1, yvals_1)
    nonlinear_slices_1[nonlinear_slices_1 == 1] = -1
    
    nonlinear_slices_2 = np.interp(slices, xvals_2, yvals_2)

    nonlinear_slices_3 = np.interp(slices, xvals_3, yvals_3)
    nonlinear_slices_3[nonlinear_slices_3 == 1] = -1

    nonlinear_slices_4 = np.interp(slices, xvals_4, yvals_4)

    nonlinear_slices_5 = np.interp(slices, xvals_5, yvals_5)
    nonlinear_slices_5[nonlinear_slices_5 == 1] = -1

    slices=norm_1(slices)
    nonlinear_slices_1=norm_1(nonlinear_slices_1)
    nonlinear_slices_2=norm_1(nonlinear_slices_2) # dont need skull
    nonlinear_slices_3=norm_1(nonlinear_slices_3)
    nonlinear_slices_4=norm_1(nonlinear_slices_4) # dont need skull
    nonlinear_slices_5=norm_1(nonlinear_slices_5)

    nonlinear_slices_1[nonlinear_slices_1>0.9] = 0
    nonlinear_slices_3[nonlinear_slices_3>0.9] = 0
    nonlinear_slices_5[nonlinear_slices_5>0.9] = 0

    return [slices, nonlinear_slices_1, nonlinear_slices_2, \
           nonlinear_slices_3, nonlinear_slices_4, nonlinear_slices_5]



if __name__ == "__main__":

    def rgb_to_gray(rgb_image):
        gray = 0.2989 * rgb_image[0,:,:] + 0.5870 * rgb_image[1, :,:] + 0.1140 * rgb_image[2, :,:]
        return gray
    
    img_path = "./data/yuhui/PT_127_2013.06.18T1c.npy"
    img = np.load(img_path)
    # img = rgb_to_gray(img)

    img_list = nonlinear_transformation(img)

    for idx, img in enumerate(img_list):
        fig = plt.figure(figsize=(6,6), frameon=False) # 6x6 is just an example size, adjust as needed
        ax = plt.Axes(fig, [0., 0., 1., 1.], )
        ax.set_axis_off()
        fig.add_axes(ax)
        
        ax.imshow(img, cmap='gray')  # Assuming your images are grayscale, if not, remove cmap
        fig.savefig(f'./data/yuhui/image_{idx}.png', dpi=300)
        plt.close(fig)