"""
test ADMM algrithom
"""
import numpy as np
import matplotlib.pyplot as plt

import scipy.io as sio
import matplotlib.cm as cm

# import sys
# sys.path.append('.')
# import pics.proximal_func as pf
# import pics.CS_MRI_solvers_func as solvers

def BacktrackingLineSearch( f, df, x, p, c = 0.0001, rho = 0.2, ls_Nite = 10 ):
    """
    Backtracking linesearch
    f: function
    x: current point
    p: direction of search
    df: gradient of f at x
    """
    #derphi = np.real(np.sum(np.multiply(p, df(x))))
    #derphi = np.real(np.dot(p.flatten(),df(x)).flatten())
    # for complex vector dot product, conj may be needed
    #
    derphi = np.real(np.dot(p.flatten(),np.conj(df(x)).flatten())) 
    #print("m %g" % derphi)
    f0 = f(x)
    alphak = 1.0
    f_try  = f(x + alphak * p)
    i = 0
    #Loop
    while i < ls_Nite and f_try-f0 >  c * alphak * derphi and f_try > f0 :
        alphak = alphak * rho 
        f_try  = f(x + alphak * p)   
        i += 1
        #print("f(x+ap)-f(x) %g" % (f_try-f0))
    #print(i)
    return alphak, i

def prox_l2_Afxnb_CGD( Afunc, invAfunc, b, x0, rho, Nite, ls_Nite = 10 ):
    #x = np.zeros(x0.shape)
    eps = 0.001
    i = 0
    def f(xi):
        return np.linalg.norm(Afunc(xi)-b)**2 + (rho/2)*np.linalg.norm(xi-x0)**2

    def df(xi):
        return 2*invAfunc(Afunc(xi)-b)+rho*(xi-x0)
    dx = -df(x0) # first step is in the steepest gradient
    #alpha linear search argmin_alpha f(x0 + alpha*dx)
    alpha,nstp = BacktrackingLineSearch(f, df, x0, dx, ls_Nite = ls_Nite)
    x = x0 + alpha * dx
    s = dx
    delta0 = np.linalg.norm(dx)
    deltanew = delta0
    # iteration
    while i < Nite and deltanew > eps*delta0 and nstp < ls_Nite:
        dx = -df(x)#-2*invAfunc(Afunc(x)-b)-rho*(x-x0) #this just -df(x)
        #Fletcher-Reeves: beta = np.linalg.norm(dx)/np.linalg.norm(dx_old)
        deltaold = deltanew
        deltanew = np.linalg.norm(dx)
        beta = float(deltanew / float(deltaold))
        s = dx + beta * s
        #alpha linear search argmin_alpha f(x + alpha*s)
        alpha,nstp = BacktrackingLineSearch(f, df, x, s, ls_Nite = ls_Nite)
        x = x + alpha * s
        i = i + 1
        #print nstp
    return x

class TV2d_r:
    "this define functions related to totalvariation minimization"
    def __init__( self ):
        self.ndim    = 2         #number of image dimension

    def grad( self, x ): #gradient of x
        sx = x.shape[0]
        sy = x.shape[1]    
        Dx = x[np.r_[1:sx, sx-1],:] - x
        self.rx = x[sx-1,:]
        Dy = x[:,np.r_[1:sy, sy-1]] - x
        self.ry = x[:,sy-1]
        #res = np.zeros((sx,sy,2), dtype=x.dtype)
        res = np.zeros(x.shape + (self.ndim,), dtype = x.dtype)
        res[...,0] = Dx
        res[...,1] = Dy
        return res

    def adjgradx( self, x ): #adj for gradient
        sx   = x.shape[0]
        x[sx-1,:] = self.rx
        x = np.flip(np.cumsum(np.flip(x,0), 0),0)
        #x[0:sx-1,:] = cumx[1:sx,:]
        return x

    def adjgrady( self, x ): #adj for gradient
        sy = x.shape[1]
        x[:,sy-1] = self.ry
        x = np.flip(np.cumsum(np.flip(x,1), 1),1)
        #x[:,0:sy-1] = cumx[:,1:sy]
        return x

    def adjgrad( self, y ): #adj for gradient
        res = self.adjgradx(y[...,0]) + self.adjgrady(y[...,1])
        return res

    def adjDy( self, x ): #used in computing divergense of x
        sx = x.shape[0]
        sy = x.shape[1]
        res = x[:,np.r_[0, 0:sy-1]] - x
        res[:,0] = -x[:,0]
        res[:,-1] = x[:,-2]
        return res

    def adjDx( self, x ): #used in computing divergense of x
        sx = x.shape[0]
        sy = x.shape[1]
        res = x[np.r_[0, 0:sx-1],:] - x
        res[0,:] = -x[0,:]
        res[-1,:] = x[-2,:]
        return res

    def Div( self, y ):  #divergense of x
        #res = np.zeros(y.shape)
        res = self.adjDx(y[...,0]) + self.adjDy(y[...,1])
        return res

    def amp( self, grad ):
        amp = np.sqrt(np.sum(grad ** 2, axis=(len(grad.shape)-1)))#nomalize u along the third dimension
        amp_shape = amp.shape + (1,)
        #amp_vec   = tuple(np.ones(len(amp.shape))) + (self.ndim,)
        #d = np.tile(amp.reshape(amp_shape), amp_vec)#.reshape(sizeg)
        d = np.ones(amp.shape + (self.ndim,), dtype = amp.dtype)
        d = np.multiply(amp.reshape(amp_shape), d)
        return d
    # image --> sparse domain
    def backward( self, x ):
        return self.grad(x)
    # sparse domain --> image
    def forward( self, y ):
        return self.Div(y)    #self.adjgrad(y)#
    

#for 2d tv on muti-dimension  (nd > 2) input data
def prox_tv2d_r( y, lambda_tv, step = 0.1 ):
    #lambda_tv = 2/rho
    #nx, ny, nz = y.shape
    sizeg = y.shape+(2,) #size of gradient tensor
    G = np.zeros(sizeg)#intial gradient tensor
    i = 0
    tvopt = TV2d_r()
    #amp = lambda u : np.sqrt(np.sum(u ** 2,axis=3))#nomalize u along the third dimension
    #norm_g0 = np.linalg.norm(tvopt.grad(y))
    #norm_g = norm_g0
    while i < 40:
        dG = tvopt.grad(tvopt.Div(G)-y/lambda_tv)#gradient of G
        G = G - step*dG#gradient desent, tested to work with negative sign for gradient update
        d = tvopt.amp(G)#np.tile(amp(G)[:,:,np.newaxis], (1,1,1,2))#.reshape(sizeg)
        G = G/np.maximum(d,1.0*np.ones(sizeg))#normalize to ensure the |G|<1
        i = i + 1
        #lambda_tv = lambda_tv*ntheta/np.linalg.norm(f-y)
        #norm_g = np.linalg.norm(G)
    f = y - lambda_tv * tvopt.Div(G)
    return f

def ADMM_l2Afxnb_tvx( Afunc, invAfunc, b, Nite, step, tv_r, rho, cgd_Nite = 3, tvndim = 2 ):
    z = invAfunc(b) #np.zeros(x.shape), z=AH(b)
    u = np.zeros(z.shape)
    # 2d or 3d, use different proximal funcitons
    if tvndim is 2:
        tvprox = prox_tv2d_r
    # elif tvndim is 3:
        # tvprox = prox_tv3d
    else:
        print('dimension imcompatiable in ADMM_l2Afxnb_tvx')
        return None
    # iteration
    for _ in range(Nite):
        # soft threshold
        #x = pf.prox_l2_Afxnb_GD(Afunc,invAfunc,b,z-u,rho,20,0.1)
        x = prox_l2_Afxnb_CGD( Afunc, invAfunc, b, z-u, rho, cgd_Nite )
        z = tvprox(x + u, 2.0 * tv_r/rho)#pf.prox_tv2d(x+u,2*tv_r/rho)
        u = u + step * (x - z)

        print( 'gradient in ADMM %g' % np.linalg.norm(x-z))
    return x

def saveim1(im, file_name):
    fig, ax = plt.subplots()
    ax.imshow(im, cmap='gray')
    ax.axis('off')
    # plt.show()
    plt.savefig(fname='./results/'+file_name, bbox_inches='tight')

def saveim2(im):
    fig, ax = plt.subplots()
    ax.imshow(im, cmap=cm.gray)
    ax.axis('off')
    plt.show()

def generate_variable_density_cartesian_mask(height, width, central_fraction=0.2, peripheral_fraction=0.05):
    """
    Generate a variable density Cartesian mask for k-space MRI, with more sampling in the center
    and less in the outer regions.

    Parameters:
    height (int): The height of the mask.
    width (int): The width of the mask.
    central_fraction (float): Fraction of lines to retain in the center of k-space.
    peripheral_fraction (float): Fraction of lines to retain in the peripheral of k-space.

    Returns:
    numpy.ndarray: The generated Cartesian mask.
    """
    mask = np.zeros((height, width))

    # Central region
    central_lines = int(np.ceil(central_fraction * height))
    start_idx = height // 2 - central_lines // 2
    end_idx = start_idx + central_lines
    mask[start_idx:end_idx, :] = 1

    # Peripheral regions
    peripheral_lines_each = int(np.ceil(peripheral_fraction * (height - central_lines) / 2))
    
    # Top peripheral region
    top_indices = np.round(np.linspace(0, start_idx - 1, peripheral_lines_each)).astype(int)
    mask[top_indices, :] = 1

    # Bottom peripheral region
    bottom_indices = np.round(np.linspace(end_idx, height - 1, peripheral_lines_each)).astype(int)
    mask[bottom_indices, :] = 1

    return mask


def test():
    # simulated image
    path = "/home/wzw/wzw/pnpadmm/mripy-master/data/sim_2dmri.mat"
    mat_contents = sio.loadmat(path)
    im = mat_contents["sim_2dmri"]
    #plotim2(im)

    saveim1(im, file_name='org_mri')

    nx,ny = im.shape
    h, w = nx, ny
    mask = generate_variable_density_cartesian_mask(h, w)

    saveim1(mask, file_name='mask')

    # define A and invA fuctions, i.e. A(x) = b, invA(b) = x
    def Afunc(image):
        ksp = np.fft.fft2(image)
        ksp = np.fft.fftshift(ksp,(0,1))
        return np.multiply(ksp,mask)

    def invAfunc(ksp):
        ksp = np.fft.ifftshift(ksp,(0,1))
        im = np.fft.ifft2(ksp)
        return im

    saveim1(np.absolute(mask), file_name='abs_mask')


    b = Afunc(im)
    saveim1(np.absolute(b), file_name='abs_b')
    saveim1(np.absolute(invAfunc(b)), file_name='abs_invAfunc(b)')

    #do soft thresholding
    #do cs mri recon
    Nite = 100 #number of iterations
    step = 1 #step size
    #th = 1000 # theshold level
    #xopt = solvers.IST_2(Afunc,invAfunc,b, Nite, step,th) #soft thresholding
    xopt = ADMM_l2Afxnb_tvx( Afunc, invAfunc, b, Nite, step, 10, 1 )
    #xopt = solvers.ADMM_l2Afxnb_l1x_2( Afunc, invAfunc, b, Nite, step, 100, 1 )

    saveim1(np.absolute(xopt), file_name='abs_xopt')


def excute_split_bregman(full_img, mask):
    # https://github.com/guanhuaw/MIRTorch/blob/master/examples/demo_cs.ipynb
    # simulated image
    # path = "/home/wzw/wzw/pnpadmm/mripy-master/data/sim_2dmri.mat"
    # mat_contents = sio.loadmat(path)
    # im = mat_contents["sim_2dmri"]
    #plotim2(im)

    # saveim1(im, file_name='org_mri')

    im = full_img

    nx,ny = im.shape
    h, w = nx, ny
    # mask = generate_variable_density_cartesian_mask(h, w)

    # saveim1(mask, file_name='mask')

    # define A and invA fuctions, i.e. A(x) = b, invA(b) = x
    def Afunc(image):
        ksp = np.fft.fft2(image)
        ksp = np.fft.fftshift(ksp,(0,1))
        return np.multiply(ksp,mask)

    def invAfunc(ksp):
        ksp = np.fft.ifftshift(ksp,(0,1))
        im = np.fft.ifft2(ksp)
        return im

    saveim1(np.absolute(mask), file_name='abs_mask')



    b = Afunc(im)
    # saveim1(np.absolute(b), file_name='abs_b')
    # saveim1(np.absolute(invAfunc(b)), file_name='abs_invAfunc(b)')

    #do soft thresholding
    #do cs mri recon
    Nite = 60 #number of iterations
    step = 1 #step size
    #th = 1000 # theshold level
    #xopt = solvers.IST_2(Afunc,invAfunc,b, Nite, step,th) #soft thresholding
    xopt = ADMM_l2Afxnb_tvx( Afunc, invAfunc, b, Nite, step, 10, 1 )
    #xopt = solvers.ADMM_l2Afxnb_l1x_2( Afunc, invAfunc, b, Nite, step, 100, 1 )

    # saveim1(np.absolute(xopt), file_name='abs_xopt')

    return np.absolute(xopt)


if __name__ == "__main__":
    test()