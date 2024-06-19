# Split-Bregman-ST-Total-Variation-MRI
Split Bregman spatiotemporal total variation for cardiac cine MRI

This repository contains a demo that shows how to use Spatiotemporal Total Variaton, which is efficiently implemented with the Split Bregman formulation, for cardiac cine MRI, as used in the paper: 
**P Montesinos, J F P J Abascal, L Cussó, J J Vaquero, M Desco. Application of the compressed sensing technique to self-gated cardiac cine sequences in small animals. Magn Reson Med., 72(2): 369–380, 2013.** 
DOI: http://dx.doi.org/10.1002/mrm.24936

SpatioTemporalTVSB.m minimizes: 

![ST-TV](https://github.com/HGGM-LIM/Split-Bregman-ST-Total-Variation-MRI/blob/master/STTVFormula.jpg)

where u is the unknown image, F is the undersampled Fourier transform, f is the undersampled data, and beta_xy and beta_t are spatial and temporal weighting sparsity parameters. This implementation allows selecting different weighting sparsity parameter for time and space.

The Split Bregman method separates L2- and L1-norm functionals in such a way that they can be solved analytically in two alternating steps. In the first step a linear system is efficiently solved in the Fourier domain, which can be done in MRI and image denoising problems where operators have representation in the Fourier domain. The computational cost is three FFT per iteration. 

The demo uses cardiac cine small-animal data to simulate an undersampling pattern based on a variable density pdf and compare Spatial TV with Spatiotemporal TV. 

![Cardiac cine data set](https://github.com/HGGM-LIM/Split-Bregman-ST-Total-Variation-MRI/blob/master/dataCine8fr.gif)

The repository contains the following files:

- **dataCine8fr.mat:** Absolute image of retrospective cardiac cine small-animal data (8 frames, healthy rat)
(Acquired data can be found at http://biig.uc3m.es/cardiac-cine-data)

- **dataCine8fr_float_192x192x8:** Raw data of cardiac cine data set, 192x192x8, float 

- **dataCine8fr.gif:** Video of images of cardiac cine data set 

- **Demo_SpatioTemporalTV_SplitBregman_Sim.m:** This demo loads image, simulate an undersampling pattern for a given acceleration and reconstruct data using spatial total variation (S-TV) and spatiotemporal TV (ST-TV)

- **SpatialTVSB.m:** S-TV solved using the Split Bregman formulation

- **SpatioTemporalTVSB.m:** ST-TV solved using the Split Bregman formulation

- **genPDF.m:** Function to generate a pdf with polynomial variable density sampling, by Michael Lustig 2007, downloaded from (SparseMRI V0.2), http://web.stanford.edu/~mlustig/SparseMRI.html, see M. Lustig, D.L

- **genSampling_LIM.m:** Monte-carlo algorithm to generate a sampling pattern. Modified from the original function by Michael Lustig 2007

- **maxSidePeakRatio.m:** Computes the maximum sidelobe to peak ratio of point spread function for undersampled data. Used within genSampling_LIM.m


If you need to contact the author, please do so at paumsdv@gmail.com, juanabascal78@gmail.com, desco@hggm.es
