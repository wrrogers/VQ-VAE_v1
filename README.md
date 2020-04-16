# VQ-VAE

This is a VQ-VAE, but includes an experiment with adapting a large classifier into an encoder.  This is achieved by cutting the fully connected layers and adding what I've deemed a Dense Layer Channel Reduction (DLCR).



The goal of an encoder in a VQ-VAE is to train a model that is also reduces the image into smaller 2d encoded space that still retains spacial information.  The goal behind the DLCR is to effectively reduce a large collection of filters into a one channel or a small number of channels.  Using a singls 1x1 convolution only allows for a single linear calculation of each individual filter which essentially allows for applying a weight to each filter and then sums them together.

Ideally it would be much better to add multiple layers to be able to create complex function in applying filters.  See the diagram below ...

![DLCR](https://github.com/wrrogers/VQ-VAE_v1/blob/master/dlcr.png)
