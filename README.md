# VQ-VAE

This is a VQ-VAE, but includes an experiment with adapting a state of the art classifier (Inception ResNet v2) into an encoder.  This is achieved by cutting the fully connected layers and adding what I've deemed a Dense Layer Channel Reduction (DLCR).

The goal of an encoder in a VQ-VAE is to train a model that is also reduces the image into smaller 2d encoded space that still retains spacial information.  The goal behind the DLCR is to effectively reduce a large collection of filters into a one channel or a small number of channels.  Using a singls 1x1 convolution only allows for a single linear calculation of each individual filter which essentially allows for applying a weight to each filter and then sums them together.

Ideally it would be much better to add multiple layers to be able to create complex function applied to the collection of filters.  The goal of the DLCR is to create a dense network that also gradually reduces channels.  See the diagram below ...

![DLCR](https://github.com/wrrogers/VQ-VAE_v1/blob/master/dlcr.png)

# Results

## Comparing VQ-VAE with 1x1 conv layer and DLCR
Inception ResNet v2 /w 1 conv:/tRecon Error: 0.00661658, Loss: 0.007578598, Perplexity: 12.36619
Inception ResNet v2 /w DLCR:/tRecon Error: 0.00657987, Loss: 0.008241967, Perplexity: 4.484635

## Comparing Basic Residual Encoder and DLCR
Residual Encoder:   Recon Error: 0.00189220, Loss: 0.0020413, Perplexity: 12.991144
DLCR:               Recon Error: 0.00214382, Loss 0.00232471, Perplexity: 12.10787

