---
title: >-
  [Paper Note] Uncertainty-Driven Spectral Compressive Imaging with Spatial-Frequency Transformer
description: >-
  [ECCV 2024][Model Compression][Hyperspectral Image Reconstruction] This paper proposes Specformer, which fully captures the spatial sparsity and inter-spectral similarity priors of hyperspectral images (HSIs) through parallel local window self-attention (LWSA) and frequency-domain self-attention (FWSA) modules. It also introduces an uncertainty-driven loss function to enhance the network's reconstruction capability for texture-rich and boundary regions…
tags:
  - "ECCV 2024"
  - "Model Compression"
  - "Hyperspectral Image Reconstruction"
  - "Coded Aperture Snapshot Spectral Imaging"
  - "Spatial-Frequency Transformer"
  - "Uncertainty-Driven"
  - "Self-Attention"
date: 2026-05-08
content_hash: e6564c52fa863253
---

# Uncertainty-Driven Spectral Compressive Imaging with Spatial-Frequency Transformer

**Conference**: ECCV 2024  
**Code**: [https://github.com/bianlab/Specformer](https://github.com/bianlab/Specformer)  
**Area**: Model Compression / Spectral Imaging  
**Keywords**: Hyperspectral Image Reconstruction, Coded Aperture Snapshot Spectral Imaging, Spatial-Frequency Transformer, Uncertainty-Driven, Self-Attention

## TL;DR
This paper proposes Specformer, which fully captures the spatial sparsity and inter-spectral similarity priors of hyperspectral images (HSIs) through parallel local window self-attention (LWSA) and frequency-domain self-attention (FWSA) modules. It also introduces an uncertainty-driven loss function to enhance the network's reconstruction capability for texture-rich and boundary regions, outperforming the state-of-the-art (SOTA) with lower computational cost on both simulated and real-world HSI datasets.

## Background & Motivation

**Background**: Coded Aperture Snapshot Spectral Imaging (CASSI) systems capture 2D compressed measurements in a single snapshot, which are then reconstructed into 3D hyperspectral data cubes using algorithms. In recent years, learning-based methods (such as Transformer-based methods like TSA-Net, MST, and CST) have made remarkable progress in reconstruction quality, gradually replacing traditional iterative optimization methods.

**Limitations of Prior Work**: Existing learning-based methods suffer from two core problems. First, they rarely exploit both the spatial sparsity prior and the inter-spectral similarity prior of HSIs simultaneously. Spatial methods (e.g., window attention) excel at capturing local spatial features but neglect cross-spectral correlations, whereas spectral methods do the opposite. Second, existing methods treat all image regions equally, ignoring the fact that texture-rich and boundary regions are much harder to reconstruct than smooth regions, which limits the overall reconstruction quality due to these difficult regions.

**Key Challenge**: Jointly modeling spatial and frequency (spectral) information requires heavy computation. Traditional approaches serially stack different types of attention modules, leading to computational redundancy. Meanwhile, uniform loss weight distribution biases the network toward optimizing easy-to-reconstruct smooth regions rather than the difficult regions that genuinely require attention.

**Goal**: (1) How to efficiently and simultaneously model both spatial sparsity and inter-spectral similarity? (2) How to enable the network to adaptively focus on regions that are difficult to reconstruct?

**Key Insight**: The authors propose parallelizing rather than serializing spatial and frequency attention. Through this parallel design, cross-window connections are established naturally, expanding the receptive field while maintaining linear complexity. For region-adaptive optimization, the authors draw inspiration from uncertainty estimation in Bayesian deep learning, enabling the network to automatically learn the reconstruction difficulty of each pixel and use it as a loss weight.

**Core Idea**: Jointly model the two HSI priors using parallel spatial-frequency attention modules, and automatically enhance learning in difficult regions using a pixel-level uncertainty-driven loss function.

## Method

### Overall Architecture
Specformer adopts a multi-scale U-shaped network architecture. The input consists of the 2D compressed measurements from the CASSI system alongside the corresponding spectral coding mask, and the output is the reconstructed 3D hyperspectral data cube. The main network body is constructed by stacking Spatial-Frequency (SF) Blocks, each of which contains parallel LWSA and FWSA attention branches. The network additionally outputs an uncertainty map to guide the weight distribution of the loss function.

### Key Designs

1. **Local Window Self-Attention (LWSA)**:

    - **Function**: Captures spatial local features within each spectral band, focusing on regions with dense spectral information.
    - **Mechanism**: Partitions the feature map of each spectral band into non-overlapping local windows and performs standard multi-head self-attention within these windows. The window size is fixed to $M \times M$, yielding a computational complexity of $O(M^2 \cdot H W)$, which is linear relative to the $O((HW)^2)$ of global attention. LWSA guides the network to focus on spatial regions with dense spectral information, as these regions yield higher attention scores.
    - **Design Motivation**: Spatial dimensions in HSIs possess sparsity, where spatially adjacent pixels typically exhibit similar spectral properties. Local window attention naturally aligns with this prior and is computationally efficient. However, using LWSA in isolation isolates information across different windows.

2. **Frequency-Domain Self-Attention (FWSA)**:

    - **Function**: Captures similarity relationships across spectra while establishing long-range dependencies across windows.
    - **Mechanism**: FWSA first transforms spatial features into the frequency domain using the Fast Fourier Transform (FFT) and then performs self-attention in the frequency domain. A key advantage of the frequency domain is that each frequency component inherently encodes global spatial information (one point in the frequency domain corresponds to a global pattern in the spatial domain), allowing frequency-domain attention to naturally possess a global receptive field. Specifically, a 1D FFT is performed on features along the spectral dimension to obtain frequency feature maps, the attention scores across frequencies are computed, and the features are transformed back to the spatial domain via the Inverse FFT (IFFT).
    - **Design Motivation**: Strong correlations exist across different spectral bands of HSIs (spectral responses of neighboring bands are highly similar). In the frequency domain, this cross-spectral similarity can be represented more compactly. Concurrently, as a complement to LWSA, frequency-domain attention naturally resolves the window isolation problem of LWSA.

3. **Uncertainty-Driven Loss**:

    - **Function**: Enables the network to automatically identify and reinforce learning on difficult regions, such as texture-rich and boundary regions.
    - **Mechanism**: The network additionally outputs an uncertainty map $\sigma(x)$ of the same size as the HSI, representing the reconstruction confidence of each pixel. The loss function is formulated as $\mathcal{L} = \frac{1}{2\sigma^2(x)} \|y - \hat{y}\|^2 + \frac{1}{2}\log\sigma^2(x)$. The first term reduces the weight of the reconstruction error in regions with high uncertainty (large $\sigma$), while the regularization of the second term prevents $\sigma$ from increasing infinitely. During training, the network automatically learns smaller $\sigma$ values (larger loss weights) for texture and boundary regions, thereby forcing the network to pay more attention to these regions.
    - **Design Motivation**: Traditional MSE loss treats all pixels with equal weight. In HSI reconstruction, smooth regions make up the majority and are easy to reconstruct, biasing network training towards optimizing these "easy" regions. Drawing inspiration from heteroscedastic uncertainty modeling (Kendall & Gal, 2017), the uncertainty-driven loss learns adaptive loss weights for each pixel.

### Loss & Training
The total loss is the uncertainty-weighted L2 loss plus the uncertainty regularization term. During training, the Adam optimizer is used with a cosine annealing learning rate scheduler. The model is pre-trained on simulated datasets such as KAIST and CAVE, and fine-tuned on real-world CASSI data.

## Key Experimental Results

### Main Results

| Dataset | Metric | Specformer | CST (ECCV22) | MST++ (CVPRW22) | Parameter Comparison |
|--------|------|------|----------|------|------|
| Simulated KAIST | PSNR | **SOTA** | Runner-up | Lower | Fewer parameters |
| Simulated TSA | PSNR/SSIM | **SOTA** | Runner-up | Lower | Lower computational cost |
| Real CASSI | Spectral Accuracy | **SOTA** | Runner-up | Lower | Lower FLOPs |

### Ablation Study

| Configuration | PSNR | Description |
|------|---------|------|
| Full model (LWSA + FWSA + UDL) | Best | Full model |
| w/o FWSA (Spatial Only) | Decreased by ~0.5dB | Loss of inter-spectral similarity modeling |
| w/o LWSA (Frequency Only) | Decreased by ~0.3dB | Loss of spatial local features |
| Serial LWSA $\rightarrow$ FWSA | Decreased by ~0.2dB with increased FLOPs | Parallel outperforms serial |
| w/o uncertainty loss (using standard MSE) | Decreased by ~0.4dB | Degraded reconstruction in texture/boundary regions |
| w/o multi-scale U-shape | Decreased by ~0.8dB | Multi-scale features are crucial |

### Key Findings
- The parallel combination of FWSA and LWSA performs better and requires less computation than serial stacking, as the parallel design allows full interaction between the two types of information at every layer.
- The uncertainty-driven loss yields the most significant PSNR improvement (~0.8dB) in texture-rich regions, validating the efficacy of the adaptive weighting mechanism.
- Visualizing the uncertainty map reveals that the network indeed learns to assign higher loss weights (lower uncertainty $\sigma$) to boundary and texture regions, which aligns with intuition.
- While maintaining SOTA accuracy, Specformer achieves lower parameter counts and FLOPs compared to competing methods such as CST and DAUHST.

## Highlights & Insights
- The design of **parallel spatial-frequency attention** is elegant—using frequency-domain attention instead of shifted windows or cross-window attention to establish global connections maintains linear complexity while naturally matching the HSI spectral similarity prior. This design can be extended to process any multi-channel signals (such as multi-frame videos and multi-modal fusion).
- The **uncertainty-driven loss** is a versatile trick that essentially allows the network to learn pixel-level loss weights. This idea can be directly transferred to other image restoration tasks such as super-resolution and deblurring—any problem with "difficult regions" can benefit from it.
- The insight that FFT naturally provides a global receptive field in frequency-domain self-attention is noteworthy, avoiding the information loss incurred by transferring to the frequency domain and then pooling.

## Limitations & Future Work
- Although the repository is public, the actual code has not been released yet (the GitHub page shows "code is coming soon"), raising reproducibility concerns.
- The learning of the uncertainty map lacks explicit supervision signals, relying entirely on implicit learning via the loss function, which may be unstable.
- Validation is limited to CASSI systems; the applicability to other spectral imaging systems (e.g., CTIS, SCI) has not been verified.
- Frequency-domain attention is sensitive to the input resolution—differing resolutions alter the physical meaning of FFT frequency components, requiring re-adaptation.
- No comparison has been made with recent diffusion-model-based HSI reconstruction methods.

## Related Work & Insights
- **vs MST**: MST uses spectral-wise multi-head self-attention, whereas Specformer incorporates a frequency-domain branch on top of it to leverage inter-spectral information more comprehensively.
- **vs CST**: CST utilizes a cross-scale Transformer and focuses on multi-scale feature fusion, whereas Specformer focuses on joint spatial-frequency modeling, presenting different entry angles.
- **vs DAUHST**: DAUHST is a hybrid framework of deep unfolding and Transformer, which suffers from heavy computation; Specformer is much more lightweight.
- The concept of uncertainty loss originates from the heteroscedastic uncertainty of Kendall & Gal, but this work represents its first application to spectral imaging reconstruction.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of parallel spatial-frequency attention and uncertainty-driven loss is novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ Validated on both simulated and real datasets with comprehensive ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Clear motivational derivation, and self-consistent logical block designs.
- Value: ⭐⭐⭐⭐ Advances the SOTA in the field of HSI reconstruction, with design ideas showing high transferability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] A Simple Low-bit Quantization Framework for Video Snapshot Compressive Imaging](a_simple_lowbit_quantization_framework_for_video_snapshot_co.md)
- [\[CVPR 2026\] When Lines Meet Textures: Spatial-Frequency Aligned Diffusion Features for Cross-Sparsity Correspondence](../../CVPR2026/model_compression/when_lines_meet_textures_spatial-frequency_aligned_diffusion_features_for_cross-.md)
- [\[ECCV 2024\] Token Compensator: Altering Inference Cost of Vision Transformer without Re-Tuning](token_compensator_altering_inference_cost_of_vision_transformer_without_re-tunin.md)
- [\[CVPR 2025\] Sampling Innovation-Based Adaptive Compressive Sensing](../../CVPR2025/model_compression/sampling_innovation-based_adaptive_compressive_sensing.md)
- [\[ICLR 2026\] Understanding Dataset Distillation via Spectral Filtering](../../ICLR2026/model_compression/understanding_dataset_distillation_via_spectral_filtering.md)

</div>

<!-- RELATED:END -->
