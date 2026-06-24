---
title: >-
  [Paper Note] End-to-End Implicit Neural Representations for Classification
description: >-
  [CVPR 2025][3D Vision][Implicit Neural Representations] Proposes the Meta Weight Transformer (MWT), which utilizes end-to-end meta-learning of SIREN initialization parameters and learning rate schedules. This allows the weight structure of INR to simultaneously optimize reconstruction quality and classification performance. Using a simple standard Transformer for classification on SIREN weights outperforms all equivariant architecture methods…
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Implicit Neural Representations"
  - "SIREN classification"
  - "meta-learning"
  - "weight space"
  - "Transformer"
date: 2026-05-08
content_hash: e06c8eab6db6535c
---

# End-to-End Implicit Neural Representations for Classification

**Conference**: CVPR 2025  
**arXiv**: [2503.18123](https://arxiv.org/abs/2503.18123)  
**Code**: [https://github.com/SanderGielisse/MWT](https://github.com/SanderGielisse/MWT)  
**Area**: 3D Vision  
**Keywords**: Implicit Neural Representations, SIREN classification, meta-learning, weight space, Transformer

## TL;DR
Proposes the Meta Weight Transformer (MWT), which utilizes end-to-end meta-learning of SIREN initialization parameters and learning rate schedules. This allows the weight structure of INR to simultaneously optimize reconstruction quality and classification performance. Using a simple standard Transformer for classification on SIREN weights outperforms all equivariant architecture methods, achieving INR classification on high-resolution ImageNet-1K for the first time.

## Background & Motivation

**Background**: Implicit Neural Representations (INRs) like SIREN encode images into MLP weight parameters $\theta$, achieving outstanding performance in signal reconstruction. However, for downstream tasks like classification, one must design a classifier $g(\theta)$ in the weight space $\theta$. Directly classifying these weights is difficult due to permutation and scaling symmetries (different permutations/scalings of weights can correspond to the same function).

**Limitations of Prior Work**: Current mainstream methods (e.g., DWS-Net, NFN, ScaleGMN) focus on designing architectures that are equivariant to weight symmetries. However, even with complex equivariant designs, INR classification performance remains far below pixel-based CNN methods. Crucially, these methods adopt a two-stage pipeline: first fitting an INR for each image individually (without considering classification feedback), and then training a classifier on the INR weights—meaning the classifier cannot influence the weight structure of the INR.

**Key Challenge**: The weights of INRs lack sufficient "structure," making it difficult for downstream models to identify useful image features. Studies show that sharing initialization and reducing the number of update steps can improve classification, but fewer steps degrade reconstruction quality—a clear trade-off exists between reconstruction and classification.

**Goal**: How to ensure that the weight structure of INRs retains good reconstruction quality while presenting a structure interpretable by downstream classifiers? How to allow the classification objective to conversely affect the INR fitting process?

**Key Insight**: By embedding the INR fitting process into the classifier's training loop through meta-learning, the classification loss backpropagates to influence SIREN's shared initialization and per-step learning rates, achieving end-to-end optimization. Since it only requires a small number of update steps (k=4~6), it is highly computationally efficient and scalable to high-resolution images.

**Core Idea**: Using end-to-end meta-learning to let the classification loss guide the optimization of SIREN initialization and learning rate schedules, making the fitted weight structures naturally suitable for classification without the need for designing complex equivariant architectures.

## Method

### Overall Architecture
The system comprises three learnable components: (1) shared SIREN initialization parameters $\theta$; (2) a per-step, per-parameter learning rate schedule $\alpha \in \mathbb{R}^{k \times |\theta|}$; (3) a standard Transformer classifier $h_\psi$. Training pipeline: For each training image, starting from the shared $\theta$, $k$ inner-loop update steps are performed using the MSE reconstruction loss to obtain image-specific parameters $\phi$, which are then classified by the Transformer. The outer loop backpropagates both the reconstruction loss and the classification loss to update $\theta$, $\alpha$ (meta-learning), and the classifier $\psi$. During inference, new images also undergo $k$ update steps before being classified.

### Key Designs

1. **Meta-Learned SIREN**:

    - **Function**: Learns a shared initialization and learning rate schedule, ensuring that the weights after $k$ update steps possess both good reconstruction quality and classification-friendly structures.
    - **Mechanism**: The inner loop uses the MSE reconstruction loss to perform $k$ steps of SGD updates: $\phi \leftarrow \phi - \alpha_i \nabla_\phi \mathcal{L}_{rec}$. The outer loop calculates two gradients—reconstruction gradient $g^{rec}_{\theta,\alpha}$ and classification gradient $g^{cls}_{\theta,\alpha}$—and updates $\theta$ and $\alpha$ using a weighted combination with $w_{cls}$. The inner loop does not use classification loss because labels are unavailable at test time.
    - **Design Motivation**: The shared initialization aligns the weights of all images within the same coordinate system, mitigating symmetry issues. Backpropagating the classification loss gradient allows the initialization to adapt to the classifier's needs. Balance is optimized when $w_{cls}=0.01$. Unlike the manual parameter tuning in fit-a-nef, this method utilizes meta-learning to automatically find the equilibrium.

2. **Weight Difference & Scaling for Transformer Input Processing**:

    - **Function**: Converts SIREN weights into token sequences that can be efficiently processed by the Transformer.
    - **Mechanism**: Treats each output neuron of each hidden layer as a token with a feature dimension of $c_{in}+1$ (input weights + bias). Crucial processing: Instead of directly inputting $\phi$, the weight difference $\phi_{scaled} = \lambda(\phi - \theta + \beta)$ is used, where $\lambda=500$ is a scaling factor and $\beta$ is a learnable positional bias. For a 4-layer 128-dimensional SIREN, there are $128 \times 4 = 512$ tokens in total.
    - **Design Motivation**: Since $\theta$ and $\phi$ are typically very close, their difference is minuscule and easily ignored by the Transformer due to its low-frequency bias. Taking the difference and applying a large scale makes the signal more pronounced. $\beta$ provides positional information, helping the Transformer distinguish between different neurons.

3. **Random Pixel Subsampling Strategy**:

    - **Function**: Reduces the computational cost of meta-learning on high-resolution images.
    - **Mechanism**: In each inner-loop step, instead of using all pixels to calculate shortest-path/reconstruction loss, a subset of pixels with a random sampling ratio $s$ is used. When $s=1/k$, each pixel is seen once on average. Experiments reveal that even with $s=0.05$ (5% of pixels), classification and reconstruction quality are barely affected.
    - **Design Motivation**: Meta-learning requires storing computation graphs for $k$ steps, leading to massive memory consumption for high-resolution images (e.g., 224x224 in ImageNet-1K). Subsampling drastically reduces memory demands (from 24.1 GiB to 13.5 GiB) while implying that SIREN may have learned implicit image priors capable of inferring the whole from partial pixels.

### Loss & Training
- Inner loop: MSE reconstruction loss $\mathcal{L}_{rec}$, optimized via plain SGD.
- Outer loop: $g_{\theta,\alpha} = g^{rec}_{\theta,\alpha} + w_{cls} \cdot g^{cls}_{\theta,\alpha}$, $w_{cls}=0.01$
- Classifier: Updated separately with $\mathcal{L}_{cls}$ to update $\psi$
- Optimizer: AdamW with lr=1e-4, and lr=1e-2 for the learning rate schedule $\alpha$
- Supports spatial augmentations (e.g., rotation, flipping, scaling) because it only requires refitting SIREN for a few steps

## Key Experimental Results

### Main Results

| Dataset | Metric | MWT-L | Prev. SOTA (ScaleGMN-B) | Gain |
|--------|------|-------|----------------------|------|
| MNIST | Accuracy | **98.80%** | 96.59% | +2.2% |
| Fashion-MNIST | Accuracy | **90.43%** | 80.78% | +9.7% |
| CIFAR-10 (w/o Aug) | Accuracy | **59.57%** | 38.82% | +20.7% |
| CIFAR-10 (w/ Aug) | Accuracy | **64.7%** | 63.4% (inr2array) | +1.3% |
| Imagenette | Accuracy | **60.8%** | - (First) | - |
| ImageNet-1K | Accuracy | **23.6%** | - (First) | - |

### Ablation Study

| Configuration | CIFAR-10 Accuracy | PSNR | Description |
|------|----------------|------|------|
| WT ($w_{cls}=0$) | 43.78% | High | No classification gradient feedback |
| MWT ($w_{cls}=0.01$) | 56.90% | Moderate | Classification gradient guides meta-learning |
| MWT-L (width=256) | **59.57%** | Moderate | Larger SIREN + larger Transformer |
| $w_{cls}=0.1$ (Too high) | Decreased | Decreased | Excessive classification intervention harms reconstruction |
| $k=4$ steps | Slightly decreased | Slightly decreased | Works well even with few steps |
| $k=6$ steps | **Optimal** | Better | Balances computation and performance |

### Key Findings
- Classification gradient feedback is the core contribution: MWT improves over WT by 13.1% on CIFAR-10 (56.90% vs 43.78%), demonstrating the critical importance of allowing the classification loss to affect the INR structure.
- WT (without classification feedback) already matches or surpasses most equivariant methods, indicating that shared initialization combined with few-step updates is inherently highly effective.
- Pixel subsampling is practically lossless: On Imagenette, the classification accuracy difference between $s=0.05$ and $s=0.25$ is <1%, while the training memory is halved.
- There is a clear sweet spot for $w_{cls}$: setting it too high harms both reconstruction and classification; 0.01 yields optimal results.
- Successfully demonstrates for the first time that INR classification scales to the ImageNet-1K level (23.6% top-1), though it remains far below standard pixel-based methods.

## Highlights & Insights
- **"Structure" is more important than "equivariance"**: Rather than designing complex equivariant architectures to handle weight symmetries, forcing weights to be structured directly via meta-learning is far more effective. This represents a paradigm shift from "adapting to symmetry" to "eliminating symmetry".
- **Generality of the end-to-end meta-learning framework**: The inner loop updates parameters using task A's loss, while the outer loop optimizes initialization using task B's loss—this framework is applicable to any scenario involving "fitting representations first, then performing downstream tasks".
- **Computational efficiency enables high resolution**: Few-step updates (k=4~6) paired with pixel subsampling make training on the scale of ImageNet feasible, presenting the first study to conduct INR classification on high-resolution datasets.

## Limitations & Future Work
- A significant gap still remains between INR classification and pixel-based classification (23.6% vs CNN's 76%+ on ImageNet), which limits its practical utility.
- Meta-learning requires storing computation graphs for $k$ steps for second-order gradient calculations, resulting in high training memory overhead.
- Only SIREN was verified as the INR architecture, leaving other choices (e.g., hash-based or hybrid INRs) unexplored.
- The classifier utilizes a standard Transformer without specialized optimizations, leaving room for potential improvements.
- The transferability of INR representations to other downstream tasks (e.g., detection, segmentation) remains unexplored.

## Related Work & Insights
- **vs ScaleGMN**: ScaleGMN designs graph networks that are equivariant to scale and permutation symmetries simultaneously. MWT bypasses equivariance design entirely but significantly outperforms it, challenging the paradigm that "equivariant architectures are necessary".
- **vs fit-a-nef**: fit-a-nef discovered that shared initialization and few-step updates aid classification, though it relied on manual parameter tuning; MWT takes this further by automatically optimizing via meta-learning and introducing classification feedback, leading to substantial performance gains.
- **vs inr2array (NFT)**: inr2array reaches 63.4% on CIFAR-10 with augmentations, whereas MWT-L achieves 64.7%. However, MWT incurs a lower computational cost for augmentations (requiring only a few refitting steps).

## Rating
- Novelty: ⭐⭐⭐⭐⭐ End-to-end meta-learning where classification guides the INR representation structure, presenting a paradigm innovation.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive coverage from MNIST to ImageNet-1K, extensive ablations, establishing the first high-resolution baseline.
- Writing Quality: ⭐⭐⭐⭐ Clearly described methodology and thorough ablation analyses.
- Value: ⭐⭐⭐⭐ Provides a significant push to the field of INR classification, although a major gap to pixel-based methods still remains.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] End-to-End HOI Reconstruction Transformer with Graph-based Encoding](end-to-end_hoi_reconstruction_transformer_with_graph-based_encoding.md)
- [\[CVPR 2025\] Rethinking End-to-End 2D to 3D Scene Segmentation in Gaussian Splatting](rethinking_end-to-end_2d_to_3d_scene_segmentation_in_gaussian_splatting.md)
- [\[CVPR 2025\] SiNR: Sparsity Driven Compressed Implicit Neural Representations](sinr_sparsity_driven_compressed_implicit_neural_representations.md)
- [\[ICLR 2026\] Mesh Splatting for End-to-end Multiview Surface Reconstruction](../../ICLR2026/3d_vision/mesh_splatting_for_end-to-end_multiview_surface_reconstruction.md)
- [\[ICML 2026\] EPS3D: End-to-End Feed-Forward 3D Panoptic Segmentation](../../ICML2026/3d_vision/eps3d_end-to-end_feed-forward_3d_panoptic_segmentation.md)

</div>

<!-- RELATED:END -->
