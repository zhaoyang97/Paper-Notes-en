---
title: >-
  [Paper Note] SPAI: Any-Resolution AI-Generated Image Detection by Spectral Learning
description: >-
  [CVPR 2025][Image Generation][Spectral Learning] This work proposes SPAI, which models the frequency distribution of real images through Masked Spectral Learning. By introducing Spectral Reconstruction Similarity (SRS) and Spectral Context Attention (SCA), it detects AI-generated images as out-of-distribution (OOD) samples. SPAI achieves an average AUC of 91.0% across 13 generation models, an absolute improvement of 5.5% over the second-best method…
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Spectral Learning"
  - "Self-Supervised"
  - "Frequency Reconstruction"
  - "OOD Detection"
  - "Any-Resolution"
date: 2026-05-08
content_hash: cd2875b59e2ec87d
---

# SPAI: Any-Resolution AI-Generated Image Detection by Spectral Learning

**Conference**: CVPR 2025  
**arXiv**: [2411.19417](https://arxiv.org/abs/2411.19417)  
**Code**: [https://mever-team.github.io/spai](https://mever-team.github.io/spai)  
**Area**: Image Generation  
**Keywords**: Spectral Learning, Self-Supervised, Frequency Reconstruction, OOD Detection, Any-Resolution

## TL;DR
This work proposes SPAI, which models the frequency distribution of real images through Masked Spectral Learning. By introducing Spectral Reconstruction Similarity (SRS) and Spectral Context Attention (SCA), it detects AI-generated images as out-of-distribution (OOD) samples. SPAI achieves an average AUC of 91.0% across 13 generation models, an absolute improvement of 5.5% over the second-best method, while supporting detection of images with arbitrary resolutions.

## Background & Motivation

**Background**: AI image generation technologies are rapidly evolving, from GANs to diffusion models (Stable Diffusion, DALL-E 3, Midjourney, etc.), producing highly realistic images while the number of generation models proliferates. As malicious fake images spread rapidly across the internet, there is an urgent need for robust AI-generated image detection (AID) methods.

**Limitations of Prior Work**: Existing detectors operate by learning and capturing specific artifacts introduced by particular generation models (such as anomalous spectral patterns, texture inconsistencies, etc.). However, artifacts introduced by different generation models vary dramatically — even models with small differences can generate completely distinct artifacts. Consequently, existing detectors perform exceptionally well on models seen during training but fail catastrophically on unseen models. For instance, NPR yields an AUC of only 38.0% on Firefly, while DMID achieves only 67.9% on SD3.

**Key Challenge**: Learning generation model artifacts is essentially chasing a "moving target" — every time a new model emerges, the detector requires retraining. However, maintaining a training set that covers all potential models is unfeasible.

**Goal**: Instead of relying on specific model artifacts, the goal of this work is to model the invariant characteristics of **real images**, thereby detecting AI-generated images as out-of-distribution (OOD) samples.

**Key Insight**: The authors point out that the **spectral distribution** of real images constitutes a pattern that is highly invariant to generation models yet highly discriminative — it is unaffected by specific generation models but effectively distinguishes real images from synthetic ones.

**Core Idea**: Use self-supervised frequency reconstruction as a pre-training task to learn the spectral model of real images, and then compare the reconstruction accuracy to detect OOD samples.

## Method

### Overall Architecture
The input image is first split into $K$ patches of size $224 \times 224$. Each patch is processed through a frequency mask to generate low-frequency and high-frequency components, which are then fed into a pre-trained spectral ViT model $\mathcal{G}$ to extract multi-layer features. The Spectral Reconstruction Similarity (SRS) is computed, combined with the Spectral Context Vector (SCV), and fused across all patches via Spectral Context Attention (SCA). Finally, an MLP outputs the probability of being real or fake.

### Key Designs

1. **Masked Spectral Learning**:

    - Function: Self-supervised learning of the frequency distribution of real images
    - Mechanism: Use 2D DFT to decompose the image into low-frequency $x^l$ and high-frequency $x^h$ components (via a circular frequency mask of radius $r$), randomly select one as input, and train the ViT model $\mathcal{G}$ to reconstruct the full spectrum. The loss function is the frequency domain distance $\mathcal{L}_{rec} = \mathcal{D}(\mathcal{F}(x), \mathcal{F}(\hat{x}))$. It is trained using only 1.2 million real images from ImageNet, and its weights are frozen in subsequent training phases.
    - Design Motivation: The frequency reconstruction model is more accurate in reconstructing real images, while generating larger reconstruction errors for AI-generated images (OOD data), which serves as the foundation for detection.

2. **Spectral Reconstruction Similarity (SRS)**:

    - Function: Quantify the alignment between the image's spectrum and the learned spectral model of real images.
    - Mechanism: The original image, low-frequency component, and high-frequency component are encoded as feature representations for each of the $N$ transformer blocks. After mapping them to a unified space via a projection operator, three pairs of cosine similarities (original-low $\omega^{ol}$, original-high $\omega^{oh}$, low-high $\omega^{lh}$) are computed. The mean and standard deviation are extracted for each pair, and these $6N$ values from $N$ layers are concatenated to form the SRS vector $z^\lambda$.
    - Design Motivation: The feature representations for the three components of real images are highly aligned (high similarity), whereas AI-generated images exhibit lower alignment due to spectral anomalies.

3. **Spectral Context Attention (SCA)**:

    - Function: Fuse multiple patch-level detection results of an arbitrary-resolution image into an image-level decision.
    - Mechanism: Define a learnable query vector $q$, compute the attention weights $\mathcal{A} = \text{softmax}(q \cdot (z^S W_K)^\top / \sqrt{D_h})$ over the spectral vectors $z_k^S$ of all $K$ patches, and obtain the image-level representation via weighted fusion. The computational complexity is only $O(K)$, enabling efficient processing of megapixel-scale images.
    - Design Motivation: Simple resizing discards high-frequency information, which is a crucial clue for detection. SCA allows processing each patch at its original resolution, followed by attention-weighted fusion.

### Loss & Training
End-to-end training employs the binary cross-entropy loss $\mathcal{L}_{cls} = BCE(\hat{y}, y)$. During training, a fixed number $K_{\text{training}}=4$ of randomly augmented views is used instead of actual patches (addressing size constraints of training data), whereas the actual number of patches is used during inference. The training data comprises only 180k low-resolution (256×256) synthetic images from a single LDM model and 180k real images.

## Key Experimental Results

### Main Results
Cross-model detection AUC on 13 generation models (covering GANs, diffusion models, and commercial models):

| Resolution | Generative Model | SPAI (Ours) | RINE (Runner-up) | DMID | PatchCr. |
|---|---|---|---|---|---|
| < 0.5 MP | Glide | 90.2 | 95.6 | 73.1 | 78.4 |
| | SD1.3 | 99.6 | 99.9 | 100.0 | 95.7 |
| | SD1.4 | 99.6 | 99.9 | 100.0 | 96.2 |
| 0.5-1.0 MP | Flux | 83.0 | 93.0 | 97.2 | 86.9 |
| | DALLE2 | 91.1 | 93.0 | 54.3 | 81.8 |
| | SD2 | 96.5 | 96.6 | 99.7 | 95.7 |
| | SDXL | 97.4 | 99.3 | 99.6 | 96.7 |
| | SD3 | **75.9** | 39.1 | 67.9 | 33.8 |
| | GigaGAN | 85.4 | 92.9 | 67.9 | 98.0 |
| > 1.0 MP | MJv5 | 94.5 | 96.4 | 99.9 | 79.0 |
| | MJv6.1 | 84.0 | 81.2 | 94.4 | 96.1 |
| | DALLE3 | **90.2** | 41.8 | 41.3 | 28.1 |
| | Firefly | **96.0** | 82.9 | 90.2 | 79.1 |
| **Average** | | **91.0** | 85.5 | 83.5 | 80.4 |

SPAI leads the runner-up method by 5.5% in average AUC (91.0%), showing significant advantages on challenging models such as SD3, DALL-E 3, and Firefly.

### Ablation Study

| Configuration | AUC | Description |
|---|---|---|
| SPAI (Full) | 91.0 | All components |
| w/o spectral pretraining | 52.5 | W/o spectral pretraining, drops by 38.5% |
| w/o SRS | 71.0 | W/o SRS, drops by 20.0% |
| w/o SCV | 84.9 | W/o context vector, drops by 6.1% |
| w/o SCA | 83.2 | W/o attention fusion, drops by 7.8% |
| w/o SCA + TenCrop (mean) | 85.3 | Replace attention with mean, drops by 5.7% |
| w/o JPEG augm. | 89.1 | W/o JPEG augmentation |
| w/o distortions | 84.2 | W/o all distortion augmentations, drops by 6.8% |

| Backbone | Training Data | AUC |
|---|---|---|
| CLIP ViT-B/16 | 400M | 87.6 |
| DINOv2 ViT-B/14 | 142M | 87.5 |
| **MFM ViT-B/16 (Ours)** | **1.2M** | **91.0** |

### Key Findings
- **Spectral pre-training is critical**: Performance drops sharply to 52.5% without it, demonstrating that frequency reconstruction is the foundation of the entire approach.
- **Exceptional data efficiency**: The spectral ViT trained on only 1.2 million images outperforms CLIP trained on 400 million images and DINOv2 trained on 142 million images.
- **Outstanding robustness**: It outperforms all baseline methods under five types of perturbations: JPEG (Q=50), WebP (Q=50), Gaussian blur (k=7), Gaussian noise ($\sigma=5$), and resizing (50%).
- **SD3 and DALL-E 3 are the most challenging models**: All methods exhibit their worst performance on these two recent models; however, SPAI still achieves 75.9% on SD3 (runner-up is 67.9%) and 90.2% on DALL-E 3 (runner-up is 41.8%).

## Highlights & Insights
- **Paradigm Shift**: From "learning artifacts" to "modeling real distributions". This avoids chasing ever-evolving generation model artifacts, representing a more sustainable detection strategy. The choice of the spectral domain is particularly critical, as the spectrum of real images exhibits higher invariance than the spatial domain.
- **Arbitrary Resolution Processing**: The SCA module utilizes a single-query attention mechanism to fuse an arbitrary number of patches with $O(K)$ complexity, avoiding the loss of high-frequency information caused by resizing. This is crucial for actual deployment to process megapixel photos.
- **Self-Supervised Detection**: No annotations are required. The frequency reconstruction task is naturally suited for modeling real image distributions, requiring only real images for training, which theoretically never becomes outdated.

## Limitations & Future Work
- The AUC drops to 83.0% on Flux and 75.9% on SD3, indicating that the latest diffusion models are increasingly approaching real images in terms of spectral features.
- SCA requires independent ViT forward passes for each patch during inference, which may lead to high inference latency for high-resolution images.
- The spectral model is pre-trained solely on ImageNet; its generalization capability to specific domains (e.g., medical imaging) remains unexplored.
- The backbone is fixed to ViT-B/16; whether larger models (e.g., ViT-L/ViT-H) can yield further improvements has not been investigated.

## Related Work & Insights
- **vs RINE**: Both use pre-trained vision models, but RINE relies on CLIP's ViT-L backbone (400M images) and only achieves 85.5% AUC; whereas SPAI achieves 91.0% AUC with a spectral ViT trained on only 1.2M images.
- **vs DMID**: Directly learns spatial-domain artifacts, reaching near 100% on SD1.3/1.4 but only 41.3% on DALL-E 3; whereas SPAI generalizes more uniformly.
- **vs NPR**: Exploits upsampling layer artifacts, achieving 97.1% on DALL-E 3 but performing poorly on other models (19.8% on Flux); "one-trick pony" approaches are inferior to distribution modeling.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The paradigm shift to spectral distribution modeling + OOD detection is highly inspiring.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive evaluations across 13 generation models, 5 real image sources, 5 types of perturbations, and complete ablation studies.
- Writing Quality: ⭐⭐⭐⭐ The structure is clear, though some mathematical symbols are slightly redundant.
- Value: ⭐⭐⭐⭐⭐ Outstanding practical deployment value; supports arbitrary resolutions and generalizes well to unseen models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] A Bias-Free Training Paradigm for More General AI-generated Image Detection](a_bias-free_training_paradigm_for_more_general_ai-generated_image_detection.md)
- [\[CVPR 2025\] Where's the Liability in the Generative Era? Recovery-Based Black-Box Detection of AI-Generated Content](wheres_the_liability_in_the_generative_era_recovery-based_black-box_detection_of.md)
- [\[NeurIPS 2025\] Physics-Driven Spatiotemporal Modeling for AI-Generated Video Detection](../../NeurIPS2025/image_generation/physics-driven_spatiotemporal_modeling_for_ai-generated_video_detection.md)
- [\[ECCV 2024\] Zero-Shot Detection of AI-Generated Images](../../ECCV2024/image_generation/zero-shot_detection_of_ai-generated_images.md)
- [\[AAAI 2026\] Aggregating Diverse Cue Experts for AI-Generated Image Detection](../../AAAI2026/image_generation/aggregating_diverse_cue_experts_for_ai-generated_image_detec.md)

</div>

<!-- RELATED:END -->
