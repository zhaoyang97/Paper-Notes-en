---
title: >-
  [Paper Note] DeepShield: Fortifying Deepfake Video Detection with Local and Global Forgery Analysis
description: >-
  [ICCV 2025][Image Generation][Deepfake Detection] This paper proposes DeepShield, a deepfake video detection framework that combines Local Patch Guidance (LPG) and Global Forgery Diversification (GFD). It provides patch-…
tags:
  - "ICCV 2025"
  - "Image Generation"
  - "Deepfake Detection"
  - "CLIP-ViT"
  - "Patch-Level Supervision"
  - "Feature Augmentation"
  - "Cross-Domain Generalization"
date: 2026-05-08
content_hash: 799640e105e48e42
---

# DeepShield: Fortifying Deepfake Video Detection with Local and Global Forgery Analysis

**Conference**: ICCV 2025
**arXiv**: [2510.25237](https://arxiv.org/abs/2510.25237)  
**Code**: [GitHub](https://github.com/lijichang/DeepShield)  
**Area**: Deepfake Detection / Image Generation
**Keywords**: Deepfake Detection, CLIP-ViT, Patch-Level Supervision, Feature Augmentation, Cross-Domain Generalization

## TL;DR

This paper proposes DeepShield, a deepfake video detection framework that combines Local Patch Guidance (LPG) and Global Forgery Diversification (GFD). It provides patch-level supervision via spatiotemporal artifact modeling and synthesizes diverse forgery representations through distribution-level feature augmentation, significantly outperforming state-of-the-art methods in cross-dataset and cross-manipulation evaluations.

## Background & Motivation

Deep generative models (GANs, VAEs, diffusion models) have substantially lowered the barrier to face video manipulation, making forgery detection a critical problem. Existing detectors face two major challenges:

**Insufficient local sensitivity**: Methods based on large models such as CLIP primarily leverage global features and tend to focus on the most salient forgery traces (e.g., exaggerated transitions in face swapping), while overlooking subtle cues (e.g., blending boundaries and minor texture inconsistencies).

**Poor cross-domain generalization**: Models tend to overfit specific manipulation types seen during training, suffering significant performance degradation against unseen forgery methods. Approaches relying on retraining or data augmentation are costly and poorly scalable.

**Core Idea**: A local-global learning paradigm is adopted to enable the model to capture fine-grained patch-level forgery traces while enhancing cross-domain generalization through forgery diversification in feature space.

## Method

### Overall Architecture

DeepShield is built upon CLIP-ViT-B/16 with ST-Adapter and comprises two complementary components:
- **Local Patch Guidance (LPG)**: patch-level supervised learning with spatiotemporal artifact modeling
- **Global Forgery Diversification (GFD)**: domain feature augmentation with contrastive learning objective

### Key Designs

1. **Spatiotemporal Artifact Modeling (SAM)**:

    - Extends the SBI technique from image-level to video-level, generating forged videos with precise masks.
    - **Spatial artifacts**: For each frame, the inner region (face) and outer region (background) are augmented separately; blending is performed via a convex-hull mask derived from facial landmarks to introduce statistical inconsistencies.
    - **Temporal artifacts**: Consistent augmentation directions and mask transformation strategies are maintained across $T$ frames to simulate the generation patterns of real deepfakes.
    - Key design: augmentation types remain consistent across frames, while random mask deformation and blurring exhibit minor inter-frame variations.

2. **Local Patch Guidance (LPG)**:

    - Each frame and its corresponding mask are divided into $P$ non-overlapping patches; binary patch-level labels are assigned via PatchMaskScore with threshold $\theta$.
    - Each patch token is treated as an independent training sample and optimized with a binary classifier $\phi$ under cross-entropy loss $\mathcal{L}_{\text{LPG}}$.
    - Effect: patch embeddings from the same class cluster together while those from different classes are separated, and the cls token aggregates richer local semantics through self-attention.

3. **Global Forgery Diversification (GFD)**:

    - **Domain-Bridging Feature Generation (DFG)**: Videos from different manipulation types are randomly paired; a mixing weight $\lambda$ is sampled from Beta(0.1, 0.1), and the mean/variance statistics of the two domains are blended via AdaIN to generate cross-domain bridging features.
    - **Boundary-Expanding Feature Generation (BFG)**: The standard deviation is scaled by $\alpha=1.1$ to push features beyond existing domain boundaries, expanding the detection coverage.
    - Training objective: cross-entropy loss + supervised contrastive loss $\mathcal{L}_{\text{GFD}} = \mathcal{L}^{\text{cls}} + \upsilon \mathcal{L}^{\text{supCon}}$

### Loss & Training

$$\mathcal{L}^{\text{overall}} = \omega \mathcal{L}_{\text{LPG}} + \mathcal{L}_{\text{GFD}}$$

- $\omega = 0.5$ balances local and global losses
- $\upsilon = 0.5$ balances cross-entropy and contrastive losses
- Adam optimizer with cosine learning rate decay, 80 epochs
- Input: 4 clips sampled per video, 12 consecutive frames per clip

## Key Experimental Results

### Main Results (Cross-Dataset Evaluation, Trained on FF++ HQ, Video-level AUC %)

| Method | Architecture | FF++ | CDF | DFDCP | DFDC | DFD |
|------|------|------|-----|-------|------|-----|
| SBI | ResNet50 | - | 85.7 | - | - | 94.0 |
| AltFreezing | 3D | 99.7 | 89.0 | - | - | 93.7 |
| TALL | Swin-B | 99.9 | 90.8 | - | 76.8 | - |
| LSDA | EfficientNet | - | 91.1 | 81.2 | 77.0 | 95.6 |
| SeeABLE | EfficientNet | - | 87.3 | 86.3 | 75.9 | - |
| VB-StA | CLIP ViT-B/16 | - | 86.6 | - | 77.8 | - |
| **DeepShield** | **CLIP ViT-B/16+ST-Adapter** | **99.2** | **92.2** | **93.2** | **82.8** | **96.1** |

DeepShield surpasses the previous best by **6.9%** and **5.8%** on the most challenging DFDCP and DFDC benchmarks, respectively.

### Ablation Study

**Component contributions of LPG and GFD (Cross-Dataset AUC %)**:

| Variant | CDF | DFDC | DFDCP | Avg |
|------|-----|------|-------|-----|
| DeepShield (full) | **92.2** | **82.8** | **93.2** | **89.4** |
| w/o LPG | 89.1 | 81.3 | 87.1 | 85.8 |
| w/o GFD | 89.0 | 81.9 | 91.9 | 87.6 |
| w/o LPG & GFD | 85.4 | 78.4 | 88.9 | 84.2 |

**DFA component contributions**:

| DFG | BFG | CDF | DFDC | DFDCP | Avg |
|:---:|:---:|-----|------|-------|-----|
| ✔ | ✔ | **92.2** | **82.8** | **93.2** | **89.4** |
| ✗ | ✔ | 91.1 | 81.9 | 92.9 | 88.6 |
| ✔ | ✗ | 90.3 | 82.1 | 92.5 | 88.3 |
| ✗ | ✗ | 92.0 | 81.3 | 91.6 | 88.3 |

### Key Findings

- Removing LPG causes a 3.6% drop in Avg AUC, with a particularly sharp 6.1% decline on DFDCP — demonstrating that local awareness is critical for complex forgeries.
- Removing GFD leads to a 1.8% drop, primarily affecting cross-domain generalization.
- Removing both components results in a 5.2% drop, confirming the synergistic effect of local and global analysis.
- DFG and BFG each contribute positively and complementarily — DFG bridges known domains while BFG expands the decision boundary.
- In cross-manipulation evaluation, training on FS yields an average AUC exceeding WATCHER by **15.95%**.
- GradCAM visualizations show that LPG enables the model to attend uniformly across the entire manipulated region rather than focusing solely on the most salient artifacts.
- t-SNE visualizations reveal that the full DeepShield achieves far superior real/fake feature separation compared to ablated variants.

## Highlights & Insights

- **Local-global synergistic paradigm**: LPG provides fine-grained local awareness while GFD delivers diversified global representations; the two are deeply coupled in an iterative collaborative manner.
- **Careful design of SAM**: Extending SBI to the video level is non-trivial — it requires ensuring cross-frame augmentation consistency and naturalness of mask variation.
- **Distribution-level feature augmentation outperforms linear interpolation**: AdaIN blending in DFG and standard deviation scaling in BFG produce richer nonlinear augmentations compared to simple linear interpolation.
- The design builds upon fine-tuning CLIP-ViT, preserving the generalization advantages of the pretrained model.

## Limitations & Future Work

- In-domain performance on FF++ is slightly below some competing methods, suggesting a modest trade-off when the local-global strategy is applied in overfitting-prone scenarios.
- The scaling factor $\alpha=1.1$ in BFG is empirically determined and may require adjustment for different datasets.
- The detection effectiveness against forgeries generated by the latest diffusion models remains unexplored.
- Patch-level labels depend on the quality of masks produced by SAM; imprecise masks may introduce noisy labels.

## Related Work & Insights

- **SBI**: The foundational method for spatial artifact synthesis; this work extends it to the spatiotemporal dimension.
- **LSDA**: A pioneer in linear feature-space augmentation; this work replaces linear interpolation with distribution-level augmentation.
- **AltFreezing**: Captures spatiotemporal information via alternating training of spatial/temporal convolutional layers, but generalization is limited when trained from scratch.
- **ST-Adapter**: A key component enabling parameter-efficient fine-tuning of CLIP-ViT.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The combination of local patch guidance and global forgery diversification is original; the video-level extension of SAM is carefully designed.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Dual evaluation across datasets and manipulation types, with detailed ablations, GradCAM, and t-SNE visualizations.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure, though the notation-heavy formulations require careful cross-referencing.
- **Value**: ⭐⭐⭐⭐⭐ Substantial gains in cross-domain detection (DFDC +5.8%) carry significant implications for real-world deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] ForgeLens: Data-Efficient Forgery Focus for Generalizable Forgery Image Detection](forgelens_data-efficient_forgery_focus_for_generalizable_forgery_image_detection.md)
- [\[ICML 2026\] From Talking to Singing: A New Challenge for Audio-Visual Deepfake Detection](../../ICML2026/image_generation/from_talking_to_singing_a_new_challenge_for_audio-visual_deepfake_detection.md)
- [\[ICML 2026\] Divide and Conquer: Reliable Multi-View Evidential Learning for Deepfake Detection](../../ICML2026/image_generation/divide_and_conquer_reliable_multi-view_evidential_learning_for_deepfake_detectio.md)
- [\[NeurIPS 2025\] FerretNet: Efficient Synthetic Image Detection via Local Pixel Dependencies](../../NeurIPS2025/image_generation/ferretnet_efficient_synthetic_image_detection_via_local_pixel_dependencies.md)
- [\[CVPR 2026\] MPDiT: Multi-Patch Global-to-Local Transformer Architecture for Efficient Flow Matching](../../CVPR2026/image_generation/mpdit_multi-patch_global-to-local_transformer_architecture_for_efficient_flow_ma.md)

</div>

<!-- RELATED:END -->
