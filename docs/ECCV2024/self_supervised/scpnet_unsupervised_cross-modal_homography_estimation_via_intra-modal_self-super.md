---
title: >-
  [Paper Note] SCPNet: Unsupervised Cross-modal Homography Estimation via Intra-modal Self-supervised Learning
description: >-
  [ECCV 2024][Self-Supervised Learning][Homography Estimation] SCPNet is proposed, which of first achieves effective unsupervised cross-modal homography estimation on datasets with large modal discrepancy (e.g., satellite-to-map) through the synergy of three key components: intra-modal self-supervised learning, correlation networks, and consistent feature map projection, achieving an MACE that is 14% lower than the supervised method MHN.
tags:
  - "ECCV 2024"
  - "Self-Supervised Learning"
  - "Homography Estimation"
  - "Unsupervised Learning"
  - "Cross-modal"
  - "Multispectral Images"
date: 2026-05-08
content_hash: 3e52da9d68e8b04d
---

# SCPNet: Unsupervised Cross-modal Homography Estimation via Intra-modal Self-supervised Learning

**Conference**: ECCV 2024  
**arXiv**: [2407.08148](https://arxiv.org/abs/2407.08148)  
**Code**: [https://github.com/RM-Zhang/SCPNet](https://github.com/RM-Zhang/SCPNet)  
**Area**: Remote Sensing  
**Keywords**: Homography Estimation, Unsupervised Learning, Cross-modal, Self-supervised Learning, Multispectral Images

## TL;DR
SCPNet is proposed, which of first achieves effective unsupervised cross-modal homography estimation on datasets with large modal discrepancy (e.g., satellite-to-map) through the synergy of three key components: intra-modal self-supervised learning, correlation networks, and consistent feature map projection, achieving an MACE that is 14% lower than the supervised method MHN.

## Background & Motivation

**Background**: Homography estimation aims to compute the global perspective transformation between images, serving as a foundation for tasks such as image registration, fusion, and navigation. Supervised approaches (DHN, MHN, IHN, RHWF) perform exceptionally well under large displacements, but obtaining ground-truth annotations for cross-modal images is extremely difficult. Unsupervised methods (UDHN, CA-UDHN, biHomE) are trained using pixel- or feature-level intensity consistency losses and do not require ground-truth annotations.

**Limitations of Prior Work**: Existing unsupervised methods primarily rely on "cross-modal intensity learning"—comparing the warped source image (warped according to the predicted homography) with the target image. However, when the modality gap is large (e.g., satellite imagery vs. maps) and the displacement is large ($\pm32$ pixels for a $128\times128$ image), cross-modal intensity similarity becomes highly non-convex, making optimization highly prone to non-convergence. On the GoogleMap dataset, all existing unsupervised methods fail or perform extremely poorly.

**Key Challenge**: The core contradiction of unsupervised learning is the "lack of direct supervision signals"—cross-modal images show massive appearance discrepancies, rendering pixel- or feature-level similarity metrics unreliable, while ground-truth homographies remain unavailable.

**Goal**: How to achieve effective unsupervised cross-modal homography estimation under large modality gaps and substantial displacements?

**Key Insight**: The authors propose a critical insight—although cross-modal ground-truth homographies are hard to obtain, intra-modal ground-truth homographies can be easily generated through simulated deformation. By conducting self-supervised learning inside each of the two modalities, a weight-shared network can generalize homography estimation knowledge from intra-modal to cross-modal. Pilot experiments verified this finding: a network trained solely with intra-modal self-supervision performs better in cross-modal testing than one trained directly using cross-modal intensity learning.

**Core Idea**: To provide reliable indirect supervision for cross-modal homography estimation through "intra-modal self-supervised learning", combined with a correlation network and consistent feature map projection to construct a powerful unsupervised learning framework.

## Method

### Overall Architecture
The training framework of SCPNet consists of three branches: two intra-modal self-supervised learning branches (which perform supervised regression after simulating homography transformations on Modality A and Modality B, respectively) and one cross-modal learning branch (which imposes unsupervised constraints by comparing consistent feature maps). The three branches share the weights of learnable modules and are trained simultaneously. During inference, only the cross-modal prediction branch is utilized.

### Key Designs

1. **Intra-modal Self-supervised Learning**:

    - Function: To provide indirect yet reliable supervision signals for cross-modal homography estimation.
    - Mechanism: A random simulated homography deformation is applied to image $\mathbf{I}_A$ of Modality A to obtain $\mathbf{I}_{A'}$, constructing the triplet $(\mathbf{I}_A, \mathbf{I}_{A'}, \mathbf{H}_{GT,A})$; Modality B is processed similarly. The weight-shared network is used to predict the intra-modal homography for both modalities, supervised by an L1 loss: $\mathcal{L}_S = \|\mathbf{O} - \mathbf{O}_{GT}\|_1$, where O represents the parameterized 4-corner displacement. Key Finding: Even without using any cross-modal loss, the network trained solely on intra-modal self-supervision can achieve comparable performance in cross-modal testing (MACE=13.06 vs. non-convergence of cross-modal training).
    - Design Motivation: Inspired by multi-task learning, the intra-modal self-supervised task is highly correlated with the cross-modal estimation task. The shared representation implicitly learns the cross-modal structural correspondence. This is equivalent to using "free" simulated data to provide high-quality supervision, bypassing the core challenge of unreliable cross-modal intensity losses.

2. **Correlation-based Homography Estimation Network**:

    - Function: To constrain the network to learn clearer, more transferable knowledge by explicitly computing feature correlations.
    - Mechanism: A weight-shared feature extractor is used to extract feature maps $\mathbf{F}_A$ and $\mathbf{F}_B$ for the two modalities respectively, and inner-product correlation is calculated within a local window: $\mathbf{C}(\mathbf{x}, \mathbf{r}) = \text{ReLU}(\mathbf{F}_A(\mathbf{x})^T \mathbf{F}_B(\mathbf{x}+\mathbf{r}))$, with $\|\mathbf{r}\|_\infty \leq R$. The correlation map is fed into a homography estimator to predict displacements. Compared to traditional methods that directly concatenate image pairs as network input, correlation decomposes the network into three clear stages: "feature extraction $\rightarrow$ similarity encoding $\rightarrow$ homography decoding."
    - Design Motivation: Correlation explicitly encodes feature similarity, normalizing the homography decoding knowledge across both intra-modal and cross-modal scenarios. Regardless of the modality, the patterns in the correlation maps are consistent, significantly facilitating the migration of intra-modal knowledge to cross-modal estimation.

3. **Consistent Feature Map Projection**:

    - Function: To project cross-modal images from an intensity-variant space to an intensity-invariant latent space, making cross-modal intensity supervision feasible.
    - Mechanism: Convolutional blocks (3×3 conv + residual block + 1×1 conv) project the input images into single-channel consistent feature maps. The cross-modal loss is calculated on the projected feature maps: $\mathcal{L}_C = \|\mathbf{P}_A - \mathbf{P}_{B,W}\|_1 / \|\mathbf{P}_A - \mathbf{P}_B\|_1$, where the numerator minimizes the similarity after warping, and the denominator maximizes the unwarped difference to prevent degenerate solutions.
    - Design Motivation: Directly comparing cross-modal images in the raw pixel or simple feature space fails due to modality discrepancies. Consistent projection learns a structural representation shared by both modalities, making intensity supervision feasible. Meanwhile, the "anti-degeneration" design of the denominator term is ingenious, preventing the network from projecting all images into constants.

### Loss & Training
The total loss is the weighted sum of the three branches: $\arg\min_{\xi,\zeta} \mathcal{L}_C(\delta_\zeta(\mathbf{I}_A), \mathcal{W}(\delta_\zeta(\mathbf{I}_B), \psi_\xi(\cdot))) + \lambda \mathcal{L}_S(\psi_\xi(\delta_\zeta(\mathbf{I}_A), \delta_\zeta(\mathbf{I}_{A'})), \mathbf{H}_{GT,A}) + \lambda \mathcal{L}_S(\psi_\xi(\delta_\zeta(\mathbf{I}_B), \delta_\zeta(\mathbf{I}_{B'})), \mathbf{H}_{GT,B})$, where $\lambda=0.1$. The AdamW optimizer is used with a learning rate of $4\times10^{-4}$ and a batch size of 8, trained for 120K iterations.

## Key Experimental Results

### Main Results

**Cross-modal dataset (GoogleMap, [-32,+32] offset)**:

| Method | Type | Easy↓ | Moderate↓ | Hard↓ | Mean MACE↓ |
|------|------|-------|-----------|-------|------------|
| SIFT | Handcrafted | 19.17 | 23.87 | 29.04 | 24.53 |
| UDHN | Unsupervised | 18.63 | 21.55 | 26.89 | 22.84 |
| CA-UDHN | Unsupervised | NC | NC | NC | NC |
| biHomE | Unsupervised | NC | NC | NC | NC |
| DHN | Supervised | - | - | - | ~6.9 |
| MHN | Supervised | - | - | - | ~5.06 |
| **SCPNet** | **Unsupervised** | - | - | - | **4.35** |

**Cross-spectral dataset (Harvard)**:

| Method | Type | Mean MACE↓ |
|------|------|------------|
| MHN | Supervised | ~10.2 |
| **SCPNet** | **Unsupervised** | 25.2% lower than MHN |

### Ablation Study

| Setting | Self | Correlation | Projection | Cross | MACE↓ |
|---------|------|-------------|------------|-------|-------|
| 1 | ✗ | ✗ | ✗ | ✓ | NC |
| 2 | ✗ | ✓ | ✗ | ✓ | NC |
| 3 | ✗ | ✗ | ✓ | ✓ | 24.64 |
| 5 | ✓ | ✗ | ✗ | ✗ | 13.06 |
| 6 | ✓ | ✓ | ✗ | ✗ | 9.68 |
| 8 | ✓ | ✓ | ✓ | ✗ | 7.70 |
| **9** | **✓** | **✓** | **✓** | **✓** | **4.35** |

### Key Findings
- Intra-modal self-supervised learning is the cornerstone of the entire framework: without it (Settings 1-4), training either fails to converge or has very poor accuracy; using it alone (Setting 5) yields a MACE of 13.06.
- Gradual addition of the three components leads to continuous improvement: 13.06 $\rightarrow$ 9.68 (+Correlation) $\rightarrow$ 7.70 (+Projection) $\rightarrow$ 4.35 (+Cross-modal Loss).
- Correlation networks improve the quality of generated consistent feature maps—ablation experiments demonstrate that structures in projected feature maps are clearer when correlation is utilized.
- Computational overhead: Intra-modal self-supervised training increases the training time from 3.18h to 6.98h and VRAM consumption from 4.6GB to 9.1GB, but introduces no extra overhead during inference.

## Highlights & Insights
- **The core insight of intra-modal self-supervised learning is brilliant**: Translating "free" intra-modal simulated supervision into cross-modal capacity essentially exploits the positive transfer effect of shared-weight networks in multi-task learning. This concept can be extended to any cross-domain/cross-modal unsupervised learning scenarios—as long as the tasks share some low-level knowledge, intra-modal self-supervision can act as a stepping stone.
- **Denominator design of the loss function**: The denominator term $\|\mathbf{P}_A - \mathbf{P}_B\|_1$ in $\mathcal{L}_C$ ingeniously prevents degenerate solutions—ensuring the meaningfulness of projections while avoiding additional regularization terms.
- The ablation study is systematically designed, comprehensively presenting both individual and synergistic contributions of each component, which serves as an excellent example.

## Limitations & Future Work
- The authors point out that multi-scale, iterative optimization, and Transformer architectures can be introduced to replace CNNs, further improving accuracy.
- Performs worse than supervised methods like LocalTrans, IHN, and RHWF on the Flash/no-flash dataset, showing that the ceiling of unsupervised methods still exhibits a gap when modality discrepancies are small.
- The training time is doubled (3.18h $\rightarrow$ 6.98h). More efficient self-supervised training strategies (such as progressive addition of intra-modal branches) can be explored.
- This work only validates the displacement range of $\pm 32$ at $128 \times 128$ resolution. The performance on larger resolutions and deformation ranges remains to be verified.

## Related Work & Insights
- **vs UDHN/CA-UDHN**: These methods only use cross-modal intensity losses and completely fail (NC) under large modality discrepancies. SCPNet's intra-modal self-supervised learning fundamentally solves the issue of missing supervision signals.
- **vs biHomE**: Switching to perceptual loss also fails to converge on GoogleMap. This indicates that the root cause of the problem is not the design of the loss function, but rather the unreliability of direct cross-modal comparison itself.
- **vs MHN (Supervised)**: SCPNet achieves a MACE that is 14% lower than MHN on GoogleMap, representing a rare case where an unsupervised method outperforms a supervised one, highlighting the potential of intra-modal self-supervised learning.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The core idea of intra-modal self-supervised learning is highly creative, and the design and insights from pilot experiments are convincing.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 5 datasets, 14 baseline methods, and a systematic set of 9 ablation experiments.
- Writing Quality: ⭐⭐⭐⭐ The narrative structure flowing from pilot experiments to method designs is excellent, though it contains relatively dense formulas and notations.
- Value: ⭐⭐⭐⭐ A landmark case where unsupervised outperforms supervised, of which the methodology is transferable to other cross-modal tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Rethinking Unsupervised Cross-Modal Flow Estimation: Learning from Decoupled Optimization and Consistency Constraint](../../ICLR2026/self_supervised/rethinking_unsupervised_cross-modal_flow_estimation_learning_from_decoupled_opti.md)
- [\[ECCV 2024\] WeCromCL: Weakly Supervised Cross-Modality Contrastive Learning for Transcription-only Supervised Text Spotting](wecromcl_weakly_supervised_cross-modality_contrastive_learning_for_transcription.md)
- [\[ECCV 2024\] ViC-MAE: Self-Supervised Representation Learning from Images and Video with Contrastive Masked Autoencoders](vic-mae_self-supervised_representation_learning_from_images_and_video_with_contr.md)
- [\[ECCV 2024\] Self-supervised Video Copy Localization with Regional Token Representation](self-supervised_video_copy_localization_with_regional_token_representation.md)
- [\[ECCV 2024\] Rethinking Unsupervised Outlier Detection via Multiple Thresholding](rethinking_unsupervised_outlier_detection_via_multiple_thresholding.md)

</div>

<!-- RELATED:END -->
