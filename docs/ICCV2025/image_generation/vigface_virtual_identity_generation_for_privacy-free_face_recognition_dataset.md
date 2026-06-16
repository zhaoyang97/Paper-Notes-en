---
title: >-
  [Paper Note] VIGFace: Virtual Identity Generation for Privacy-Free Face Recognition Dataset
description: >-
  [ICCV 2025][Image Generation][Virtual Identity Generation] This paper proposes VIGFace, a framework that pre-allocates virtual prototypes orthogonal to real identities in the feature space of a face recognition (FR) mode…
tags:
  - "ICCV 2025"
  - "Image Generation"
  - "Virtual Identity Generation"
  - "Privacy Safety"
  - "Synthetic Face Dataset"
  - "Diffusion Model"
  - "Face Recognition"
date: 2026-05-08
content_hash: 9917ca3228c40a4b
---

# VIGFace: Virtual Identity Generation for Privacy-Free Face Recognition Dataset

**Conference**: ICCV 2025
**arXiv**: [2403.08277](https://arxiv.org/abs/2403.08277)  
**Code**: [GitHub](https://github.com/kim1102/VIGFace)  
**Area**: Diffusion Models / Face Recognition
**Keywords**: Virtual Identity Generation, Privacy Safety, Synthetic Face Dataset, Diffusion Model, Face Recognition

## TL;DR

This paper proposes VIGFace, a framework that pre-allocates virtual prototypes orthogonal to real identities in the feature space of a face recognition (FR) model, and trains a diffusion model to generate face images conditioned on these prototypes—producing identities that do not exist in the real world, thereby enabling privacy-free face recognition dataset construction and data augmentation.

## Background & Motivation

Training deep face recognition models relies on large-scale face datasets, which face serious privacy and ethical challenges:

**Privacy Issues**: Existing datasets (e.g., CASIA-WebFace, MS-Celeb-1M) were collected via web scraping without subject consent. Datasets containing images of minors (e.g., VGGFace2) have been retracted due to privacy concerns.

**Three Deficiencies of Existing Synthetic Methods**: An ideal synthetic face dataset must simultaneously satisfy: (a) data distribution consistent with real data, (b) generated identities non-overlapping with real persons, and (c) intra-class consistency. However, no existing method satisfies all three—SynFace generates fewer than 500 distinct identities; DigiFace suffers from a domain gap due to 3D rendering; DCFace lacks data augmentation capability; IDiffFace cannot guarantee identity uniqueness.

**Identity Leakage Risk**: Some SOTA methods (CemiFace, HSFace) exhibit identity leakage—by sampling identity embeddings from the WebFace4M dataset, the generated faces closely resemble real persons in the training set, posing a significant privacy risk.

**Insufficient Intra-Class Diversity**: Real datasets often follow a long-tail distribution, where some identities have few images with limited variation, restricting model generalization.

The core insight of this paper is: if virtual identity positions can be pre-planned in feature space such that they are orthogonal to all real identities, face images generated from those positions will naturally not overlap with any real person.

## Method

### Overall Architecture

VIGFace consists of two stages:
1. **Stage 1: FR Model Training + Virtual Prototype Allocation**: Train an FR model on real data while simultaneously learning virtual identity prototypes.
2. **Stage 2: Diffusion-Based Face Generation**: Train a conditional diffusion model in the feature space of the pretrained FR model to generate face images.

### Key Designs

1. **Orthogonal Allocation of Virtual Prototypes (Core Innovation)**: Standard ArcFace training uses only real identity prototypes $W_R = [w_r^1, ..., w_r^n]$. This work additionally introduces $k$ virtual prototypes $W_V = [w_v^1, ..., w_v^k]$, extending the prototype matrix to $W \in \mathbb{R}^{(n+k) \times D}$. The key challenge is that virtual identities have no corresponding real images, so their prototypes cannot be updated naturally via ArcFace loss. The solution is to generate virtual embeddings (simulating the distribution of real embeddings) to update the virtual prototypes:

    $f'_{FR}(x_j) = w_v^j + \mathcal{N}(0, 1) \cdot \sigma$
    $\sigma^2 = \frac{1}{b} \sum_{i=1}^{b} (f_{FR}(x_i) - w_r^i)^2$

   The standard deviation $\sigma$ of virtual embeddings is matched to the distribution of real embeddings (smoothed via EMA). Virtual and real embeddings are then jointly fed into the ArcFace loss. This pushes virtual prototypes away from all other prototypes (real and virtual) during training, ultimately forming an orthogonal distribution in feature space. Gradients from virtual embeddings are not back-propagated to the backbone; only virtual prototypes are updated.

2. **Conditional Diffusion Face Generation**: A DiT architecture is adopted, with conditioning inputs including: FR prototype vector $w_r$ (identity condition), five-point facial landmark image $y$ (pose condition), and timestep $t$. The model predicts velocity $v_t$ rather than noise. A key constraint minimizes the feature distance between generated images and the input prototype:

    $\min_\theta \mathbb{E}_{\epsilon,t} \| f_{FR}(\hat{x}_\theta(x_t, t, w_r, y)) - w_r \|_2^2$

   Classifier-free guidance is applied (with $w_r$ set to zero with 10% probability); a guidance weight of $g=4.0$ yields the best results at inference. Five-point landmarks are extracted by RetinaFace, enabling control over pose variation in generated images.

3. **Dataset Attribute Metrics**: Three evaluation metrics are proposed to quantify synthetic dataset quality:

    - **Class Consistency** $C_k$: Mean cosine similarity of intra-class feature pairs, measuring identity coherence.
    - **Class Separability** $S_k$: Mean distance between class centers and negative class centers, measuring identity uniqueness.
    - **Intra-Class Diversity** $D_k$: Variance of CR-FIQA scores, measuring the richness of pose, occlusion, and lighting conditions.

### Loss & Training

- **Stage 1**: ArcFace loss $L_{arc}$ applied jointly to real and virtual embeddings.
- **Stage 2**: Velocity-prediction MSE loss + feature distance constraint loss.
- The number of virtual embeddings per batch is $b_v = (k \times b_r) / n$, ensuring balanced updates between virtual and real prototypes.
- EMA smoothing coefficient $\alpha = 0.9$.
- Training dataset: CASIA-WebFace (~0.49M images, ~10.5K identities).

## Key Experimental Results

### Main Results

**FR Benchmark Comparison Trained on Purely Synthetic Data (IR-SE50 + AdaFace)**:

| Method | Training Source | # Images | LFW | CFP-FP | CPLFW | AgeDB | CALFW | Avg. |
|--------|----------------|-----------|-----|--------|-------|-------|-------|------|
| CASIA (Real) | - | 0.49M | 99.40 | 96.63 | 90.23 | 94.68 | 93.70 | 94.93 |
| SynFace | FFHQ | 0.5M | 91.93 | 75.03 | 70.43 | 61.63 | 74.73 | 74.75 |
| DCFace | FFHQ+CASIA | 0.5M | 98.55 | 85.33 | 82.62 | 89.70 | 91.60 | 89.56 |
| CemiFace | CASIA+WF4M | 0.5M | 99.03 | 91.06 | 87.62 | 91.33 | 92.42 | 92.30 |
| HSFace300K | WF4M | 15M | 99.30 | 91.54 | 87.70 | 94.45 | 94.58 | 93.52 |
| **VIGFace(S)** | **CASIA** | **0.5M** | **99.02** | **95.09** | **87.72** | 90.95 | 90.00 | **92.56** |
| **VIGFace(L)** | **CASIA** | **6.0M** | 99.33 | **97.31** | **91.12** | 93.82 | 92.95 | **94.91** |

### Ablation Study

**Data Augmentation Effect (Real + Synthetic Data Combinations)**:

| Setting | Real Images | Synthetic (Real ID) | Synthetic (Virtual ID) | LFW | CFP-FP | Avg. | Δ |
|---------|------------|--------------------|-----------------------|-----|--------|------|---|
| CASIA | ✓ | | | 99.40 | 96.63 | 94.93 | - |
| +Virtual ID Aug. | ✓ | | ✓ | 99.45 | 97.23 | 95.19 | +0.26 |
| +Real ID Aug. | ✓ | ✓ | | 99.55 | 98.03 | 95.85 | +0.92 |
| +All Aug. | ✓ | ✓ | ✓ | **99.70** | **98.10** | **95.92** | **+0.99** |

**Real ID augmentation** expands long-tail classes (<50 images) to 50 images, adding approximately 0.15M additional images.

### Key Findings

- VIGFace(L) surpasses HSFace300K (94.91 vs. 93.52 average accuracy) using only 6M images compared to HSFace300K's 15M.
- On CFP-FP (cross-pose) and CPLFW benchmarks, VIGFace even outperforms models trained on real data, attributed to the pose diversity enabled by five-point landmark conditioning.
- VIGFace(B)'s nearest real-identity cosine similarity is lower than CASIA-WebFace's own nearest negative-class similarity, demonstrating the absence of identity leakage.
- CemiFace and Vec2Face exhibit identity leakage: generated faces are highly similar to real persons in the WebFace4M training set.
- Augmentation effect: Real ID augmentation (+0.92) contributes more than Virtual ID augmentation (+0.26); combining both yields the best result (+0.99).

## Highlights & Insights

- **Mathematical Elegance of Orthogonal Prototype Design**: The method exploits the near-orthogonality of vectors in high-dimensional spaces, naturally pushing virtual prototypes away from real identities via ArcFace loss—without requiring post-hoc thresholding or selective sampling.
- **Rigorous Privacy Verification**: Identity safety is quantitatively demonstrated via cosine similarity rather than relying solely on visual inspection. The analysis also reveals identity leakage in CemiFace and Vec2Face.
- **Dual Capability as Substitute and Augmentor**: VIGFace can serve as a complete replacement for real datasets in privacy-sensitive scenarios, and can also be combined with real data to improve performance.
- **Dataset Quality Evaluation Framework**: The proposed consistency/separability/diversity three-dimensional evaluation framework is generalizable to quality assessment of other synthetic datasets.

## Limitations & Future Work

- The number of virtual identities is constrained by the dimensionality of the embedding space; orthogonality may degrade when $n+k$ is excessively large.
- Pose control via five-point landmarks is relatively coarse; finer-grained conditioning using 3DMM or denser landmarks could be explored.
- Validation is currently limited to the CASIA-WebFace scale; scalability to larger datasets (e.g., WebFace4M) warrants further investigation.
- The resolution and visual quality of generated images have room for improvement.
- Extending the virtual prototype approach to other biometric recognition scenarios requiring identity protection is a promising direction.

## Related Work & Insights

- The angular margin mechanism of ArcFace loss underpins the orthogonalization of virtual prototypes.
- The trend of applying DiT architectures to conditional generation is leveraged in this work.
- DCFace's dual-condition disentanglement (identity + style) informed the identity conditioning design of this paper.
- The identity leakage analysis establishes a new standard for evaluating synthetic face datasets.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ The virtual prototype orthogonalization idea is highly original and simultaneously addresses both privacy and performance concerns.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Comprehensive evaluation across 5 benchmarks, multiple data scales, augmentation experiments, and identity leakage analysis.
- **Writing Quality**: ⭐⭐⭐⭐ Motivation is clearly articulated, with rich visualizations (t-SNE, similarity matrices).
- **Value**: ⭐⭐⭐⭐⭐ Addresses a long-standing privacy dilemma in face recognition with significant practical implications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] OmniVTON: Training-Free Universal Virtual Try-On](omnivton_training-free_universal_virtual_try-on.md)
- [\[ICCV 2025\] NullSwap: Proactive Identity Cloaking Against Deepfake Face Swapping](nullswap_proactive_identity_cloaking_against_deepfake_face_swapping.md)
- [\[ICCV 2025\] Bridging the Skeleton-Text Modality Gap: Diffusion-Powered Modality Alignment for Zero-shot Skeleton-based Action Recognition](bridging_the_skeleton_text_modality_gap_diffusion_powered_modality_alignment_for.md)
- [\[ICCV 2025\] CaO2: Rectifying Inconsistencies in Diffusion-Based Dataset Distillation](cao2_rectifying_inconsistencies_in_diffusion-based_dataset_distillation.md)
- [\[ICCV 2025\] Unlocking the Potential of Diffusion Priors in Blind Face Restoration](unlocking_the_potential_of_diffusion_priors_in_blind_face_restoration.md)

</div>

<!-- RELATED:END -->
