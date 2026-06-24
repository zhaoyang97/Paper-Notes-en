---
title: >-
  [Paper Note] CryptoFace: End-to-End Encrypted Face Recognition
description: >-
  [Human Understanding] This paper proposes CryptoFace, the first end-to-end Fully Homomorphic Encrypted (FHE) face recognition system. By utilizing a hybrid shallow patch CNN architecture (CryptoFaceNet), it significantly reduces the multiplicative depth, achieving encrypted inference that is 7 times faster than state-of-the-art (SOTA) FHE networks while improving verification accuracy.
tags:
  - "Human Understanding"
date: 2026-05-08
content_hash: 00852e2489baefc6
---

# CryptoFace: End-to-End Encrypted Face Recognition

| Property | Value |
|------|------|
| Conference | CVPR 2025 |
| arXiv | [2509.00332](https://arxiv.org/abs/2509.00332) |
| Code | [GitHub](https://github.com/human-analysis/CryptoFace) |
| Area | Human Understanding / Face Recognition / Privacy Protection |
| Keywords | fully homomorphic encryption, face recognition, patch CNN, privacy-preserving, CryptoFaceNet |

## TL;DR

This paper proposes CryptoFace, the first end-to-end Fully Homomorphic Encrypted (FHE) face recognition system. By utilizing a hybrid shallow patch CNN architecture (CryptoFaceNet), it significantly reduces the multiplicative depth, achieving encrypted inference that is 7 times faster than state-of-the-art (SOTA) FHE networks while improving verification accuracy.

## Background & Motivation

### Background

Face recognition is widely used in device unlocking, law enforcement, and financial services, but it faces severe privacy risks. Biometric data is immutable, and once leaked, the consequences are irreversible. Homomorphic Encryption (HE) allows computations on encrypted data, providing provable post-quantum security. Existing secure face recognition systems only encrypt features rather than raw images.

### Limitations of Prior Work

1. **Incomplete Security Protection**: Existing systems only encrypt the features during matching. Clients must perform feature extraction locally and cannot fully delegate computation to the server. Malicious clients can infer original templates through feature reconstruction attacks.
2. **Extremely Slow FHE CNN Inference**: The high multiplicative depth of SOTA FHE networks (MPCNN, AutoFHE) requires a large number of bootstrapping operations, leading to inference times of up to hours.
3. **Inability to Handle High-Resolution Face Images**: Existing FHE CNNs only support small resolutions (e.g., CIFAR's $32 \times 32$) and cannot directly process face images.
4. **Inability to Directly Compute Cosine Similarity under FHE**: $\ell_2$ normalization involves non-homomorphic non-linear operations.

### Goal

To build the first **end-to-end** encrypted face recognition system: from encrypted image input to encrypted matching result output, without any decryption throughout the entire pipeline, while achieving acceptable inference latency and high recognition accuracy.

### Key Insight & Core Idea

The face image is partitioned into patches, which are processed independently using multiple shallow CNNs to reduce the multiplicative depth of a single network (requiring only 1 bootstrapping operation). Multiple patch CNNs are evaluated in parallel under FHE, achieving near-resolution-independent inference latency.

## Method

### Overall Architecture

The CryptoFace system consists of an offline phase (enrolling reference faces) and an online phase (verifying probe faces). CryptoFaceNet partitions the face image into $L$ patches, where each patch is processed by an independent shallow PCNN. The local features are linearly fused to obtain the global feature, which is then matched in the encrypted domain using polynomial-approximated cosine similarity.

### Key Design 1: Hybrid Shallow Patch CNN (CryptoFaceNet)

- **Function**: Efficiently extract face features under FHE-compatible constraints.
- **Mechanism**: The image $x \in \mathbb{R}^{C \times H \times W}$ is partitioned into $L = HW/P^2$ patches, where each patch $x_i \in \mathbb{R}^{C \times P \times P}$ is processed by an independent PCNN $f_{\omega_i}$. The PCNNs do not share weights and independently learn features of different facial regions. Feature fusion is simplified into a matrix multiplication $y = y'A^T + b$.
- **Design Motivation**: (1) The patch resolution is much smaller than the original image ($P \ll H$), allowing the use of shallower networks (lower multiplicative depth); (2) different PCNNs are evaluated in parallel under FHE, which does not increase inference time; (3) the fusion matrix $A$ is decomposed into block operations of $L$ square matrices, avoiding the high overhead of large matrix FHE multiplication.
- **Auxiliary Task**: A jigsaw puzzle task that predicts the original position of each patch using local features to inject positional information into the feature representations.
- **Loss Function**: $\mathcal{L} = \mathcal{L}_{\text{ArcFace}}(\omega, W, A, b) + \alpha \mathcal{L}_{\text{Jigsaw}}(\omega)$, where $\alpha = 0.005$.

### Key Design 2: Depth-Optimized Convolution Block

- **Function**: Minimize FHE multiplicative depth.
- **Mechanism**: The coefficient $a$ of the Hermite polynomial activation $ax^2 + bx + c$ (depth 2) from AESPA is fused into the convolutional weights, transforming it into $x^2 + \frac{b}{a}x + \frac{c}{a}$ (depth 1), which saves 2 multiplicative levels per block.
- **Design Motivation**: Bootstrapping is the most time-consuming operation in FHE (about 100x slower than regular multiplication), and it is required whenever the available multiplicative levels are exhausted. This depth optimization enables the entire CryptoFaceNet to require only 1 bootstrapping operation (compared to 31–43 times in MPCNN).

### Key Design 3: Distribution-Aware Polynomial $\ell_2$ Normalization

- **Function**: Approximate the $\ell_2$ normalization required for cosine similarity computation under FHE.
- **Mechanism**: A quadratic polynomial $p(t) = \beta_2 t^2 + \beta_1 t + \beta_0$ is used to approximate $q(t) = 1/\sqrt{t}$, where $t = \|y\|_2^2$. The three control points are selected as $t_1 = \text{Mean}(t) - \text{Std}(t)$, $t_2 = \text{Mean}(t)$, and $t_3 = \text{Mean}(t) + \text{Std}(t)$.
- **Design Motivation**: Traditional minimax approximation requires high-degree polynomials (high multiplicative depth), and Taylor expansion is inaccurate over wide domains. The distribution-aware control point selection allows a quadratic polynomial to achieve an approximation accuracy of $|p(t) - q(t)| \leq 2^{-10}$, consuming only 2 levels of multiplicative depth.

## Key Experimental Results

### Main Results Table (End-to-End Encrypted Face Recognition, 64×64)

| Method | Backbone | Avg Acc(%) | Latency (s) | RAM | #Boot |
|------|----------|-----------|---------|-----|-------|
| MPCNN | ResNet44 | 89.64 | 9,845 | 286G | 43 |
| MPCNN | ResNet32 | 85.60 | 7,367 | 286G | 31 |
| AutoFHE | ResNet32 | 82.69 | 4,001 | 286G | 8 |
| **CryptoFace** | CFNet4 | 89.42 | **1,364** | **269G** | **1** |

**Key Numbers**:
- **7.2×** faster than MPCNN-ResNet44 (saving 8,481 seconds) with only a 0.22% drop in accuracy.
- **2.9×** faster than AutoFHE with an accuracy gain of **+6.73%**.

### Resolution Scalability

| Resolution | Model | Avg Acc(%) | Latency (s) |
|--------|------|-----------|---------|
| 64×64 | CFNet4 | 89.42 | 1,364 |
| 96×96 | CFNet9 | 90.99 | 1,395 |
| 128×128 | CFNet16 | 91.46 | 1,446 |

When the resolution scales from 64 to 128, the accuracy improves by +2.04% while the latency only increases by 82 seconds (**near-resolution-independent**).

### Operation Latency Analysis

Bootstrapping accounts for ~70% of the execution time in MPCNN, whereas it only takes ~10% in CryptoFace. Convolution operations dominate (~63%), and parallelization overhead is <3.26%.

### Key Findings

- Outperforms MPCNN with 92.19% Rank-1 accuracy in a 1:128 closed-set retrieval task, showing a gain of +3.91%.
- Consistently outperforms baseline models in terms of AUC on challenging IJB-B/IJB-C benchmarks.
- The polynomial $\ell_2$ approximation latency is only 0.3 seconds, contributing to merely 0.02% of the overall inference time.

## Highlights & Insights

1. **First End-to-End Encrypted FR System**: From encrypted images to encrypted matching results, there is no decryption throughout the process, leaving no security loopholes.
2. **Ingenious Patch Parallelization Strategy**: Converts a high-resolution issue into multiple parallelized low-resolution problems, achieving multiple benefits: lowering network depth, enabling parallelization, and achieving resolution independence.
3. **Depth-Optimized Convolution Blocks**: Fusing the coefficient of the activation function into convolutional weights is a simple but highly effective technique.
4. **Distribution-Aware Polynomial Approximation**: Elegantly bypasses the traditional challenge of non-linear function approximation in FHE.

## Limitations & Future Work

1. The honest-but-curious (semi-honest) security model is a weaker assumption and cannot defend against malicious adversaries.
2. The patch-based design loses global contextual information, which may degrade performance under extreme occlusions or pose variations.
3. Even after optimization, it still requires ~23 minutes for online inference, which is still far from real-time applications.
4. Training is conducted using cleartext data, so safety guarantees are limited to the inference phase only.

## Related Work & Insights

- **MPCNN** (2023): An FHE convolution scheme using multiplexed convolution, whose convolution implementation is reused by CryptoFace.
- **AESPA** (2023): Low-degree Hermite polynomial activation functions, whose depth is further optimized by CryptoFace.
- **AutoFHE** (USENIX Security 2024): A method for searching FHE architectures, where CryptoFace demonstrates that manual designs can yield better performance.
- **Insights**: The divide-and-conquer patch-based approach can be generalized to other privacy-preserving computer vision tasks (such as object detection and segmentation).

## Rating

⭐⭐⭐⭐ — Outstanding system engineering and cryptographic architecture design, presenting pioneering work as the first end-to-end encrypted FR system. The 7x speedup and resolution independence are solid contributions, though practical deployment is still constrained by the inherently high latency of FHE.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] ControlFace: Harnessing Facial Parametric Control for Face Rigging](controlface_harnessing_facial_parametric_control_for_face_rigging.md)
- [\[CVPR 2025\] CRISP: Object Pose and Shape Estimation with Test-Time Adaptation](crisp_object_pose_and_shape_estimation_with_test-time_adaptation.md)
- [\[ICCV 2025\] What's Making That Sound Right Now? Video-centric Audio-Visual Localization](../../ICCV2025/human_understanding/whats_making_that_sound_right_now_video-centric_audio-visual_localization.md)
- [\[ICCV 2025\] SemGes: Semantics-aware Co-Speech Gesture Generation using Semantic Coherence and Relevance Learning](../../ICCV2025/human_understanding/semges_semantics-aware_co-speech_gesture_generation_using_semantic_coherence_and.md)
- [\[ICCV 2025\] SemTalk: Holistic Co-speech Motion Generation with Frame-level Semantic Emphasis](../../ICCV2025/human_understanding/semtalk_holistic_co-speech_motion_generation_with_frame-level_semantic_emphasis.md)

</div>

<!-- RELATED:END -->
