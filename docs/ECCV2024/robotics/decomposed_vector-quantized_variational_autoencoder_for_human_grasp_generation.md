---
title: >-
  [Paper Note] Decomposed Vector-Quantized Variational Autoencoder for Human Grasp Generation
description: >-
  [ECCV 2024][Robotics][Grasp Generation] Proposes Decomposed VQ-VAE (DVQ-VAE), which decomposes the hand into six parts to encode them into independent codebooks, and designs a dual-stage decoding strategy (posture first, then position), achieving an approximate 14.1% relative improvement in quality index across four benchmark datasets.
tags:
  - "ECCV 2024"
  - "Robotics"
  - "Grasp Generation"
  - "VQ-VAE"
  - "Decomposed Architecture"
  - "Dual-Stage Decoding"
  - "Hand-Object Interaction"
date: 2026-05-08
content_hash: c226c87a0fc859e5
---

# Decomposed Vector-Quantized Variational Autoencoder for Human Grasp Generation

**Conference**: ECCV 2024  
**arXiv**: [2407.14062](https://arxiv.org/abs/2407.14062)  
**Code**: [GitHub](https://github.com/florasion/D-VQVAE)  
**Area**: Robotics  
**Keywords**: Grasp Generation, VQ-VAE, Decomposed Architecture, Dual-Stage Decoding, Hand-Object Interaction

## TL;DR

Proposes Decomposed VQ-VAE (DVQ-VAE), which decomposes the hand into six parts to encode them into independent codebooks, and designs a dual-stage decoding strategy (posture first, then position), achieving an approximate 14.1% relative improvement in quality index across four benchmark datasets.

## Background & Motivation

Generating realistic human hand grasping postures is crucial in robotics, human-computer interaction, and augmented reality. Existing methods face two major challenges:

**Limitations of Holistic Encoding**: Existing methods encode the hand as a holistic representation, making it difficult to finely control the interaction between each finger and the object, which leads to generated grasps that resemble "touching" rather than genuine "grasping".

**Defects of Continuous Latent Spaces**: Methods based on CVAE and GANs use a continuous latent space to model all grasp types, failing to reflect the discrete and categorical nature of real hand movements, leading to insufficient diversity and implausible grasps.

**Insufficient Generalization**: Existing methods require time-consuming test-time adaptation when facing unseen objects, significantly increasing inference costs.

The authors observe that the discrete codebook of VQ-VAE is naturally suited for modeling the discrete nature of grasp types, and the fixed relative positions of fingers make decomposed encoding a reasonable design choice.

## Method

### Overall Architecture

DVQ-VAE is based on the encoder-decoder paradigm, with core innovations in:
- **During Training**: Taking the object point cloud and hand vertices as input, the hand is decomposed into 6 parts (five fingers + palm) and encoded separately into independent codebooks, generating MANO parameters via a dual-stage decoder.
- **During Inference**: Taking only the object point cloud as input, the codebook indices of each hand part are predicted from the object codebook index via an autoregressive model (PixelCNN), and then decoded to generate the hand mesh.

The encoder uses PointNet to process point clouds, while the decoder is an MLP, maintaining 7 codebooks in total (1 for the object + 6 for hand parts).

### Key Designs

#### 1. Part-Aware Decomposed Architecture

Traditional VQ-VAE uses a single codebook, which is suitable for images (where the same codebook index can appear at different locations). However, because hand fingers have fixed relative positions, the VQ-VAE is extended into a decomposed architecture:

- Hand vertices (778 vertices) are divided into 6 segments: thumb, index finger, middle finger, ring finger, little finger, and palm.
- Each segment is independently encoded into an exclusive codebook $Z = \{Z_o, Z_1, ..., Z_N\}$.
- The object feature $z_t$ serves only as a condition for autoregression, and the object codebook is trained via unsupervised learning.
- During inference, starting from the object codebook index, the codebook indices of each hand part are generated autoregressively.

#### 2. Two Object Encoders

Unlike prior methods that use a single object encoder, this work designs:
- **Type Encoder**: Extracts $z_t$, which is used to learn grasp type clustering in the object codebook.
- **Pose Encoder**: Extracts $z_p$, helping decode the grasp position to determine the contact location between the hand and the object.

#### 3. Dual-Stage Decoding Strategy

The MANO parameters are divided into posture and position, generated sequentially:

**Stage 1 — Posture Generation**: Concatenates hand codebook features $\hat{z}_f$ with $z_t$, outputs $\hat{M}_{posture}$ (shape parameters $M_\alpha \in \mathbb{R}^{10}$ + joint rotations $M_\beta \in \mathbb{R}^{45}$) through the Posture Decoder, and introduces skeletal physical constraints:
- Extracts joint points $J$ from the reconstructed hand and calculates adjacent joint angles $\theta_i$.
- Generates refinement values through a gated network $G(\theta)$ and Transformer layers $T(\cdot)$.

**Stage 2 — Position Generation**: Applies gradient truncation (stop gradient) to the posture encoding result $z_h$, concatenates it with $z_p$, and outputs $\hat{M}_{position}$ (translation $M_\gamma \in \mathbb{R}^3$ + rotation $M_\delta \in \mathbb{R}^3$) through the Position Decoder.

### Loss & Training

The total loss consists of three components: $\mathcal{L} = \mathcal{L}_R + \mathcal{L}_E + \mathcal{L}_{contact}$

**1. Codebook Embedding Loss $\mathcal{L}_E$**: Standard VQ-VAE loss including commitment loss ($\beta=0.25$), which respectively constrains the encoders' outputs of the hand and object to approximate the codebook vectors.

**2. Reconstruction Loss $\mathcal{L}_R$**:
- Posture Loss $\mathcal{L}_{posture}$: L2 distance
- Position Loss $\mathcal{L}_{position}$: L2 distance
- Vertex Loss $\mathcal{L}_v$: L2 distance between reconstructed hand vertices and ground truth (GT).

**3. Contact Loss $\mathcal{L}_{contact}$**:
- Object-centric Contact Loss $\mathcal{L}_c$: Distance between hand and object contact points.
- Contact Map Consistency Loss $\mathcal{L}_m$: Intersection over Union (IoU) between predicted and GT contact maps.
- Penetration Loss $\mathcal{L}_p$: Penalizes hand points penetrating inside the object.

Training details: Adam optimizer, initial learning rate of 1e-4, 200 epochs, trained on the Obman dataset, using a single RTX 3090.

## Key Experimental Results

### Main Results

Trained on Obman and evaluated on four datasets (HO-3D, FPHA, GRAB, Obman), where objects in HO-3D/FPHA/GRAB are unseen in the training set.

| Dataset | Method | Contact Ratio↑ | Penetration Vol.↓ | Grasp Displ.↓ | Time (s)↓ | Quality Index↓ |
|--------|------|---------|-----------|-----------|----------|-----------|
| HO-3D | GraspCVAE | 99.60% | 7.23 | 2.78 | 0.004 | 4.12 |
| HO-3D | GraspTTA | 100% | 9.00 | 2.65 | 19.67 | 4.56 |
| HO-3D | **DVQ-VAE** | 99.50% | **5.36** | **2.75** | **0.14** | **3.54** |
| FPHA | GraspCVAE | 98.98% | 7.46 | 2.97 | 0.004 | 4.32 |
| FPHA | **DVQ-VAE** | 97.96% | **4.58** | 3.35 | **0.14** | **3.72** |
| GRAB | GraspCVAE | 97.10% | 3.54 | 2.02 | 0.004 | 2.48 |
| GRAB | **DVQ-VAE** | 98.60% | **3.18** | **2.13** | **0.15** | **2.45** |

Inference time is reduced by 99.8% compared to ContactGen, and by 99.3% compared to GraspTTA.

### Ablation Study

| Variant | Penetration Vol.↓ | Grasp Displ.↓ | Quality Index↓ |
|------|-----------|-----------|-----------|
| VQ-VAE (Baseline) | 6.67 | 7.21 | 7.05 |
| DVQ-VAE (Decomposed Architecture Only) | 10.88 | 4.98 | 6.76 |
| VQ-VAE + Dual-Stage | 4.44 | 3.61 | 3.86 |
| DVQ-VAE + Dual-Stage (Single Encoder) | 11.20 | 4.57 | 6.57 |
| DVQ-VAE + Dual-Stage (Reverse Order) | 7.56 | 2.93 | 4.32 |
| **DVQ-VAE + Dual-Stage (Full)** | **5.36** | **2.75** | **3.54** |

### Key Findings

- The decomposed architecture increases cluster scale by 151%, significantly enhancing grasp diversity.
- The dual-stage strategy yields a relative improvement in quality index of 45.2% (DVQ-VAE) and 47.6% (VQ-VAE).
- Objects are clustered into 21 grasp types in the codebook, with the index and ring fingers exhibiting higher degrees of freedom.
- The model still generates plausible grasps even when object point clouds are 50% or 90% occluded.
- Achieves the highest mean score of 3.36 in human evaluation (vs ContactGen 3.25, GraspTTA 3.23).

## Highlights & Insights

1. **Rationality of Discretized Modeling**: Grasp types are inherently discrete; hence, the discrete codebook of VQ-VAE is more suitable for modeling than the continuous Gaussian distribution in CVAEs.
2. **Intuition of Decomposed Encoding**: The relative positions of fingers are fixed. Unlike image pixels that can reuse the same codebook, establishing independent codebooks for different hand parts is a natural choice.
3. **Rational Design of Posture-First, Position-Second**: Posture parameters are numerous (55 dimensions) and more fundamental, whereas position parameters are fewer (6 dimensions) and dependent on posture. Sequential generation thus aligns with intuition.
4. **Introduction of the Quality Index**: Balances penetration and displacement, preventing misleading results from a single metric.

## Limitations & Future Work

- May generate grasps with insufficient contact for objects with complex geometries (e.g., irregular shapes).
- Could consider using Signed Distance Fields (SDF) to enhance object representation.
- Currently supports only single-hand grasping, without considering bimanual collaboration scenarios.
- The codebook size is fixed; exploring dynamic or hierarchical codebooks remains future work.

## Related Work & Insights

- **Success of VQ-VAE in Motion Generation**: Inspired by Pi et al.'s work that decomposes the human body into 5 parts for motion generation encoding, this work extends it to a 6-part encoding of the hand.
- **MANO Parametric Hand Model**: Provides the foundation for representing the hand with differentiable parameters.
- **Test-Time Adaptation in ContactNet**: Although effective, it is time-consuming. The dual-stage strategy avoids optimization during inference.

## Rating

- Novelty: ⭐⭐⭐⭐ — First to propose the VQ-VAE and decomposed architecture in grasp generation.
- Technical Depth: ⭐⭐⭐⭐ — Well-designed components with thorough ablation validation.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — 4 datasets, human evaluation, robustness test, and detailed ablation.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure and rich illustrations.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] SemGrasp: Semantic Grasp Generation via Language Aligned Discretization](semgrasp_semantic_grasp_generation_via_language_aligned_discretization.md)
- [\[ECCV 2024\] An Economic Framework for 6-DoF Grasp Detection](an_economic_framework_for_6-dof_grasp_detection.md)
- [\[ICCV 2025\] EvolvingGrasp: Evolutionary Grasp Generation via Efficient Preference Alignment](../../ICCV2025/robotics/evolvinggrasp_evolutionary_grasp_generation_via_efficient_preference_alignment.md)
- [\[CVPR 2026\] DextER: Language-driven Dexterous Grasp Generation with Embodied Reasoning](../../CVPR2026/robotics/dexter_language-driven_dexterous_grasp_generation_with_embodied_reasoning.md)
- [\[CVPR 2026\] Towards Human-Like Robot Handwriting via Contour-Aware Generation](../../CVPR2026/robotics/towards_human-like_robot_handwriting_via_contour-aware_generation.md)

</div>

<!-- RELATED:END -->
