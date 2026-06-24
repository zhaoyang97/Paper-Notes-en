---
title: >-
  [Paper Note] Representation Learning for Spatiotemporal Physical Systems
description: >-
  [CVPR2025][Self-Supervised Learning][Spatiotemporal Physical Systems] This work systematically evaluates the capability of general self-supervised learning methods to learn physically meaningful representations in spatiotemporal physical systems. The evaluation reveals that JEPA, which performs predictions in the latent space, significantly outperforms pixel-level reconstruction methods (MAE) and autoregressive models, closely approaching the performance of the domain-specifi…
tags:
  - "CVPR2025"
  - "Self-Supervised Learning"
  - "Spatiotemporal Physical Systems"
  - "JEPA"
  - "Representation Learning"
  - "Parameter Estimation"
  - "Masked Autoencoder"
  - "Physical Modeling"
date: 2026-05-08
content_hash: 6bb09bd3e1b29e7a
---

# Representation Learning for Spatiotemporal Physical Systems

**Conference**: CVPR2025  
**arXiv**: [2603.13227](https://arxiv.org/abs/2603.13227)  
**Code**: [GitHub](https://github.com/helenqu/physical-representation-learning)  
**Area**: Self-Supervised Learning  
**Keywords**: Self-Supervised Learning, Spatiotemporal Physical Systems, JEPA, Representation Learning, Parameter Estimation, Masked Autoencoder, Physical Modeling

## TL;DR
This work systematically evaluates the capability of general self-supervised learning methods to learn physically meaningful representations in spatiotemporal physical systems. The evaluation reveals that JEPA, which performs predictions in the latent space, significantly outperforms pixel-level reconstruction methods (MAE) and autoregressive models, closely approaching the performance of the domain-specific physical modeling method DISCO.

## Background & Motivation
- The application of machine learning to spatiotemporal physical systems primarily focuses on **frame-by-frame prediction** (surrogate modeling), aiming to learn accurate simulators of system evolution.
- However, training these simulators is expensive, and they suffer from error accumulation during autoregressive rollout.
- A more critical question: For downstream scientific tasks (such as estimating physical parameters of a system), which learning paradigm is most effective at extracting physically relevant information?
- The capability of parameter estimation serves as a quantifiable metric for a model's "understanding" of the underlying physics, as these parameters govern the temporal evolution of the system.
- Prior work has rarely focused on the differences in physical representation quality across different learning paradigms.

## Core Problem
Can general self-supervised learning methods effectively learn physically meaningful spatiotemporal representations? Latent-space prediction vs. pixel-level reconstruction: which paradigm is more suitable for extracting physical information?

## Method

### 1. JEPA (Joint Embedding Predictive Architecture)
- Temporal JEPA based on VICReg loss: Given a sample $x_{0:T}$ of $T$ steps, adjacent $k$ frames are grouped together, encoded using an encoder $f$, and a predictor $g$ is used to predict the representation of the next group of $k$ frames in the latent space.
- Loss function = invariance term (MSE) + variance regularization + covariance regularization, designed to prevent mode collapse.
- Encoder: 3D CNN (ConvNeXt-style), outputting $l/16 \times w/16 \times 128$.
- Predictor: CNN with inverted bottlenecks.
- **Key Design**: Making predictions in the latent space rather than the pixel space, which avoids learning low-level details.

### 2. Masked Autoencoder (VideoMAE)
- Standard VideoMAE ViT-tiny/16 architecture.
- Temporal tube masking: all frames use the same spatial mask.
- Minimizing the pixel-level reconstruction error in the masked regions.
- Encoder outputs $l/16 \times w/16 \times t/2 \times 384$.

### 3. Physical Modeling Baselines
- **DISCO**: In-context operator learning, which infers trajectory-specific evolution rules from a short context window, with an embedding dimension of $1 \times 384$.
- **MPP**: An autoregressive foundation model that performs pixel-level frame-by-frame prediction, utilizing the released pretrained AViT-tiny weights.

### 4. Fine-Tuning and Evaluation
- The encoder is frozen, and an attentive probe is trained on top of it (100 epochs).
- Evaluation task: Physical parameter regression (MSE loss).
- As MPP was not pretrained on the target datasets, end-to-end fine-tuning is used instead.

### Evaluated Physical Systems (from The Well dataset)
1. **Active Matter**: Collective dynamics of active rod-like particles in a Stokes fluid. Parameters: $\alpha$ (active dipole strength), $\zeta$ (particle alignment strength).
2. **Rayleigh-Bénard Convection**: Horizontal fluid layers heated from below and cooled from above, forming convection cells. Parameters: Rayleigh number $\nu$, Prandtl number $\kappa$.
3. **Shear Flow**: The boundary between fluid layers flowing parallelly at different velocities. Parameters: Reynolds number, Schmidt number.

## Key Experimental Results

### Physical Parameter Estimation MSE ($\downarrow$)

| Method | Active Matter | Shear Flow | Rayleigh-Bénard |
|------|---------|--------|-----------------|
| **JEPA** | **0.079** | **0.38** | **0.13** |
| VideoMAE | 0.160 | 0.67 | 0.18 |
| DISCO | 0.057 | 0.13 | 0.01 |
| MPP (End-to-End Fine-Tuning) | 0.230 | 0.59 | 0.08 |

- JEPA vs. VideoMAE: Gains of 51% in Active Matter, 43% in Shear Flow, and 28% in Rayleigh-Bénard.
- JEPA closely approaches DISCO (a domain-specific physical modeling method), with a gap of only 0.022 on Active Matter.

### Data Efficiency (Shear Flow)

| Fine-tuning Data Ratio | JEPA | VideoMAE |
|-------------|------|---------|
| 10% | 0.57 | 0.98 |
| 50% | 0.40 | 0.75 |
| 100% | 0.38 | 0.67 |

- JEPA using only 10% of the data (0.57) already outperforms VideoMAE using 100% of the data (0.67).

## Highlights & Insights
1. **Latent-Space Prediction $\gg$ Pixel-Level Reconstruction**: The core finding is simple yet powerful; JEPA significantly outperforms VideoMAE across all systems.
2. **Not All Physical Modeling Methods are Superior**: MPP (an autoregressive model) is even weaker than general JEPA, which aligns with findings in NLP where autoregressive models underperform encoder-based models on non-generative tasks.
3. **Superb Data Efficiency**: The fine-tuning data scaling behavior of JEPA is superior to that of VideoMAE.
4. **Both DISCO and JEPA are Latent-Space Prediction Models**: A shared characteristic between the two top-performing methods is operating in the latent space, revealing a core design principle.
5. **Code Available**: Complete experimental code is provided.

## Limitations & Future Work
- Evaluation is limited to only three physical systems (all of which are 2D PDE systems).
- The JEPA encoder utilizes a 3D CNN; a ViT-based JEPA architecture was not explored.
- Downstream tasks are confined to parameter estimation, with other scientific tasks like classification (e.g., laminar vs. turbulent flow) remaining unexplored.
- DISCO significantly outperforms JEPA on Rayleigh-Bénard ($MSE = 0.01$ vs. $0.13$), indicating that physical inductive biases remain critical in certain systems.
- Analysis of what physical information is actually captured by the JEPA-learned representations is not provided.

## Related Work & Insights
- vs. **VideoMAE**: JEPA outperforms it across the board, proving that latent-space prediction is superior to pixel-level reconstruction.
- vs. **MPP (Autoregressive Foundation Model)**: The general JEPA even outperforms end-to-end fine-tuned MPP, suggesting that the next-frame prediction paradigm is not necessarily optimal for representation learning.
- vs. **DISCO**: Domain-specific physical modeling methods still hold an advantage (particularly in Rayleigh-Bénard), but the gap for general JEPA is relatively small.
- vs. **V-JEPA/I-JEPA**: The JEPA variant in this work is specifically designed for physical systems, adopting a 3D CNN rather than a ViT.

## Insights & Connections
- Crucial takeaway for the Scientific ML community: Pursuing the most accurate surrogate model may not be necessary; the latent-space prediction paradigm might be superior for downstream tasks.
- Consistent with LeCun’s JEPA philosophy: Abstract representations are more effective than pixel-level reconstruction.
- The trade-off between autoregressive and encoder-based models shows similar trends in both AI for Physics and NLP.
- Offers a new direction for the design of foundation models in physical systems: potentially utilizing JEPA rather than an autoregressive paradigm.

## Rating
- Novelty: ⭐⭐⭐⭐ (First systematic comparison of self-supervised paradigms for parameter estimation in physical systems)
- Experimental Thoroughness: ⭐⭐⭐ (3 systems × 4 methods, though downstream tasks are relatively limited)
- Writing Quality: ⭐⭐⭐⭐⭐ (Concise and clear, with key findings presented straightforwardly)
- Value: ⭐⭐⭐⭐ (Strong guiding significance for the Scientific ML community)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] QAEncoder: Towards Aligned Representation Learning in Question Answering Systems](../../ACL2025/self_supervised/qaencoder_aligned_representation.md)
- [\[CVPR 2025\] CheXWorld: Image World Modeling for Radiograph Representation Learning](chexworld_exploring_image_world_modeling_for_radiograph_representation_learning.md)
- [\[CVPR 2025\] Spectral State Space Model for Rotation-Invariant Visual Representation Learning](spectral_state_space_model_for_rotation-invariant_visual_representation_learning.md)
- [\[ACL 2025\] Magnet: Augmenting Generative Decoders with Representation Learning and Infilling Capabilities](../../ACL2025/self_supervised/magnet_augmenting_generative_decoders_with_representation_learning_and_infilling.md)
- [\[ICML 2025\] Generalization Analysis for Supervised Contrastive Representation Learning under Non-IID Settings](../../ICML2025/self_supervised/generalization_analysis_for_supervised_contrastive_representation_learning_under.md)

</div>

<!-- RELATED:END -->
