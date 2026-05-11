---
title: >-
  [Paper Note] HEEGNet: Hyperbolic Embeddings for EEG
description: >-
  [ICLR 2026][EEG] This work presents the first systematic empirical validation that EEG data exhibits hyperbolicity (hierarchical structure), and proposes HEEGNet…
tags:
  - "ICLR 2026"
  - "EEG"
  - "hyperbolic space"
  - "domain adaptation"
  - "hierarchy"
  - "brain-computer interface"
date: 2026-05-08
content_hash: f8e0fafda71dfd15
---

# HEEGNet: Hyperbolic Embeddings for EEG

**Conference**: ICLR 2026  
**arXiv**: [2601.03322](https://arxiv.org/abs/2601.03322)  
**Code**: [GitHub](https://github.com/fightlesliefigt/HEEGNet)  
**Area**: Brain-Computer Interface / Geometric Deep Learning  
**Keywords**: EEG, hyperbolic space, domain adaptation, hierarchy, brain-computer interface

## TL;DR
This work presents the first systematic empirical validation that EEG data exhibits hyperbolicity (hierarchical structure), and proposes HEEGNet, a hybrid hyperbolic network architecture. The model combines a Euclidean encoder for spatiotemporal-spectral feature extraction with a hyperbolic encoder for capturing hierarchical relationships, augmented by a novel coarse-to-fine domain adaptation strategy (DSMDBN). HEEGNet achieves state-of-the-art performance across multiple cross-domain tasks spanning visual evoked potentials, emotion recognition, and intracranial EEG.

## Background & Motivation

### State of the Field

**Background**: EEG-based BCIs suffer from poor generalization due to distribution shifts across subjects and sessions. Domain adaptation methods based on moment alignment represent the current SOTA, yet fail under large domain discrepancies. EEG decoding has been conducted almost exclusively in Euclidean embedding spaces.

**Limitations of Prior Work**: (1) Cognitive processes such as visual processing and emotion regulation exhibit hierarchical organization in the brain, yet Euclidean space is inefficient for representing hierarchical data — the circumference of a circle grows linearly whereas the number of tree nodes grows exponentially; (2) moment alignment alone cannot guarantee positive transfer, particularly under large domain shifts.

**Key Challenge**: The hierarchical structure of EEG features demands exponential representational capacity, which Euclidean space cannot provide; however, hyperbolic neural networks have not yet been systematically explored for EEG.

**Key Insight**: Prior analysis reveals that EEG data exhibits hyperbolicity (low $\delta_{rel}$), and replacing the Euclidean MLR with a hyperbolic MLR alone improves cross-domain performance — demonstrating that hyperbolic embeddings genuinely benefit EEG generalization.

**Core Idea**: Exploit hyperbolic space to capture the hierarchical structure of EEG, combined with a two-stage domain adaptation strategy (moment alignment → distribution alignment) to achieve cross-domain generalization.

## Method

### Overall Architecture
HEEGNet = Euclidean encoder (temporal → spatial → temporal convolution) → projection into hyperbolic space → hyperbolic convolutional layers → DSMDBN domain adaptation → hyperbolic MLR classifier.

### Key Designs

1. **Hybrid Euclidean–Hyperbolic Architecture**:

    - **Function**: Euclidean convolutions extract spectral-spatial-temporal features; after projection, hyperbolic convolutions refine hierarchical relationships.
    - **Mechanism**: Three EEGNet-style convolutional layers → ProjX projection onto the Lorentz model $\mathbb{L}_K^n$ → hyperbolic point-wise convolution.
    - **Design Motivation**: Euclidean convolutions excel at signal processing and carry neurophysiological interpretability, while hyperbolic space is well-suited for representing hierarchical relations. The hybrid design leverages the strengths of both.

2. **DSMDBN (Two-Stage Domain Adaptation)**:

    - **Stage 1 — DSMDBN(1)**: Riemannian batch normalization performs domain-specific moment alignment; centering in hyperbolic space is implemented via gyro-subtraction and scaling via gyro-multiplication.
    - **Stage 2 — DSMDBN(2)**: The HHSW divergence is minimized to align each source domain distribution to a canonical hyperbolic Gaussian $\mathcal{N}(\bar{0}, 1)$.
    - **Design Motivation**: Moment alignment alone is insufficient under large domain shifts; adding distributional alignment provides theoretical guarantees. The coarse-to-fine ordering — moment alignment followed by distributional alignment — progressively closes the domain gap.

3. **Lorentz Model Operations**:

    - Gyroaddition, gyromultiplication, and gyroinverse in the Lorentz model.
    - Fréchet mean and variance defined on the Lorentz model.
    - Hyperbolic MLR performs classification via the hyperbolic distance from a point to a hyperplane.

### Loss & Training
- Classification loss + HHSW distributional alignment loss.
- Domain-specific momentum batch normalization: momentum is updated with decay during training and fixed during inference.
- Riemannian Adam optimizer.

## Key Experimental Results

### Preliminary Study

| Dataset | $\delta_{rel}$ (raw EEG) | $\delta_{rel}$ (embedding layer) | Notes |
|---------|--------------------------|----------------------------------|-------|
| Nakanishi | low | low | Visual |
| Wang | low | low | Visual |
| Seed | low | low | Emotion |
| Faced | low | low | Emotion |
| Boran | low | low | Intracranial |

→ All datasets exhibit low $\delta_{rel}$, confirming the hyperbolicity of EEG data.

### Main Results
Cross-subject / cross-session adaptation:

| Method | Visual EEG | Emotion EEG | Intracranial EEG | Average |
|--------|-----------|-------------|------------------|---------|
| EEGNet | baseline | baseline | baseline | baseline |
| EEGNet+HMLR | ↑ | ↑ | ↑ | consistent gain |
| HEEGNet | **SOTA** | **SOTA** | **SOTA** | **best overall** |

### Key Findings
- Replacing the Euclidean MLR with a hyperbolic MLR alone yields improvements across all datasets, confirming that hyperbolic geometry is better suited for EEG.
- t-SNE visualizations show markedly superior class separation in hyperbolic embeddings compared to Euclidean ones.
- The two-stage DSMDBN strategy yields significant gains over moment alignment alone.
- Performance improvements are also observed on motor imagery datasets (where hierarchical structure was not explicitly reported), suggesting the possible presence of unidentified hierarchical structure.

## Highlights & Insights
- **First systematic validation of EEG hyperbolicity**: Quantitative $\delta_{rel}$ analysis validated across multiple datasets provides a solid empirical foundation for this research direction.
- **Rationale for the hybrid architecture**: Rather than adopting a purely hyperbolic network (which would discard signal-processing priors), the design first extracts meaningful features with Euclidean convolutions and then maps them to hyperbolic space — a design philosophy transferable to other domains.
- **Coarse-to-fine nature of DSMDBN**: Moment alignment followed by distributional alignment constitutes a natural two-step strategy: the former brings means and scales closer, while the latter aligns overall distributional shape.

## Limitations & Future Work
- Hyperbolic operations (exponential/logarithmic maps) incur greater computational overhead than their Euclidean counterparts, posing challenges for real-time BCI applications.
- The curvature $K$ is treated as a hyperparameter requiring manual tuning; adaptive curvature learning may be preferable.
- HHSW may require a large number of projection directions to achieve accurate estimation in high dimensions.
- Varying electrode counts across subjects in intracranial EEG datasets constrain cross-subject experimental setups.

## Related Work & Insights
- **vs. EEGNet**: HEEGNet extends EEGNet with hyperbolic layers and DSMDBN, achieving comprehensive improvements.
- **vs. Chang et al. (hyperbolic EEG)**: Their work focuses solely on contrastive pre-training, whereas HEEGNet presents a complete architecture design coupled with a domain adaptation framework.
- **vs. Riemannian methods (e.g., SPDNet)**: SPDNet operates on the covariance matrix manifold, while HEEGNet operates in hyperbolic space, targeting distinct geometric structures.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ First systematic demonstration of EEG hyperbolicity and first hybrid hyperbolic architecture for EEG.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Comprehensive coverage from preliminary studies to multi-dataset, multi-task, and ablation experiments.
- **Writing Quality**: ⭐⭐⭐⭐ Background is well-introduced and methodology is clearly described.
- **Value**: ⭐⭐⭐⭐ Introduces a novel geometric perspective to EEG decoding and opens a new direction in hyperbolic EEG research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] HypeVPR: Exploring Hyperbolic Space for Perspective to Equirectangular Visual Place Recognition](../../CVPR2026/others/hypevpr_exploring_hyperbolic_space_for_perspective_to_equirectangular_visual_pla.md)
- [\[AAAI 2026\] Shrinking the Teacher: An Adaptive Teaching Paradigm for Asymmetric EEG-Vision Alignment](../../AAAI2026/others/shrinking_the_teacher_an_adaptive_teaching_paradigm_for_asymmetric_eeg-vision_al.md)
- [\[ICCV 2025\] Learning Visual Hierarchies in Hyperbolic Space for Image Retrieval](../../ICCV2025/others/learning_visual_hierarchies_in_hyperbolic_space_for_image_retrieval.md)
- [\[AAAI 2026\] CAT-Net: A Cross-Attention Tone Network for Cross-Subject EEG-EMG Fusion Tone Decoding](../../AAAI2026/others/cat-net_a_cross-attention_tone_network_for_cross-subject_eeg-emg_fusion_tone_dec.md)
- [\[AAAI 2026\] CellStream: Dynamical Optimal Transport Informed Embeddings for Reconstructing Cellular Trajectories from Snapshots Data](../../AAAI2026/others/cellstream_dynamical_optimal_transport_informed_embeddings_for_reconstructing_ce.md)

</div>

<!-- RELATED:END -->
