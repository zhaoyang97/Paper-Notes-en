---
title: >-
  [Paper Note] Active Inference for Micro-Gesture Recognition: EFE-Guided Temporal Sampling and Adaptive Learning
description: >-
  [CVPR 2026][Medical Imaging][Micro-Gesture Recognition] This paper proposes the UAAI framework, which for the first time introduces Active Inference into micro-gesture recognition. By combining EFE-guided temporal frame…
tags:
  - "CVPR 2026"
  - "Medical Imaging"
  - "Micro-Gesture Recognition"
  - "Active Inference"
  - "Expected Free Energy"
  - "POMDP"
  - "Uncertainty-Aware Augmentation"
date: 2026-05-08
content_hash: f18cd4295086a4d2
---

# Active Inference for Micro-Gesture Recognition: EFE-Guided Temporal Sampling and Adaptive Learning

**Conference**: CVPR 2026
**arXiv**: [2603.07559](https://arxiv.org/abs/2603.07559)
**Authors**: Weijia Feng et al. (Tianjin Normal University, Shenzhen University, Zhejiang University, Tianjin University)
**Area**: Medical Imaging
**Keywords**: Micro-Gesture Recognition, Active Inference, Expected Free Energy, POMDP, Uncertainty-Aware Augmentation

## TL;DR

This paper proposes the UAAI framework, which for the first time introduces Active Inference into micro-gesture recognition. By combining EFE-guided temporal frame selection, spatial attention, and UMIX uncertainty-aware augmentation, UAAI achieves 63.47% on the SMG dataset (RGB modality), substantially outperforming conventional RGB-based methods.

## Background & Motivation

Micro-gestures refer to subtle, unconscious body movements produced during communication, such as light finger tapping or slight head tilting. Compared to conventional gesture recognition, micro-gesture recognition presents unique challenges:

**Extremely short duration**: Typically <0.5 seconds, occupying a negligible proportion of long videos.

**Minimal amplitude**: Motion scale is far smaller than everyday gestures, making signals easily overwhelmed by noise.

**High inter-subject variability**: The same micro-gesture category manifests very differently across individuals.

**Spatiotemporal sparsity**: Informative content exists only in a small number of specific frames and local regions.

Existing methods (e.g., C3D, TSM, SlowFast) are designed for conventional action recognition and lack targeted modeling for such fleeting signals. The core problem is: **how to precisely capture these transient, weak signals along both temporal and spatial dimensions?**

Active Inference is a cognitive framework grounded in the Bayesian Brain hypothesis, where an agent selects actions by minimizing Expected Free Energy (EFE)—simultaneously pursuing epistemic value (information gain) and pragmatic value (goal achievement). This naturally aligns with the demand for actively searching key frames and key regions in micro-gesture recognition.

## Method

### Overall Architecture

The UAAI (Uncertainty-Aware Active Inference) framework consists of three core modules:

1. **EFE-Guided Temporal Selection**: POMDP-based temporal frame selection.
2. **EFE-Guided Spatial Selection**: EFE decomposition-driven spatial attention.
3. **UMIX**: Uncertainty-aware data augmentation.

### Key Designs

**Module 1: EFE-Guided Temporal Frame Selection**

Frame selection is formulated as a Partially Observable Markov Decision Process (POMDP):

- **State**: The latent semantic state of the video (current belief over micro-gesture categories).
- **Observation**: Visual features of the current frame.
- **Action**: Selection of the next frame to observe.

Core mechanism:
- At each timestep $t$, the agent maintains a posterior distribution $q(s_t)$ over micro-gesture categories (Bayesian belief).
- An MLP-parameterized likelihood matrix $A_{a_t}$ maps observations to state updates.
- The EFE for each candidate action is computed as: $G(a) = \underbrace{-D_{KL}[q(o|a) \| \tilde{p}(o)]}_{\text{pragmatic value}} - \underbrace{E_{q(o|a)}[H[q(s|o,a)]]}_{\text{epistemic value}}$
- The action minimizing EFE (i.e., the most informative next frame) is selected.
- Bayesian update: upon observing a new frame, the posterior belief is updated via the likelihood matrix.

**Module 2: EFE-Guided Spatial Attention**

- EFE is decomposed along the spatial dimension to yield an informativeness score for each spatial location.
- A learnable spatial weight mask is generated: $M = \sigma(\text{Conv}([F_{\text{avg}}; F_{\text{max}}]))$
- where $F_{\text{avg}}$ and $F_{\text{max}}$ are channel-wise average-pooled and max-pooled features, respectively.
- The spatial mask enhances local regions where micro-gestures occur while suppressing irrelevant background.

**Module 3: UMIX Uncertainty-Aware Augmentation**

- Per-sample predictive uncertainty is estimated via Monte Carlo Dropout.
- High-uncertainty samples → stronger augmentation + reduced loss weight (to avoid noisy gradients).
- Low-uncertainty samples → weaker augmentation + normal loss weight (to avoid over-perturbing confident samples).
- The mixing ratio $\lambda$ is adaptively adjusted based on uncertainty: $\lambda = \text{Beta}(\alpha(u), \beta(u))$

### Loss & Training

The total loss is based on minimizing Variational Free Energy (VFE):

$$L = L_{\text{accuracy}} + \beta \cdot L_{\text{complexity}}$$

- **$L_{\text{accuracy}}$**: Cross-entropy classification loss to ensure correct predictions.
- **$L_{\text{complexity}}$**: KL divergence between posterior and prior, preventing overfitting and encouraging compact representations.

Training strategy:
- Alternating optimization: the base feature extractor is first warmed up, followed by end-to-end training of the spatiotemporal selection modules.
- UMIX computes per-sample uncertainty online within each mini-batch to adjust augmentation.
- EFE computation is made differentiable via reparameterization.

## Key Experimental Results

### Main Results (SMG Dataset, RGB Modality)

| Method | Backbone | Top-1 Acc (%) |
|--------|----------|---------------|
| C3D | 3D CNN | 45.90 |
| I3D | Inception | 50.23 |
| TSM | ResNet-50 | 58.69 |
| SlowFast | ResNet-50 | 56.42 |
| Video Swin-T | Swin | 59.14 |
| TimeSformer | ViT | 57.83 |
| **UAAI (Ours)** | **ResNet-50** | **63.47** |
| MS-G3D (Skeleton) | GCN | 64.75 |

UAAI achieves 63.47% on the RGB modality, substantially outperforming all other RGB-based methods and approaching MS-G3D, which requires skeleton annotations.

### Ablation Study

| Configuration | Top-1 Acc (%) | Change |
|---------------|---------------|--------|
| Full UAAI | 63.47 | — |
| w/o EFE Temporal | 59.83 | -3.64 |
| w/o EFE Spatial | 61.02 | -2.45 |
| w/o UMIX | 61.54 | -1.93 |
| w/o EFE Temporal + Spatial | 57.26 | -6.21 |
| Random Temporal Sampling | 56.91 | -6.56 |
| Uniform Temporal Sampling | 58.12 | -5.35 |

### Key Findings

1. **EFE temporal selection contributes most** (−3.64%), confirming that identifying key frames is the central bottleneck in micro-gesture recognition.
2. **Spatial attention is complementary** (−2.45%), further focusing on body regions where micro-movements occur.
3. **UMIX stabilizes training** (−1.93%), with uncertainty-aware augmentation effectively combating noise and ambiguity in micro-gesture data.
4. **Removing both temporal and spatial modules causes a sharp drop** (−6.21%), indicating synergistic rather than additive interaction.
5. **EFE significantly outperforms random/uniform sampling** (by 6.56/5.35 points), demonstrating the clear advantage of active inference-based frame selection over heuristic approaches.

## Highlights & Insights

1. **Cross-disciplinary fusion of cognitive science and computer vision**: Active Inference originates from the Free Energy Principle in neuroscience; its introduction to micro-gesture recognition is theoretically elegant.
2. **POMDP formulation of frame selection**: Unlike attention- or sampling-based post-hoc methods, EFE is forward-looking—it selects frames anticipated to maximally reduce uncertainty.
3. **Uncertainty as a unified guiding signal**: From epistemic value in EFE to adaptive augmentation in UMIX, uncertainty permeates the entire framework.
4. **RGB approaching skeleton performance**: The pure RGB method achieves 63.47%, nearly matching skeleton-based MS-G3D (64.75%), thereby avoiding the overhead of skeleton estimation in deployment.

## Limitations & Future Work

1. **Computational efficiency**: EFE computation requires a forward pass for each frame-action pair; complexity may be substantial when the number of timesteps is large.
2. **POMDP approximation**: True EFE requires integrating over future trajectories; the implementation uses a single-step approximation, which may miss long-range dependencies.
3. **Limited dataset coverage**: Validation is primarily on the SMG dataset, with no cross-dataset generalization experiments (e.g., iMiGUE).
4. **Area classification concern**: Micro-gesture recognition more strictly belongs to behavior understanding rather than medical imaging, though the methodology is transferable to medical video analysis.
5. **No multimodal fusion**: Skeleton modality is not incorporated; combining modalities could yield further improvements.
6. **Uncertainty estimation overhead**: UMIX relies on multiple MC Dropout forward passes, increasing training time.

## Related Work & Insights

- **Free Energy Principle (Friston)**: The theoretical foundation of Active Inference; agents perceive and act by minimizing free energy.
- **AdaFrame / SCSampler**: Frame sampling methods for video understanding, but based on reinforcement learning rather than active inference.
- **MS-G3D**: A strong skeleton-based GCN baseline for micro-gesture recognition, nearly matched by UAAI's RGB approach.
- **Mixup / CutMix**: Classic data augmentation techniques; UMIX's uncertainty-adaptive extension is a meaningful contribution.
- **Inspiration**: The EFE-guided spatiotemporal selection framework is generalizable to other tasks requiring precise spatiotemporal localization, such as micro-expression analysis, pain detection, and surgical step recognition.

## Rating

| Dimension | Score (1–5) | Notes |
|-----------|-------------|-------|
| Novelty | 4.5 | First application of active inference to micro-gesture recognition; strong cross-disciplinary innovation. |
| Technical Depth | 4 | Coherent system integrating POMDP, EFE, and Bayesian updates. |
| Experimental Thoroughness | 3.5 | Thorough ablations, but limited to a single dataset. |
| Value | 3.5 | Practical RGB-only deployment, but narrow application scope. |
| Writing Quality | 4 | Balances formal derivations with intuitive explanations. |
| **Overall** | **3.9** | Unique cognitive science perspective; framework holds general transferability. |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] EI: Early Intervention for Multimodal Imaging based Disease Recognition](ei_early_intervention_for_multimodal_imaging_based_disease_recognition.md)
- [\[CVPR 2026\] Cross-Slice Knowledge Transfer via Masked Multi-Modal Heterogeneous Graph Contrastive Learning for Spatial Gene Expression Inference](cross-slice_knowledge_transfer_via_masked_multi-modal_heterogeneous_graph_contra.md)
- [\[CVPR 2026\] CARE: A Molecular-Guided Foundation Model with Adaptive Region Modeling for Whole Slide Image Analysis](care_a_molecular-guided_foundation_model_with_adaptive_region_modeling_for_whole.md)
- [\[CVPR 2026\] T-Gated Adapter: A Lightweight Temporal Adapter for Vision-Language Medical Segmentation](t-gated_adapter_a_lightweight_temporal_adapter_for_vision-language_medical_segme.md)
- [\[ICLR 2026\] DistMLIP: A Distributed Inference Platform for Machine Learning Interatomic Potentials](../../ICLR2026/medical_imaging/distmlip_a_distributed_inference_platform_for_machine_learning_interatomic_poten.md)

</div>

<!-- RELATED:END -->
