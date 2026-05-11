---
title: >-
  [Paper Note] NegoCollab: A Common Representation Negotiation Approach for Heterogeneous Collaborative Perception
description: >-
  [NeurIPS 2025][Multimodal VLM][Collaborative Perception] This paper proposes the NegoCollab framework, which introduces a Negotiator module to negotiate a common representation from the local representations of heterogen…
tags:
  - "NeurIPS 2025"
  - "Multimodal VLM"
  - "Collaborative Perception"
  - "Heterogeneity"
  - "Common Representation"
  - "Domain Adaptation"
  - "Autonomous Driving"
date: 2026-05-08
content_hash: 1a00b5689112eb42
---

# NegoCollab: A Common Representation Negotiation Approach for Heterogeneous Collaborative Perception

**Conference**: NeurIPS 2025
**arXiv**: [2510.27647](https://arxiv.org/abs/2510.27647)
**Code**: None
**Area**: Multimodal VLM / Collaborative Perception
**Keywords**: Collaborative Perception, Heterogeneity, Common Representation, Domain Adaptation, Autonomous Driving

## TL;DR

This paper proposes the NegoCollab framework, which introduces a Negotiator module to negotiate a common representation from the local representations of heterogeneous multimodal agents during training, effectively eliminating domain gaps between heterogeneous collaborative agents and enabling low-cost collaborative connected perception.

## Background & Motivation

**Background**: Multi-agent collaborative perception expands perception range and overcomes blind spots through feature sharing, making it a key direction for V2X communication.

**Limitations of Prior Work**: Agents may be equipped with different or fixed perception models, leading to domain gaps between intermediate features. Pairwise adaptation methods (MPDA/PnPDA) require training a large number of adapters, with training cost scaling quadratically with the number of agent types.

**Key Challenge**: Designating a single agent's representation as the common representation introduces bias, making alignment difficult for modalities that differ significantly from that agent.

**Key Insight**: The common representation should not be fixed to any single agent's representation; instead, it should be **negotiated** from the local representations of agents across modalities.

**Core Idea**: Multi-dimensional alignment (distributional + structural + pragmatic) combined with cyclic consistency to negotiate a neutral common representation from multimodal features.

## Method

### Overall Architecture

The system comprises agents from $M$ modalities and $N$ agents in total. The pipeline proceeds as: Local Representation → Sender → Common Representation (Negotiator) → Receiver → Local Representation.

### Key Designs

1. **Sender (Feature → Common Representation)**

    - **Function**: Maps local features into the common representation space.
    - **Mechanism**: A dual-module design — Recombiner (ConvNeXt architecture to enhance local features and adjust dimensions) + Aligner (fusion axis attention to capture global and local dependencies).
    - **Design Motivation**: Both dimensional and semantic alignment must be addressed simultaneously.

2. **Negotiator (Negotiating the Common Representation)**

    - **Function**: Negotiates a unified common representation from the outputs of multimodal Senders.
    - **Mechanism**: Feature Pyramid Network (FPN)-based fusion strategy $P = \bigoplus_{l,m} (u_l(P^{(m)}_l) \odot \text{norm}(P^{(m)}_l))$.
    - **Design Motivation**: Explicitly learns to generate the common representation $P$ rather than designating any single modality, eliminating bias.

3. **Receiver (Common → Local)**

    - **Function**: Transforms the common representation back into the local modality space.
    - **Mechanism**: Converter (fusion axis attention + local guidance, with Query derived from Recombiner output) + Recombiner.
    - **Design Motivation**: The common representation contains multimodal fused information and requires targeted conversion for each modality.

4. **Multi-Dimensional Alignment Loss (Section 3.2.3)**

    - **Distributional alignment**: Matches mean and standard deviation $\mathcal{L}_{uni-dis}^{(m)} = \|P^{(m)} - P\|_2^2 + \alpha\|Std(P^{(m)}) - Std(P)\|_2^2$
    - **Structural alignment**: Maintains consistency of feature similarity matrices at 9 keypoints.
    - **Pragmatic alignment**: Ensures consistent organization of foreground information $\mathcal{L}_{uni-pragma}^{(m)} = L_{focal}(\mathcal{N}(P^{(m)}), Y)$
    - **Cyclic consistency**: $\mathcal{L}_{cycle}^{(m)} = \|F^{(m)} - L^{(m)}\|_2^2$, minimizing information loss through forward-backward transformations.

### Loss & Training

Three-stage training: Stage 1 trains the Sender/Receiver with multi-dimensional alignment and cyclic consistency losses; Stage 2 jointly trains the Negotiator; Stage 3 performs end-to-end fine-tuning.

## Key Experimental Results

### Main Results (OPV2V-H Dataset)

| Method | Agent Types | AP@0.5 | AP@0.7 | Notes |
|--------|-------------|--------|--------|-------|
| No Fusion | m1, m2 | 0.482 | 0.350 | Single-agent baseline |
| MPDA (pairwise) | m1, m2 | 0.815 | 0.692 | Pairwise adaptation |
| PnPDA | m2, m4 | 0.532 | 0.331 | Poor cross-modal gap |
| **NegoCollab** | **m1, m2** | **0.872** | **0.911** | **Common representation** |
| **NegoCollab** | **m1, m3** | **0.949** | **0.854** | **New agent onboarding** |

### Ablation Study

| Alignment | AP@0.5 | Gain | Notes |
|-----------|--------|------|-------|
| Distributional only | 0.812 | Baseline | Conventional method |
| + Structural | 0.841 | +3.6% | Spatial relationships |
| + Pragmatic | 0.858 | +5.7% | Foreground consistency |
| Full three-dimensional | **0.872** | **+7.4%** | Comprehensive constraints |

### Key Findings

- Training cost reduced by 60% compared to pairwise adaptation.
- The common representation natively supports new agent onboarding without retraining the Negotiator.
- Over 40% improvement on real-world datasets V2V4Real and DAIR-V2X.

## Highlights & Insights

- **Negotiation Framework**: Breaks the limitation of "designation" by generating a more neutral and informative common representation. This paradigm is transferable to other multimodal fusion scenarios.
- **Multi-Dimensional Alignment**: Goes beyond conventional distributional alignment by incorporating structural and pragmatic constraints, forming a more complete alignment mechanism.
- **Cost-Performance Balance**: New agents can be onboarded by training only new Sender/Receiver modules — $O(M)$ complexity rather than $O(M^2)$.

## Limitations & Future Work

- Experiments are limited to LiDAR+Camera dual-modality settings; generalization to more than three modalities remains unverified.
- The paper does not discuss strategies for compressing the common representation to reduce communication bandwidth.
- The synchronization assumption across agents may not hold in real-world network environments.
- The additional computation of the Negotiator may become a bottleneck on edge devices.

## Related Work & Insights

- **vs. MPDA**: MPDA requires training adapters for each modality pair at $O(M^2)$ cost; NegoCollab requires only $O(M)$.
- **vs. PnPDA**: PnPDA performs poorly under large cross-modal gaps (AP@0.7 of only 0.331); NegoCollab's negotiation mechanism is more robust.

## Rating

- Novelty: ⭐⭐⭐⭐ — Introduction of negotiated common representation
- Experimental Thoroughness: ⭐⭐⭐⭐ — Multiple collaboration scenarios with real-world validation
- Writing Quality: ⭐⭐⭐⭐ — Clear framework presentation with rigorous formulations
- Value: ⭐⭐⭐⭐⭐ — High practical deployment value for V2X scenarios

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Mint: A Simple Test-Time Adaptation of Vision-Language Models against Common Corruptions](mint_a_simple_testtime_adaptation_of_visionlanguage_models_a.md)
- [\[NeurIPS 2025\] Struct2D: A Perception-Guided Framework for Spatial Reasoning in MLLMs](struct2d_a_perception-guided_framework_for_spatial_reasoning_in_mllms.md)
- [\[NeurIPS 2025\] On the Value of Cross-Modal Misalignment in Multimodal Representation Learning](on_the_value_of_cross-modal_misalignment_in_multimodal_representation_learning.md)
- [\[NeurIPS 2025\] SSR: Enhancing Depth Perception in VLMs via Rationale-Guided Spatial Reasoning](ssr_enhancing_depth_perception_in_vision-language_models_via_rationale-guided_sp.md)
- [\[NeurIPS 2025\] Evaluating Multimodal Large Language Models on Core Music Perception Tasks](evaluating_multimodal_large_language_models_on_core_music_perception_tasks.md)

</div>

<!-- RELATED:END -->
