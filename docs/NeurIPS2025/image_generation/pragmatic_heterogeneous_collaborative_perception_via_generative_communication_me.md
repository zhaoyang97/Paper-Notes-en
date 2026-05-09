---
title: >-
  [Paper Note] Pragmatic Heterogeneous Collaborative Perception via Generative Communication Mechanism
description: >-
  [NeurIPS 2025][Image Generation][Heterogeneous Collaborative Perception] This paper proposes GenComm — a generative communication mechanism for heterogeneous multi-agent collaborative perception. By extracting spatial messages and employing a conditional diffusion model, the ego agent locally generates aligned collaborator features without modifying any existing network, enabling new heterogeneous agents to be onboarded at minimal cost.
tags:
  - NeurIPS 2025
  - Image Generation
  - Heterogeneous Collaborative Perception
  - Generative Communication
  - Conditional Diffusion Model
  - BEV Feature Generation
  - Scalability
date: 2026-05-08
content_hash: f41f0e66ab90d8d6
---

# Pragmatic Heterogeneous Collaborative Perception via Generative Communication Mechanism

**Conference**: NeurIPS 2025
**arXiv**: [2510.19618](https://arxiv.org/abs/2510.19618)
**Code**: [Available](https://github.com/jeffreychou777/GenComm)
**Area**: Image Generation
**Keywords**: Heterogeneous Collaborative Perception, Generative Communication, Conditional Diffusion Model, BEV Feature Generation, Scalability

## TL;DR

This paper proposes GenComm — a generative communication mechanism for heterogeneous multi-agent collaborative perception. By extracting spatial messages and employing a conditional diffusion model, the ego agent locally generates aligned collaborator features without modifying any existing network, enabling new heterogeneous agents to be onboarded at minimal cost.

## Background & Motivation

Multi-agent collaborative perception enhances individual agent perception through information sharing, yet real-world deployments involve agents with heterogeneous sensors and models, introducing a domain gap. Existing methods fall into two categories:

**Adaptation-based methods** (e.g., MPDA, PnPDA, STAMP, HEAL): Transform features via adapters or reverse alignment, but require intrusive retraining that disrupts established semantic consistency.

**Reconstruction-based methods** (e.g., CodeFilling): Reconstruct features via shared codebook indices, but incur high computational cost when scaling to new agents.

The core limitations of both categories are: (1) intrusive modification of encoders or core modules undermines inter-agent semantic consistency; (2) onboarding new agents demands substantial computation and parameter overhead, limiting scalability. The paper addresses the following core problem: how to integrate new agents into collaboration at minimal cost while preserving semantic consistency among existing agents?

## Method

### Overall Architecture

The core idea of GenComm is that each ego agent uses received spatial messages to locally **generate** collaborator features, ensuring that the generated features are aligned with the ego semantic space while retaining the collaborator's spatial information. The framework comprises three components:

- **Deformable Message Extractor**: Extracts spatial messages from BEV features.
- **Spatial-Aware Feature Generator**: Generates aligned features via a conditional diffusion model.
- **Channel Enhancer**: Refines generated features along the channel dimension.

Overall pipeline: each agent extracts BEV features with its own encoder → the message extractor compresses them into spatial messages for transmission → the ego agent conditions a diffusion model on the received messages to generate collaborator features → features are refined by the channel enhancer, fused, and decoded.

### Key Designs

**Deformable Message Extractor**: Employs deformable convolution to dynamically attend to neighboring pixels, enhancing foreground/background discrimination. An offset prediction network predicts sampling offsets; weighted deformable convolution extracts spatial information; a learnable resizer handles varying resolutions. The extracted message has dimension $C' \times H_j \times W_j$, substantially smaller than the original intermediate features, reducing communication overhead.

**Spatial-Aware Feature Generator**: A conditional diffusion model adds noise to an initial feature (initialized from ego features), then conditions a U-Net on the received spatial messages to iteratively denoise and generate features aligned with the ego semantic space. The generation process is supervised with an MSE loss:

$$\mathcal{L}_{gen} = \sum_{j \in \mathcal{G}_i} \|\hat{\mathcal{F}}_j - \mathcal{F}_j\|_2^2$$

**Channel Enhancer**: Introduces PConv operations to enhance informative elements, combined with a gating mechanism to suppress redundant channel information and channel attention to emphasize key features. Features are split along the channel dimension into modifiable and static portions, refined via depthwise separable convolution and attention.

### Loss & Training

A two-stage training strategy is adopted:

- **Stage 1 (Homogeneous Training)**: End-to-end training with loss $\mathcal{L}_{stage1} = \alpha_1 \mathcal{L}_{cls} + \alpha_2 \mathcal{L}_{reg} + \alpha_3 \mathcal{L}_{gen}$, where classification uses focal loss, regression uses smooth L1 loss, and generation uses MSE loss.
- **Stage 2 (Heterogeneous Extension)**: Only the lightweight message extractor is fine-tuned to resolve numerical inconsistencies in spatial information across heterogeneous agents. Loss: $\mathcal{L}_{stage2} = \alpha_1 \mathcal{L}_{cls} + \alpha_2 \mathcal{L}_{reg}$.

When onboarding a new agent, only a lightweight extractor needs to be initialized and fine-tuned; the ego core modules remain unmodified.

## Key Experimental Results

### Main Results

| Fusion Network | Method | OPV2V-H LP64-LS32 AP50/AP70 | OPV2V-H LP64-CE AP50/AP70 | DAIR-V2X LP64-LS40 AP30/AP50 | Comm. (log2) |
|---|---|---|---|---|---|
| AttFuse | MPDA | 0.767/0.570 | 0.737/0.574 | 0.425/0.364 | 22.0 |
| AttFuse | BackAlign | 0.787/0.584 | 0.685/0.524 | 0.456/0.373 | 22.0 |
| AttFuse | CodeFilling | 0.722/0.536 | 0.666/0.510 | 0.385/0.319 | 15.0 |
| AttFuse | STAMP | 0.759/0.569 | 0.726/0.561 | 0.447/0.391 | 22.0 |
| AttFuse | **GenComm** | **0.804/0.633** | **0.753/0.601** | **0.459/0.379** | **16.0** |
| V2X-ViT | MPDA | 0.850/0.660 | 0.687/0.502 | 0.472/0.379 | 22.0 |
| V2X-ViT | BackAlign | 0.855/0.693 | 0.691/0.523 | 0.490/0.392 | 22.0 |
| V2X-ViT | CodeFilling | 0.860/0.689 | 0.560/0.416 | 0.445/0.356 | 15.0 |
| V2X-ViT | STAMP | 0.844/0.628 | 0.751/0.544 | 0.542/0.494 | 22.0 |
| V2X-ViT | **GenComm** | **0.867/0.699** | **0.763/0.576** | **0.565/0.467** | **16.0** |

### Ablation Study — Scalability Cost

| Method | Training Parameters for New Agent | FLOPs for New Agent |
|---|---|---|
| MPDA | Baseline | Baseline |
| BackAlign | High | High |
| STAMP | High | High |
| CodeFilling | Moderate | Moderate |
| **GenComm** | **↓81%** | **↓81%** |

### Key Findings

1. GenComm surpasses existing state-of-the-art methods on both simulated (OPV2V-H) and real-world (DAIR-V2X, V2X-Real) datasets.
2. Communication volume is reduced from 22.0 to 16.0 (log2), yielding significant bandwidth savings.
3. The computation and parameter overhead for onboarding new agents is reduced by 81% compared to adaptation-based methods and by 62% compared to reconstruction-based methods.
4. Ablation studies confirm the individual contribution of each component (message extractor, feature generator, channel enhancer).

## Highlights & Insights

- **Paradigm Innovation**: The first work to propose a generation-based (rather than adaptation- or reconstruction-based) heterogeneous collaborative communication mechanism, avoiding intrusive modification of existing networks.
- **Lightweight Scalability**: Onboarding a new agent requires fine-tuning only a small extractor, greatly reducing the admission cost for new participants.
- **Communication Efficiency**: Compressed spatial messages are transmitted instead of full intermediate features, reducing bandwidth requirements.
- **Novel Application of Diffusion Models**: Leveraging a conditional diffusion model on the ego side to "imagine" collaborator features represents a novel use of diffusion models in collaborative perception.

## Limitations & Future Work

- The inference latency of conditional diffusion models may become a bottleneck for real-time systems, necessitating acceleration strategies.
- Validation is currently limited to 3D object detection; extension to other downstream tasks such as semantic segmentation is warranted.
- The two-stage training strategy still requires fine-tuning for each new heterogeneous agent combination; exploring zero-shot heterogeneous collaboration is a promising direction.
- The trade-off between spatial message compression rate and information retention merits further investigation.

## Related Work & Insights

- Unlike adaptation-based methods such as HEAL and STAMP, GenComm does not require defining a shared protocol semantic space.
- Similar to DiffBEV/CoDiff in using diffusion models for BEV feature generation, GenComm innovatively applies this to cross-agent heterogeneous feature translation.
- Insight: Generative approaches may represent a general paradigm for addressing interoperability in multi-modal and multi-architecture systems.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ (First generative collaborative communication paradigm)
- Technical Depth: ⭐⭐⭐⭐ (Diffusion model + deformable convolution design is well-motivated)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (Three datasets, multiple settings, comprehensive ablations)
- Practicality: ⭐⭐⭐⭐ (Excellent scalability, though diffusion inference speed requires further validation)

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Janus-Pro-R1: Advancing Collaborative Visual Comprehension and Generation via Reinforcement Learning](janus-pro-r1_advancing_collaborative_visual_comprehension_and_generation_via_rei.md)
- [\[ICLR 2026\] Bridging Generalization Gap of Heterogeneous Federated Clients Using Generative Models](../../ICLR2026/image_generation/bridging_generalization_gap_of_heterogeneous_federated_clients_using_generative_.md)
- [\[CVPR 2026\] Heterogeneous Decentralized Diffusion Models](../../CVPR2026/image_generation/heterogeneous_decentralized_diffusion_models.md)
- [\[NeurIPS 2025\] Multimodal Generative Flows for LHC Jets](multimodal_generative_flows_for_lhc_jets.md)
- [\[NeurIPS 2025\] Diffusion Generative Modeling on Lie Group Representations](diffusion_generative_modeling_on_lie_group_representations.md)

<!-- RELATED:END -->
