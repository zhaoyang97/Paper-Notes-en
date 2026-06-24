---
title: >-
  [Paper Note] Low-Rank Interconnected Adaptation across Layers
description: >-
  [ACL 2025][Model Compression][LoRA] Lily (Low-rank Interconnected Adaptation across Layers) is proposed, which decouples and interconnects/shares LoRA's A/B adapters across layers, combined with a data-dependent routing mechanism, to achieve high-rank weight updates with equivalent or fewer parameters, consistently outperforming LoRA across multimodal, multi-architecture, and multi-scale scenarios.
tags:
  - "ACL 2025"
  - "Model Compression"
  - "LoRA"
  - "parameter-efficient fine-tuning"
  - "cross-layer adaptation"
  - "Mixture-of-Experts"
  - "high-rank update"
date: 2026-05-08
content_hash: c179a3ccfd13042b
---

# Low-Rank Interconnected Adaptation across Layers

**Conference**: ACL 2025  
**arXiv**: [2407.09946](https://arxiv.org/abs/2407.09946)  
**Code**: [Yes](https://github.com/) (The GitHub link is annotated in the paper)  
**Area**: Others  
**Keywords**: LoRA, parameter-efficient fine-tuning, cross-layer adaptation, Mixture-of-Experts, high-rank update

## TL;DR

Lily (Low-rank Interconnected Adaptation across Layers) is proposed, which decouples and interconnects/shares LoRA's A/B adapters across layers, combined with a data-dependent routing mechanism, to achieve high-rank weight updates with equivalent or fewer parameters, consistently outperforming LoRA across multimodal, multi-architecture, and multi-scale scenarios.

## Background & Motivation

LoRA is the most popular Parameter-Efficient Fine-Tuning (PEFT) method, which approximates the weight update $\Delta W = AB$ via low-rank projection matrices A and B. However, LoRA has a fundamental limitation: **A and B are tightly coupled within each layer**, and the parameter budget is uniformly allocated to each layer, which restricts the rank of weight updates per layer.

Core Problem: Can a more dynamic and expressive high-rank weight update be achieved under the same parameter budget?

Observation shows that LoRA allocates the same budget to each layer, ignoring differences in layer importance. If the number of adapters is reduced while their rank is increased, a higher-rank update can be achieved without changing the total parameters. The key lies in how a small number of large-rank adapters can effectively serve all layers.

## Method

### Overall Architecture

Lily deconstructs the tight intra-layer coupling of traditional LoRA's A-B into a cross-layer interconnected structure:

- **A adapter** (down-projection): Locally shared among adjacent layers, with its quantity being less than the number of model layers.
- **B adapter** (up-projection): Globally shared, allowing all layers to access all B experts.
- **Router R**: Data-dependent gating mechanism that decides which B experts to use for each layer and their corresponding weights.

### Key Designs

#### 1. Down-Projection and Selective Weight Allocation

**Function**: Projects the input into a low-dimensional space using the shared A, and then dynamically selects the weight distribution of B experts via the router.

**Mechanism**:

- The input $x$ passes through the locally shared A to obtain a low-dimensional representation $x' = xA$.
- The router $R \in \mathbb{R}^{N_e \times d}$ calculates the weight distribution of B experts based on $x'$:

$$S = \text{softmax}\left(\sum_{i=1}^{N}(x'R^T)_i\right)$$

**Design Motivation**: The router makes the A-B connection data-driven. Different inputs activate different combinations of B experts, preventing B expert behaviors from homogenizing and thereby enhancing expressiveness.

#### 2. Weighted Mixture-of-Experts and Up-Projection

**Function**: Mixes the outputs of multiple B experts according to routing weights to obtain the final weight update.

**Mechanism**: A mathematically equivalent but highly efficient implementation—mixing B before projection to avoid calculating for each B separately:

$$x_\Delta = x'\left(\sum_{i=1}^{N_e} S_i \cdot B^i\right)$$

Final output: $y = xW_0 + s \cdot x_\Delta$

**Design Motivation**: Since $S_i$ is a scalar, mixing the B matrices first before performing matrix multiplication maintains a computational cost comparable to a single LoRA while significantly increasing the rank.

#### 3. Parameter Efficiency and High-Rank Update

**Key Insight**: Traditional LoRA uses one pair of (A, B) per layer, each with rank r. Lily uses a small number of shared adapters (e.g., 2 A's, 4 B's) where each rank can be set larger (e.g., 32), achieving a higher effective update rank with fewer total parameters.

### Loss & Training

Lily follows standard training strategies of various baseline tasks and does not introduce extra loss terms. The scaling factor s controls the magnitude of the adaptive update's impact on the original weights.

## Key Experimental Results

### Main Results

**Common Sense Reasoning (LLaMA3-8B, average accuracy across 8 tasks)**:

| Method | Parameters | BoolQ | PIQA | SIQA | HellaSwag | WinoGrande | ARC-e | ARC-c | OBQA | Avg. |
|------|--------|-------|------|------|-----------|------------|-------|-------|------|------|
| LoRA | 56M | 70.8 | 85.2 | 79.9 | 91.7 | 84.3 | 84.2 | 71.2 | 79.0 | 80.8 |
| MiLoRA | 56.6M | 68.8 | 86.7 | 77.2 | 92.9 | 85.6 | 86.8 | 75.5 | 81.8 | 81.9 |
| **Lily** | **1.2M** | 72.9 | 85.6 | 77.8 | 92.7 | 83.3 | 89.7 | 77.6 | 82.8 | **82.8** |

Note: Lily uses only **1.2M** parameters (1/46 of LoRA), yet its average accuracy outperforms LoRA by 2 percentage points.

**Natural Language Understanding (RoBERTa-Base, GLUE 6 tasks)**:

| Method | Parameters | SST-2 | MRPC | CoLA | QNLI | RTE | STS-B | Avg. |
|------|--------|-------|------|------|------|-----|-------|------|
| LoRA | 0.3M | 94.8 | 89.8 | 63.3 | 92.9 | 78.2 | 91.5 | 85.1 |
| AdaLoRA | 0.3M | 94.5 | 88.7 | 62.0 | 93.1 | 81.0 | 90.5 | 85.0 |
| **Lily** | **0.3M** | **95.0** | **90.2** | **66.0** | 92.5 | **81.6** | 90.8 | **86.0** |

### Ablation Study

**Falcon-Mamba-7B Common Sense Reasoning**:

| Method | Parameters | Avg. |
|------|--------|------|
| LoRA (3.7M) | 3.7M | 32.7 |
| Lily (Δ + in) | 3.7M | 57.0 |
| Lily (in) | 3.3M | 59.5 |

Lily also significantly outperforms LoRA on the Mamba architecture, demonstrating its cross-architecture generalization capabilities.

### Key Findings

1. **Extreme Parameter Efficiency**: On LLaMA3-8B, Lily template outperforms LoRA (56M parameters) using only 1.2M parameters, yielding a 46-fold improvement in parameter efficiency.
2. **Cross-Architecture Generalization**: Lily significantly outperforms LoRA across Transformers (LLaMA3, RoBERTa), Mamba (Falcon-Mamba), and Diffusion Models (SDXL).
3. **Cross-Modal Effectiveness**: Covers NLU, common sense reasoning, image generation, and visual adaptation (VTAB-1K).
4. **Crucial Routing Mechanism**: Data-dependent routing prevents B expert behaviors from degenerating into homogeneity, ensuring diverse knowledge combinations.
5. **High-Rank Update is the Key Source of Benefit**: Reducing the number of adapters and increasing their rank is more effective than increasing the number of adapters.

## Highlights & Insights

- **Deep Core Insight**: The bottleneck of LoRA lies not in the method itself, but in how the parameter budget is distributed—uniform allocation limits the rank of each layer.
- **Simple and Efficient Design**: Lily does not introduce extra loss, modify pre-trained weights, or increase inference latency (B can be pre-mixed).
- **Exquisite Application of MoE Concept**: Treating B as experts is natural, and weighted mixture is completed at the scalar level, avoiding the routing overhead of traditional MoE.
- **Eliminating Redundancy**: Traditional LoRA has significant redundancy with one AB pair per layer; Lily proves that sharing + interconnection can substantially reduce redundancy.

## Limitations & Future Work

1. The router adds a small amount of parameters and computational overhead, which might not be suitable for extremely resource-constrained scenarios.
2. The optimal grouping strategy for sharing A (which layers share one A) requires heuristic design and lacks an automatic search mechanism.
3. Choosing the number of B experts (Ne) affects performance but lacks theoretical guidance.
4. Image generation experiments in the paper only show qualitative results and lack quantitative metrics (such as FID or CLIP Score).
5. It was not compared with newer PEFT methods (such as DoRA, GaLore).

## Related Work & Insights

- **LoRA-based Improvements**: PiSSA (principal subspace initialization), MiLoRA (minor component initialization), AdaLoRA (adaptive rank allocation), etc., focus on rank utilization but maintain tight intra-layer A-B coupling.
- **MoE + PEFT**: MoLORA and MOLA treat the entire LoRA as an expert, whereas Lily decouples A and B into experts at different levels.
- **HydraLoRA**: Concurrent work that also explores asymmetric design, but only within a single layer, whereas Lily achieves global cross-layer interconnection.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — The idea of cross-layer decoupling of A/B and interconnecting them is highly novel, fundamentally changing the parameter distribution paradigm of LoRA.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Covers NLU/reasoning/generation/vision multi-modalities and multi-architectures, but lacks quantitative evaluation in image generation.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear motivation, complete mathematical formulation, and intuitive diagrams (Fig.1 explains the core idea exceptionally well).
- **Value**: ⭐⭐⭐⭐⭐ — High practical value due to extreme parameter efficiency and cross-architecture versatility, making it a promising candidate for a next-generation PEFT baseline.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] TeamLoRA: Boosting Low-Rank Adaptation with Expert Collaboration and Competition](teamlora_boosting_low-rank_adaptation_with_expert_collaboration_and_competition.md)
- [\[ACL 2025\] CoLA: Collaborative Low-Rank Adaptation](cola_collaborative_low-rank_adaptation.md)
- [\[ACL 2025\] BeamLoRA: Beam-Constraint Low-Rank Adaptation](beamlora_beam_constraint_lora.md)
- [\[NeurIPS 2025\] GoRA: Gradient-Driven Adaptive Low Rank Adaptation](../../NeurIPS2025/model_compression/gora_gradient-driven_adaptive_low_rank_adaptation.md)
- [\[ACL 2025\] MoRE: A Mixture of Low-Rank Experts for Adaptive Multi-Task Learning](more_a_mixture_of_low-rank_experts_for_adaptive_multi-task_learning.md)

</div>

<!-- RELATED:END -->
