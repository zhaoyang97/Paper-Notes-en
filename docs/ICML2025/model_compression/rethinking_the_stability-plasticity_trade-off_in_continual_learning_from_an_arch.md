---
title: >-
  [Paper Note] Rethinking the Stability-Plasticity Trade-off in Continual Learning from an Architectural Perspective
description: >-
  [ICML2025][Model Compression][Continual Learning] This paper reveals an inherent conflict between stability and plasticity at the **architectural level** in continual learning: wide-and-shallow networks exhibit better stability, whereas deep-and-narrow networks possess stronger plasticity. Consequently, the authors propose the Dual-Arch framework, which delegates stability and plasticity to two dedicated lightweight architectures and coordinates them via knowledge distillatio…
tags:
  - "ICML2025"
  - "Model Compression"
  - "Continual Learning"
  - "Stability-Plasticity Trade-off"
  - "Network Architecture Design"
  - "Knowledge Distillation"
  - "Dual-Architecture Framework"
date: 2026-05-08
content_hash: fbaa6008915d5fe8
---

# Rethinking the Stability-Plasticity Trade-off in Continual Learning from an Architectural Perspective

**Conference**: ICML2025  
**arXiv**: [2506.03951](https://arxiv.org/abs/2506.03951)  
**Code**: [byyx666/Dual-Arch](https://github.com/byyx666/Dual-Arch)  
**Area**: Continual Learning  
**Keywords**: Continual Learning, Stability-Plasticity Trade-off, Network Architecture Design, Knowledge Distillation, Dual-Architecture Framework

## TL;DR

This paper reveals an inherent conflict between stability and plasticity at the **architectural level** in continual learning: wide-and-shallow networks exhibit better stability, whereas deep-and-narrow networks possess stronger plasticity. Consequently, the authors propose the Dual-Arch framework, which delegates stability and plasticity to two dedicated lightweight architectures and coordinates them via knowledge distillation. This achieves up to an 87% reduction in parameter count while simultaneously improving CL performance.

## Background & Motivation

The core challenge of continual learning (CL) is **catastrophic forgetting**: when learning new tasks, networks rapidly forget old knowledge. Existing methods (replay, regularization, architecture expansion) balance stability and plasticity primarily at the **parameter level**, yet neglect the influence of the network architecture itself on this trade-off.

Prior pioneering work (Mirzadeh et al., 2022) indicates that wide-and-shallow networks generally perform better in CL (rendering stability), whereas deep networks are stronger in representation learning (favoring plasticity). This raises a critical question: **under a given parameter constraint, does an inherent conflict between stability and plasticity also exist at the architectural level?**

The authors validate this hypothesis by comparing ResNet-18 with its iso-parameter wide-and-shallow counterpart:

- **ResNet-18** (deep-and-narrow): Higher accuracy on new tasks (good plasticity) but more severe forgetting
- **Wide-and-shallow variant** (10 layers, 96 channels): Lower forgetting (good stability) but diminished performance on new tasks

This suggests that a single architecture cannot optimize both objectives simultaneously, and the common practice of using a unified architecture inherently limits the performance ceiling of CL.

## Method

### Core Idea: Dual-Arch Framework

Dual-Arch is a **plug-and-play** CL framework. The core mechanism is to decouple stability and plasticity into two independent and dedicated networks:

- **Pla-Net (Plasticity Network)**: A deep-and-narrow architecture focused on learning new knowledge
- **Sta-Net (Stability Network)**: A wide-and-shallow architecture responsible for retaining old knowledge and integrating new knowledge

### Architecture Design (Based on Modified ResNet-18)

| Network | Depth | Width | Special Design | Parameter Size |
|------|------|------|----------|--------|
| ResNet-18 (Original) | 18 layers | 64 channels | GAP → 1×1 | 11.23M |
| Sta-Net | Half the residual blocks | 64 channels | AvgPool → 2×2 output | ~7M |
| Pla-Net | 18 layers | 42 channels | Keep original structure | ~7M |

The sum of parameters of the two networks (~15M) remains smaller than the typical expanded baseline (~22M), achieving parameter compression.

### Training Process

For each new task $k$, training is performed sequentially in two stages:

**Stage One: Training Pla-Net**  
Optimize the current task data purely with classification loss, without considering forgetting:

$$L_{plastic} = L_{CE}(x, y; \phi_k) = -\log \frac{\exp(o_y)}{\sum_{m=1}^{N^k} \exp(o_m)}$$

**Stage Two: Training Sta-Net**  
Freeze Pla-Net as a teacher model, and transfer new knowledge to Sta-Net via knowledge distillation while using CL methods to retain old knowledge:

$$L_{stable} = \alpha \cdot L_{CE} + (1-\alpha) \cdot L_{KD} + L_{CL}$$

Where $\alpha=0.5$ balances the hard label loss and distillation loss, and $L_{CL}$ is the loss item of the corresponding CL method itself.

The knowledge distillation loss $L_{KD}$ computes the KL divergence between the soft targets of the teacher (Pla-Net) and the student (Sta-Net):

$$L_{KD} = -\sum_{i=1}^{N^k} P_T^i \log P_S^i, \quad P_T = \text{SoftMax}(O_T / t), \quad P_S = \text{SoftMax}(O_S / t)$$

Where $t$ is the temperature factor controlling the smoothness of soft outputs.

**Inference Stage**: Only Sta-Net is used; Pla-Net does not participate in inference, thereby minimizing inference overhead.

## Key Experimental Results

### Main Results: Combination with 5 SOTA CL Methods (Tab. 2)

On CIFAR-100 and ImageNet-100 (10/20 task splits), Dual-Arch consistently improves all baseline methods as a plug-and-play module:

| Method | Parameter Reduction | Max LA Gain | Max AIA Gain | Forgetting Reduction |
|------|-----------|------------|-------------|---------|
| iCaRL + Dual-Arch | ↓33% | +3.10% | +2.17% | ↓7.69% |
| WA + Dual-Arch | ↓33% | +8.24% | +6.09% | ↓7.32% |
| DER + Dual-Arch | ↓52% | +5.69% | +3.67% | ↓5.55% |
| Foster + Dual-Arch | ↓32% | +7.70% | +7.62% | ↓11.28% |
| MEMO + Dual-Arch | ↓41% | +10.29% | +5.09% | ↓11.62% |

MEMO + Dual-Arch delivers the most notable performance, improving LA by 10.29%, reducing FAF (forgetting) by 11.62%, and shrinking parameter size by 41%.

### Ablation Study (Tab. 3, CIFAR-100/10, AIA)

| Configuration | Average AIA | Gap with Full Method |
|------|---------|--------------|
| Sta-Net + Pla-Net (Full) | 72.92% | — |
| Sta-Net only (No auxiliary network) | 70.29% | -2.63% |
| Pla-Net + Pla-Net (Unified architecture) | 71.18% | -1.74% |
| Sta-Net + Sta-Net (Unified architecture) | 72.27% | -0.65% |
| Pla-Net + Sta-Net (Role reversal) | 71.24% | -1.68% |

Ablation results demonstrate that: (1) dual-network collaboration outperforms a single network by 2.63%; (2) dedicated architectures outperform unified ones by 0.65% to 1.74%; (3) role assignments are non-interchangeable.

### Parameter Efficiency

Even under an extreme parameter reduction of **87%**, Dual-Arch outperforms the original baseline (with an AIA increase of +0.90% on DER and +1.94% on Foster).

### Computational Efficiency

| Metric | Sta-Net | Pla-Net | Total | ResNet-18 |
|------|---------|---------|------|-----------|
| FLOPs | 255M | 241M | 496M | 558M |

Total training FLOPs are reduced, and during inference, only Sta-Net is utilized (255M vs 558M), dropping computational overhead by 54%. However, training time is prolonged by 1.39× to 1.77× due to sequential optimization.

## Highlights & Insights

1. **Novel Perspective**: For the first time, this work systematically reveals the inherent stability-plasticity conflict in CL at the architectural level, extending the trade-off from parameter dimensions to architectural dimensions.
2. **Plug-and-Play**: Dual-Arch can be seamlessly integrated into mainstream methods like iCaRL, WA, DER, Foster, and MEMO, showcasing high generalizability.
3. **Parameter Efficient**: It achieves superior performance with fewer parameters, avoiding simple stacking of two large models and instead substituting a single large network with two dedicated lightweight ones.
4. **Zero Inference Overhead**: In the inference stage, only Sta-Net is used, requiring only 46% of the original FLOPs.
5. **Rigorous Experimental Design**: Diverse methods execution across multiple datasets and task splits, with thorough ablation studies validating the contribution of each component.

## Limitations & Future Work

1. **Increased Training Time**: Sequential training of two networks introduces a 1.39× to 1.77× training overhead, where the inability to parallelize remains a major bottleneck.
2. **Heuristics-Dependent Architecture Design**: The specific designs (layer/channel count) of Sta-Net/Pla-Net are manually adjusted based on ResNet, lacking automated search.
3. **Validation Limited to ResNet**: Not yet evaluated on modern architectures such as ViTs and ConvNeXts; the generalizability of the "deep-narrow = plastic, wide-shallow = stable" conclusion is yet to be fully validated.
4. **Limited to Class-IL Scenario**: Evaluation on Task-IL and Domain-IL scenarios is lacking.
5. **Simplistic Distillation Strategy**: Only logit-level KD is utilized, leaving feature-level, relation-level, or other advanced KD schemes unexplored.
6. **Default Values for Hyperparameters**: The temperature factor $t$ and balance coefficient $\alpha$ ($\alpha=0.5$) are set to default values without comprehensive hyperparameter search or sensitivity analysis.

## Related Work & Insights

- **ArchCraft** (Lu et al., 2024): A single-architecture optimization scheme. Dual-Arch outperforms it in most settings.
- **MKD / Hare & Tortoise**: Dual-model approaches but with identical architectures. This work highlights the critical importance of utilizing dedicated architectures.
- **DER / MEMO**: Architecture expansion methods. Dual-Arch can build upon them to yield further performance gains.
- **Insight**: The paradigm of "functional decoupling + specialized architectures" can be generalized to federated learning, multi-task learning, and other contexts.

## Rating

- Novelty: ⭐⭐⭐⭐ — Mapping the stability-plasticity trade-off to the architectural level is a fresh and valuable research direction.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Comprehensive coverage across 5 methods, 4 benchmarks, ablation study, and efficiency analysis.
- Writing Quality: ⭐⭐⭐⭐ — Logically coherent, fluid transition from observation → problem definition → method formulation → validation.
- Value: ⭐⭐⭐⭐ — The plug-and-play framework offers practical utility, though generalization to modern architectures remains to be validated.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Null-Space Filtering for Data-Free Continual Model Merging: Preserving Stability, Promoting Plasticity](../../ICLR2026/model_compression/null-space_filtering_for_data-free_continual_model_merging_preserving_stability_.md)
- [\[ICML 2025\] BECAME: BayEsian Continual Learning with Adaptive Model MErging](became_bayesian_continual_learning_with_adaptive_model_merging.md)
- [\[ICLR 2026\] Rethinking Continual Learning with Progressive Neural Collapse](../../ICLR2026/model_compression/rethinking_continual_learning_with_progressive_neural_collapse.md)
- [\[ICLR 2026\] DTO-KD: Dynamic Trade-off Optimization for Effective Knowledge Distillation](../../ICLR2026/model_compression/dto-kd_dynamic_trade-off_optimization_for_effective_knowledge_distillation.md)
- [\[NeurIPS 2025\] When Worse is Better: Navigating the Compression-Generation Trade-off in Visual Tokenization](../../NeurIPS2025/model_compression/when_worse_is_better_navigating_the_compression-generation_tradeoff_in_visual_to.md)

</div>

<!-- RELATED:END -->
