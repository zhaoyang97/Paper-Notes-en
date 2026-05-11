---
title: >-
  [Paper Note] On Token's Dilemma: Dynamic MoE with Drift-Aware Token Assignment for Continual Learning of Large Vision Language Models
description: >-
  [CVPR 2026][Multimodal VLM][Continual Learning] This paper identifies the "token's dilemma" in dynamic MoE continual learning — ambiguous and old tokens in new-task data contribute minimally to new knowledge acquisition yet cause routing drift and catastrophic forgetting. The proposed LLaVA-DyMoE mitigates routing drift via Token Assignment Guidance and Routing Score Regularization, achieving over 7% MFN improvement and 12% forgetting reduction on the CoIN benchmark.
tags:
  - CVPR 2026
  - Multimodal VLM
  - Continual Learning
  - Mixture of Experts
  - Routing Drift
  - Large Vision-Language Models
  - Token Assignment
date: 2026-05-08
content_hash: b30c749b17cad5b8
---

# On Token's Dilemma: Dynamic MoE with Drift-Aware Token Assignment for Continual Learning of Large Vision Language Models

**Conference**: CVPR 2026
**arXiv**: [2603.27481](https://arxiv.org/abs/2603.27481)
**Code**: [https://zhaoc5.github.io/DyMoE](https://zhaoc5.github.io/DyMoE)
**Area**: Multimodal VLM / Continual Learning
**Keywords**: Continual Learning, Mixture of Experts, Routing Drift, Large Vision-Language Models, Token Assignment

## TL;DR

This paper identifies the "token's dilemma" in dynamic MoE continual learning — ambiguous and old tokens in new-task data contribute minimally to new knowledge acquisition yet cause routing drift and catastrophic forgetting. The proposed LLaVA-DyMoE mitigates routing drift via Token Assignment Guidance and Routing Score Regularization, achieving over 7% MFN improvement and 12% forgetting reduction on the CoIN benchmark.

## Background & Motivation

1. **Background**: Large Vision-Language Models (LVLMs) such as LLaVA demonstrate strong performance across diverse vision-language tasks. Multimodal Continual Instruction Tuning (MCIT) aims to enable LVLMs to incrementally acquire new tasks while retaining performance on prior ones. Mixture-of-Experts (MoE) architectures have become a mainstream approach for MCIT due to their dynamic modularity and parameter isolation properties.

2. **Limitations of Prior Work**: (1) Fixed-size MoE with shared experts and routers leads to cross-task interference; (2) Incrementally expanding MoE while freezing old experts achieves parameter isolation but still suffers from **routing drift**: old-task tokens are erroneously attracted to new experts, causing forgetting; (3) Existing methods circumvent this via task-level routers, but at the cost of heavy task identification overhead and sacrificing the inherent token-level routing flexibility of MoE.

3. **Key Challenge**: Even when old experts and routing parameters are frozen, training newly added components still induces forgetting. The fundamental issue is that **not all new-task tokens carry genuinely new patterns** — some resemble old-task patterns or exhibit ambiguous affinity between old and new experts.

4. **Goal**: (1) Precisely analyze the token-level causes of routing drift; (2) Design differentiated routing strategies for distinct token types to mitigate forgetting.

5. **Key Insight**: Through controlled two-task experiments, tokens are categorized into three types (new/old/ambiguous) based on their routing score distributions toward new versus old expert groups. Ambiguous tokens are identified as the primary culprit of routing drift — they neither facilitate new-task learning nor protect old knowledge, yet train the router to attract old-task patterns when routed to new experts.

6. **Core Idea**: Identify token types and steer ambiguous/old tokens away from new experts, while applying soft regularization to encourage exclusive token-to-expert-group routing and specialization of new experts.

## Method

### Overall Architecture

LLaVA-DyMoE augments each layer of the LLM backbone in LLaVA with MoE (LoRA experts). Upon arrival of a new task, new experts and routing parameters are added while old components are frozen, and only the newly added parts are trained. The core innovation consists of two regularization mechanisms: Token Assignment Guidance (TAG), which identifies token types during training and adjusts routing scores to prevent drift; and Routing Score Regularization (RSR), which further enforces exclusive routing and new-expert specialization.

### Key Designs

1. **Token Type Analysis and the Token's Dilemma**:

    - **Function**: Reveals the token-level causes of routing drift.
    - **Mechanism**: In controlled two-task experiments, new-task tokens are classified into three types based on their relative routing score advantage toward new versus old expert groups: **new tokens** (high affinity for the new group) → primarily drive new knowledge acquisition with minimal forgetting; **old tokens** (high affinity for the old group) → contribute little to new tasks, but residual affinity for new experts contaminates the router; **ambiguous tokens** (small affinity difference between new and old groups) → neither facilitate new-task learning nor protect old knowledge, directly inducing forgetting risk.
    - **Design Motivation**: To precisely locate the source of forgetting. The "token's dilemma" refers to the fact that these tokens carry the least learning value, yet without guidance cause the greatest forgetting cost through router training signals.

2. **Token Assignment Guidance (TAG)**:

    - **Function**: Identifies token types during training and guides each token to the appropriate expert group.
    - **Mechanism**: For each token, the maximum routing scores over old and new expert groups are extracted as $c_\text{old} = \max(\mathbf{s}_{t-1})$ and $c_\text{new} = \max(\mathbf{s}_{t,\text{new}})$, and the relative difference is computed as $D_\text{rel} = \frac{|c_\text{new} - c_\text{old}|}{\max(|c_\text{new}|, |c_\text{old}|) + \epsilon}$. A token is routed to new experts only when it is non-ambiguous ($D_\text{rel} > \tau$) and the new group dominates ($c_\text{new} > c_\text{old}$); otherwise, the new-group routing scores are set to $-\infty$, forcing routing to the old group. This ensures that ambiguous and old tokens are safely directed to old experts.
    - **Design Motivation**: Directly resolves the dilemma at the token assignment level — allowing new tokens to naturally flow to new experts for new knowledge acquisition, while preventing ambiguous/old tokens from contaminating the new router.

3. **Routing Score Regularization (RSR)**:

    - **Function**: Complements TAG with soft regularization to reinforce exclusive routing between expert groups and promote new-expert utilization.
    - **Mechanism**: Two loss terms are employed. The **exclusivity loss** $\mathcal{L}_\text{exc} = g_\text{old} \cdot g_\text{new}$ penalizes tokens that simultaneously activate both expert groups, promoting a binary routing decision. The **specialization loss** $\mathcal{L}_\text{spe}$ uses $y = 1 - \max\{w_i\}_{i \in \mathcal{S}_{t-1}}$ as a soft target and encourages higher routing weight $g_\text{new}$ to new experts via BCE. Together, they balance stability (anti-forgetting) and plasticity (new knowledge acquisition).
    - **Design Motivation**: TAG acts as a hard constraint (masking), while RSR acts as a soft constraint (gradient signal); the two are complementary and cover different aspects of the routing score space.

### Loss & Training

The overall objective is: $\mathcal{L} = \mathcal{L}_\text{NTP} + \lambda\mathcal{L}_\text{aux} + \alpha(\mathcal{L}_\text{exc} + \mathcal{L}_\text{spe})$

where $\mathcal{L}_\text{NTP}$ is the standard autoregressive cross-entropy loss, $\mathcal{L}_\text{aux}$ is the standard load-balancing loss applied only to new experts, and $\alpha$ controls the regularization strength. TAG and RSR are active only during training and impose no additional constraints at inference, enabling seamless integration with other MCIT methods. The backbone is an instruction-tuning-free LLaVA-v1.5-7B, with new LoRA experts added per new task.

## Key Experimental Results

### Main Results

Comparison on the CoIN benchmark (sequential learning over 8 VQA tasks):

| Method | MFN↑ | MAA↑ | BWT↑ |
|--------|------|------|------|
| LoRA | 41.79 | 43.99 | -23.12 |
| MoELoRA | 43.93 | 43.92 | -22.18 |
| O-LoRA | 49.53 | 46.65 | -17.54 |
| IncMoELoRA | 49.68 | 49.50 | -16.67 |
| **LLaVA-DyMoE** | **57.03** | **57.70** | **-4.67** |

### Ablation Study

| Configuration | MFN↑ | MAA↑ | BWT↑ |
|---------------|------|------|------|
| IncMoELoRA (baseline) | 49.68 | 49.50 | -16.67 |
| + $\mathcal{L}_\text{aux}$ | 50.76 | 51.17 | -15.44 |
| + TAG | 54.44 | 52.18 | -7.04 |
| + $\mathcal{L}_\text{exc}$ | 55.18 | 54.38 | -6.83 |
| + $\mathcal{L}_\text{spe}$ (full) | 57.03 | 57.70 | -4.67 |

Effect of ambiguity threshold $\tau$:

| $\tau$ | MFN↑ | BWT↑ |
|--------|------|------|
| 10% | 56.87 | -4.94 |
| **20%** | **57.03** | **-4.67** |
| 30% | 56.27 | -5.21 |
| 50% | 55.32 | -5.54 |

### Key Findings

- TAG is the most critical component: adding TAG alone improves BWT from -15.44 to -7.04 and MFN from 50.76 to 54.44 (+3.68%).
- The exclusivity loss and specialization loss contribute additional MFN gains of 0.74% and 1.85%, respectively.
- The optimal ambiguity threshold is $\tau=20\%$; too low (10%) leaves some ambiguous tokens uncaptured, while too high (50%) excessively restricts new-expert learning.
- LLaVA-DyMoE achieves the best or second-best final accuracy on all 8 individual tasks.
- Gains are particularly pronounced on the ImageNet task (95.80% vs. 68.42% for IncMoELoRA), likely because its visual features diverge most from other VQA tasks, making it most susceptible to routing drift.

## Highlights & Insights

- **The discovery of the token's dilemma is highly valuable**: It refines the stability-plasticity dilemma in continual learning to the token granularity, identifying ambiguous tokens as the primary source of forgetting. This insight transfers to all dynamic MoE expansion methods.
- **Analysis-driven method design**: Controlled experiments first establish causal relationships, which then motivate targeted solutions. The three-group masking experiments clearly demonstrate the distinct roles of the three token types, providing principled justification for the design choices.
- **Training-time regularization with zero inference overhead**: TAG and RSR affect routing scores only during training; no constraints are applied at inference, preserving efficiency and enabling orthogonal combination with any other MCIT method.
- The paradigm of "analyzing token routing distributions → targeted regularization" generalizes to other MoE scenarios such as multi-task learning and domain adaptation.

## Limitations & Future Work

- Evaluation is limited to the CoIN benchmark (8 VQA tasks); generalization to more diverse task types (e.g., generation, detection) remains unverified.
- Adding new experts per task leads to linear parameter growth, raising efficiency concerns for long-term expansion.
- Token type classification relies entirely on instantaneous routing score snapshots, which may be inaccurate during early training.
- The dynamic evolution of token types throughout training is unexplored — ambiguous tokens may gradually transition to new or old tokens as training progresses.
- Combination experiments with recent task-level routing methods (e.g., ProgLoRA) are absent, leaving the claimed orthogonality unverified.

## Related Work & Insights

- **vs. MoELoRA**: Shared routers and experts cause severe cross-task interference (MFN 43.93 vs. 57.03); LLaVA-DyMoE fundamentally addresses this through parameter isolation combined with routing regularization.
- **vs. IncMoELoRA**: Incremental expansion achieves parameter isolation but still suffers from routing drift (BWT -16.67 vs. -4.67), demonstrating that freezing old parameters alone is insufficient — active management of new parameter training is necessary.
- **vs. O-LoRA**: O-LoRA enforces orthogonality constraints in the LoRA subspace, while LLaVA-DyMoE applies constraints at the routing level; the two approaches are complementary.
- **vs. ProgLoRA**: ProgLoRA mitigates interference via a progressive LoRA pool and task isolation, while LLaVA-DyMoE performs fine-grained token-level management; the two can be combined.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ The discovery and analysis of the token's dilemma are thorough; the three-way token categorization is both intuitive and empirically supported.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Ablations are comprehensive with stepwise validation of each component and sufficient hyperparameter analysis, though evaluation is limited to a single benchmark.
- **Writing Quality**: ⭐⭐⭐⭐⭐ The analysis-driven paper structure is exceptionally clear; the controlled experiment design and visualizations are excellent.
- **Value**: ⭐⭐⭐⭐⭐ Provides deep mechanistic insight into MoE continual learning with a concise, effective, and generalizable method.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Dynamic Token Reweighting for Robust Vision-Language Models](dynamic_token_reweighting_for_robust_vision-language_models.md)
- [\[CVPR 2026\] Quant Experts: Token-aware Adaptive Error Reconstruction with Mixture of Experts for Large Vision-Language Models Quantization](quant_experts_token_aware_vlm_quantization.md)
- [\[CVPR 2026\] Variation-Aware Vision Token Dropping for Faster Large Vision-Language Models](variation-aware_vision_token_dropping_for_faster_large_vision-language_models.md)
- [\[CVPR 2026\] DTR: Dynamic Token Reweighting for Robust Vision-Language Models](dynamic_token_reweighting_for_robust_vision-language_models.md)
- [\[CVPR 2026\] V2Drop: Variation-aware Vision Token Dropping for Faster Large Vision-Language Models](variation-aware_vision_token_dropping_for_faster_large_vision-language_models.md)

</div>

<!-- RELATED:END -->
