---
title: >-
  [Paper Note] Beyond Higher Rank: Token-wise Input-Output Projections for Efficient Low-Rank Adaptation
description: >-
  [NeurIPS 2025][Model Compression][LoRA] TopLoRA analyzes the expressive capacity of LoRA from an input-output projection perspective, identifying that all tokens sharing a single projection matrix constitutes a critical bottleneck. It proposes dynamically adjusting LoRA weights via a learnable token-wise diagonal matrix $\Sigma_X$ (i.e., $\Delta W_X = B\Sigma_X A$), achieving fine-grained adaptation without increasing rank, and consistently outperforming LoRA by 2–3% across tasks.
tags:
  - NeurIPS 2025
  - Model Compression
  - LoRA
  - Low-Rank Adaptation
  - Token-wise Adaptation
  - Input-Output Projection
  - PEFT
date: 2026-05-08
content_hash: fef616351e67d1b6
---

# Beyond Higher Rank: Token-wise Input-Output Projections for Efficient Low-Rank Adaptation

**Conference**: NeurIPS 2025
**arXiv**: [2510.23123](https://arxiv.org/abs/2510.23123)
**Code**: [https://github.com/Leopold1423/toplora-neurips25](https://github.com/Leopold1423/toplora-neurips25)
**Area**: Parameter-Efficient Fine-Tuning
**Keywords**: LoRA, Low-Rank Adaptation, Token-wise Adaptation, Input-Output Projection, PEFT

## TL;DR
TopLoRA analyzes the expressive capacity of LoRA from an input-output projection perspective, identifying that all tokens sharing a single projection matrix constitutes a critical bottleneck. It proposes dynamically adjusting LoRA weights via a learnable token-wise diagonal matrix $\Sigma_X$ (i.e., $\Delta W_X = B\Sigma_X A$), achieving fine-grained adaptation without increasing rank, and consistently outperforming LoRA by 2–3% across tasks.

## Background & Motivation

**State of the Field**: LoRA enables parameter-efficient fine-tuning of large models via low-rank matrices $\Delta W = BA$. Existing improvements primarily focus on increasing rank—HiRA (Hadamard product), MELoRA (mini-ensemble stacking), and MoELoRA (mixture of experts).

**Limitations of Prior Work**: In LoRA, all tokens share the same $\Delta W = BA$, i.e., the same input-output projection matrix $P = R_B L_A$. However, tokens differ substantially in semantics, and identical projection directions may encode entirely different information for different tokens, necessitating distinct processing.

**Root Cause**: Increasing rank expands the dimensionality of the input/output space but incurs linear parameter growth; the expressive bottleneck from shared projections persists regardless of rank. Even at high rank, all tokens remain subject to the same mapping.

**Paper Goals**: Learn distinct input-output projections for each token without increasing the LoRA rank.

**Starting Point**: LoRA is decomposed via QR/LQ factorization into three components—input space $Q_A$, output space $Q_B$, and projection matrix $P = R_B L_A$. Rank determines the dimensionality of these spaces, while $P$ governs the mapping; projection should therefore be token-specific.

**Core Idea**: A lightweight projection network generates a diagonal matrix $\Sigma_X$ from each token $X$, modifying the projection as $P \to P_X = R_B \Sigma_X L_A$, thereby achieving token-adaptive input-output mapping.

## Method

### Overall Architecture
Standard LoRA: $Y = (W + BA)X$. TopLoRA: $Y = (W + B\Sigma_X A)X$, where $\Sigma_X = \text{Diag}(\text{Exp}(\text{RMSNorm}(\Theta X)))$ is an $r \times r$ diagonal matrix dynamically generated from input token $X$, and $\Theta \in \mathbb{R}^{r \times n}$ is a learnable projection parameter.

### Key Designs

1. **Token-wise Diagonal Matrix Generation**:

    - Function: Generates a distinct $r$-dimensional diagonal scaling matrix for each input token.
    - Mechanism: $\Sigma_X = \text{Diag}(\text{Exp}(\text{RMSNorm}(\Theta X)))$. $\Theta X$ projects tokens into an $r$-dimensional space → RMSNorm normalizes away magnitude effects → Exp converts values into positive scaling factors.
    - Design Motivation: RMSNorm prevents $\Sigma_X$ from being influenced by token magnitude or the scale of $\Theta$, amplifying inter-token differences in $\Sigma_X$; Exp avoids near-zero values that would cause information loss, ensuring even subtle normalized differences are magnified.

2. **Relationship to Standard LoRA**:

    - Function: TopLoRA output decomposes into a LoRA base term and a token-adaptive correction term.
    - Mechanism: $\Delta W_X X = BAX + B(\Sigma_X - I)AX$. The first term $BAX$ corresponds to standard LoRA output (global pattern); the second term $B(\Sigma_X - I)AX$ provides a token-specific correction.
    - Design Motivation: When $\Sigma_X = I$, TopLoRA reduces to standard LoRA, maintaining full compatibility.

3. **Kaiming Initialization**:

    - Function: $\Theta$ is initialized with Kaiming initialization rather than zero initialization.
    - Mechanism: Consistent with the initialization strategy of the $A$ matrix, ensuring training stability under a unified learning rate.
    - Design Motivation: Zero initialization causes the initial $\Sigma_X$ to be uniformly 1 (degenerating to LoRA); Kaiming initialization provides meaningful initial diversity.

### Loss & Training
- The downstream task loss is used directly, fully consistent with the standard LoRA training pipeline.
- AdamW optimizer; LoRA dropout of 0.05.
- Parameter analysis: An additional $\Theta \in \mathbb{R}^{r \times n}$ is introduced, amounting to approximately 0.5× the parameter count of LoRA.

## Key Experimental Results

### Main Results

| Task | Model | LoRA-r8 | LoRA-r32 | TopLoRA-r8 | Gain |
|------|-------|---------|----------|-----------|------|
| GLUE (NLU) | RoBERTa-Base | 82.55% | 84.19% | **84.14%** | +1.59% |
| GLUE (NLU) | RoBERTa-Large | 87.06% | 87.75% | **87.64%** | +0.58% |
| Math Reasoning | Gemma-7B | 71.44% | 72.22% | **73.11%** | +1.67% |
| Math Reasoning | LLaMA-3-8B | 64.06% | 64.48% | **66.36%** | +2.30% |
| Commonsense Reasoning | Gemma-7B | 80.22% | 80.43% | **81.10%** | +0.88% |
| Vision-Language | BLIP-2 COCO | B4=40.7 | — | B4=**43.3** | +2.6 |

### Ablation Study

| Variant | GLUE Avg | Note |
|---------|---------|------|
| TopLoRA (full) | **84.14%** | Full model |
| w/o RMSNorm | 83.72% | RMSNorm amplifies inter-token differences |
| w/o Exp | 83.85% | Exp prevents information loss |
| Zero-init $\Theta$ | 83.51% | Initially degenerates to LoRA |
| Fixed $\Sigma_X = I$ (=LoRA) | 82.55% | No token adaptation |

### Key Findings
- TopLoRA-r8 matches LoRA-r32 on GLUE (84.14 vs. 84.19), using ~1.5× the parameters of LoRA-r8 versus 4×.
- Consistent effectiveness across models (RoBERTa/Gemma/LLaMA/BLIP-2) and tasks (NLU/NLG/Vision-Language).
- Gains are most pronounced on mathematical reasoning and vision-language tasks (2–3%), with 1%+ improvements also observed on NLU.
- Outperforms LoRA variants including DoRA, MELoRA, and HydraLoRA.
- TopLoRA can be applied as a plug-in within the MoELoRA framework for further gains.

## Highlights & Insights
- **Novel input-output projection perspective**: Decomposing LoRA into input space + output space + projection matrix reveals that token-shared projection is an overlooked bottleneck. This analytical framework is itself highly informative.
- **Extremely simple and efficient method**: Only a linear projection, RMSNorm, and Exp are added, introducing negligible architectural complexity while achieving consistent cross-task gains.
- **Expressiveness without rank increase**: In contrast to the mainstream paradigm of "higher rank = greater expressiveness," this work demonstrates that rank is not the sole dimension of expressive capacity.

## Limitations & Future Work
- Advantages diminish when the base rank is already very high.
- Additional parameters and computational overhead from the projection network $\Theta$ (small but nonzero).
- The diagonal constraint limits the expressiveness of $\Sigma_X$—a full-matrix $\Sigma_X$ could be more expressive but would incur prohibitive parameter costs.
- Layer-wise behavioral differences of $\Sigma_X$ remain unexplored.

## Related Work & Insights
- **vs. LoRA**: LoRA shares projections across tokens; TopLoRA enables token-level adaptation. TopLoRA-r8 achieves performance comparable to LoRA-r32.
- **vs. DoRA**: DoRA decomposes weights into direction and magnitude; TopLoRA decomposes into space and projection—orthogonal perspectives.
- **vs. HiRA/MELoRA**: These methods increase rank; TopLoRA improves projection without altering rank. The two approaches are orthogonal and composable.
- **vs. MoELoRA**: MoELoRA uses multiple LoRA experts for token adaptation at higher parameter cost; TopLoRA achieves similar adaptivity more parameter-efficiently.

## Rating
- Novelty: ⭐⭐⭐⭐ Novel input-output projection analysis perspective; method is concise and elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Three task categories × multiple models × comprehensive ablations × comparison with multiple baselines.
- Writing Quality: ⭐⭐⭐⭐ Motivation is clearly derived; mathematical derivations are rigorous.
- Value: ⭐⭐⭐⭐ Contributes to both theoretical understanding and practical improvement of the LoRA family.
- Value: ⭐⭐⭐⭐ Introduces a new dimension for LoRA improvement (projection diversity vs. rank); strong practical utility.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] GoRA: Gradient-Driven Adaptive Low Rank Adaptation](gora_gradient-driven_adaptive_low_rank_adaptation.md)
- [\[NeurIPS 2025\] RefLoRA: Refactored Low-Rank Adaptation for Efficient Fine-Tuning of Large Models](reflora_refactored_low-rank_adaptation_for_efficient_fine-tuning_of_large_models.md)
- [\[NeurIPS 2025\] Data Efficient Adaptation in Large Language Models via Continuous Low-Rank Fine-Tuning](data_efficient_adaptation_in_large_language_models_via_continuous_low-rank_fine-.md)
- [\[NeurIPS 2025\] Gated Integration of Low-Rank Adaptation for Continual Learning of Large Language Models](gated_integration_of_low-rank_adaptation_for_continual_learning_of_large_languag.md)
- [\[NeurIPS 2025\] Accurate and Efficient Low-Rank Model Merging in Core Space](accurate_and_efficient_low-rank_model_merging_in_core_space.md)

<!-- RELATED:END -->
